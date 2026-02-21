//===- ForthToMLIR.cpp - Forth to MLIR translation -----------------------===//
//
// This file implements the Forth-to-MLIR translation.
//
//===----------------------------------------------------------------------===//

#include "ForthToMLIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>

using namespace mlir;
using namespace mlir::forth;

/// Mangle a Forth word name into a valid LLVM/PTX identifier.
/// - alphanumeric chars kept as-is
/// - '-' → '_'
/// - '_' → '__'
/// - other → '_XX' (uppercase hex)
/// - leading digit → prepend '_'
static std::string mangleForthName(llvm::StringRef name) {
  std::string result;
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      result += c;
    } else if (c == '-') {
      result += '_';
    } else if (c == '_') {
      result += "__";
    } else {
      result += '_';
      result += llvm::hexdigit((c >> 4) & 0xF, /*UpperCase=*/true);
      result += llvm::hexdigit(c & 0xF, /*UpperCase=*/true);
    }
  }
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

/// Convert a string to uppercase for case-insensitive word matching.
static std::string toUpperCase(llvm::StringRef str) {
  std::string result;
  result.reserve(str.size());
  for (char c : str)
    result += std::toupper(static_cast<unsigned char>(c));
  return result;
}

//===----------------------------------------------------------------------===//
// ForthLexer implementation
//===----------------------------------------------------------------------===//

ForthLexer::ForthLexer(llvm::SourceMgr &sourceMgr, unsigned bufferID)
    : sourceMgr(sourceMgr), bufferID(bufferID) {
  auto buffer = sourceMgr.getMemoryBuffer(bufferID);
  curPtr = buffer->getBufferStart();
  endPtr = buffer->getBufferEnd();
}

void ForthLexer::skipWhitespace() {
  while (curPtr < endPtr) {
    // Skip whitespace characters
    while (curPtr < endPtr && isWhitespace(*curPtr)) {
      ++curPtr;
    }

    // Check for line comment: '\' followed by space/tab or at end of input.
    // Also matches '\' at the start of a line (curPtr == bufferStart or
    // preceded by '\n').
    if (curPtr < endPtr && *curPtr == '\\') {
      const char *next = curPtr + 1;
      if (next >= endPtr || *next == ' ' || *next == '\t' || *next == '\n' ||
          *next == '\r') {
        // Skip to end of line
        while (curPtr < endPtr && *curPtr != '\n') {
          ++curPtr;
        }
        // Skip the newline itself
        if (curPtr < endPtr)
          ++curPtr;
        continue;
      }
    }

    break;
  }
}

bool ForthLexer::isWhitespace(char c) const {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

void ForthLexer::reset() {
  auto buffer = sourceMgr.getMemoryBuffer(bufferID);
  curPtr = buffer->getBufferStart();
  endPtr = buffer->getBufferEnd();
}

void ForthLexer::resetTo(const char *ptr) {
  auto buffer = sourceMgr.getMemoryBuffer(bufferID);
  curPtr = ptr;
  endPtr = buffer->getBufferEnd();
}

bool ForthLexer::isNumber(const std::string &str) const {
  if (str.empty())
    return false;

  size_t start = 0;
  if (str[0] == '-' || str[0] == '+') {
    if (str.length() == 1)
      return false;
    start = 1;
  }

  for (size_t i = start; i < str.length(); ++i) {
    if (!std::isdigit(str[i]))
      return false;
  }

  return true;
}

bool ForthLexer::isFloat(const std::string &str) const {
  if (str.empty())
    return false;

  // Try to parse as a double. A valid float must contain a '.' or 'e'/'E'.
  bool hasDotOrExp = false;
  for (char c : str) {
    if (c == '.' || c == 'e' || c == 'E') {
      hasDotOrExp = true;
      break;
    }
  }
  if (!hasDotOrExp)
    return false;

  char *end = nullptr;
  std::strtod(str.c_str(), &end);
  return end == str.c_str() + str.size();
}

Token ForthLexer::nextToken() {
  skipWhitespace();

  if (curPtr >= endPtr) {
    return Token(Token::Kind::EndOfFile, "",
                 llvm::SMLoc::getFromPointer(curPtr));
  }

  const char *tokenStart = curPtr;
  llvm::SMLoc loc = llvm::SMLoc::getFromPointer(tokenStart);

  if (*curPtr == ':') {
    ++curPtr;
    return Token(Token::Kind::Colon, ":", loc);
  }
  if (*curPtr == ';') {
    ++curPtr;
    return Token(Token::Kind::Semicolon, ";", loc);
  }

  while (curPtr < endPtr && !isWhitespace(*curPtr)) {
    ++curPtr;
  }

  std::string text(tokenStart, curPtr - tokenStart);
  Token::Kind kind;
  if (isFloat(text)) {
    kind = Token::Kind::Float;
    // Don't uppercase float tokens (preserve original text for strtod)
  } else if (isNumber(text)) {
    kind = Token::Kind::Number;
  } else {
    kind = Token::Kind::Word;
    text = toUpperCase(text);
  }

  return Token(kind, text, loc);
}

//===----------------------------------------------------------------------===//
// ForthParser implementation
//===----------------------------------------------------------------------===//

ForthParser::ForthParser(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context), builder(context),
      lexer(sourceMgr, sourceMgr.getMainFileID()),
      currentToken(Token::Kind::EndOfFile, "", llvm::SMLoc()) {
  consume(); // Load first token
}

void ForthParser::consume() { currentToken = lexer.nextToken(); }

Location ForthParser::getLoc() {
  auto lineAndCol = sourceMgr.getLineAndColumn(currentToken.location);
  auto bufferID = sourceMgr.getMainFileID();
  auto bufferName = sourceMgr.getMemoryBuffer(bufferID)->getBufferIdentifier();
  return FileLineColLoc::get(builder.getStringAttr(bufferName),
                             lineAndCol.first, lineAndCol.second);
}

LogicalResult ForthParser::emitError(const llvm::Twine &message) {
  sourceMgr.PrintMessage(currentToken.location, llvm::SourceMgr::DK_Error,
                         message);
  return failure();
}

LogicalResult ForthParser::emitErrorAt(llvm::SMLoc loc,
                                       const llvm::Twine &message) {
  sourceMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, message);
  return failure();
}

LogicalResult ForthParser::parseHeader() {
  auto buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  const char *bufStart = buffer->getBufferStart();
  const char *bufEnd = buffer->getBufferEnd();

  auto trim = [](llvm::StringRef s) { return s.ltrim(" \t").rtrim(" \t\r"); };

  auto splitWS = [](llvm::StringRef s) {
    SmallVector<llvm::StringRef> parts;
    s.split(parts, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    // Re-split on tabs: filter empty parts produced by tab-only tokens.
    SmallVector<llvm::StringRef> result;
    for (auto part : parts) {
      SmallVector<llvm::StringRef> sub;
      part.split(sub, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      result.append(sub.begin(), sub.end());
    }
    return result;
  };

  bool headerEnded = false;
  headerEndPtr = bufStart;

  const char *lineStart = bufStart;
  while (lineStart <= bufEnd) {
    const char *lineEnd = lineStart;
    while (lineEnd < bufEnd && *lineEnd != '\n' && *lineEnd != '\r')
      ++lineEnd;
    llvm::StringRef line(lineStart, lineEnd - lineStart);
    llvm::StringRef trimmed = trim(line);

    auto lineLoc = llvm::SMLoc::getFromPointer(lineStart);

    if (trimmed.empty()) {
      if (!headerEnded)
        headerEndPtr = lineEnd < bufEnd ? lineEnd + 1 : lineEnd;
    } else if (trimmed.starts_with("\\!")) {
      if (headerEnded) {
        return emitErrorAt(lineLoc,
                           "header directive must appear before any code");
      }

      llvm::StringRef directiveLine = trim(trimmed.drop_front(2));
      size_t annotPos = directiveLine.find("--");
      if (annotPos != llvm::StringRef::npos) {
        directiveLine = trim(directiveLine.substr(0, annotPos));
      }

      if (directiveLine.empty()) {
        return emitErrorAt(lineLoc, "empty header directive");
      }

      auto tokens = splitWS(directiveLine);
      if (tokens.empty()) {
        return emitErrorAt(lineLoc, "empty header directive");
      }

      std::string directive = toUpperCase(tokens[0]);
      if (kernelName.empty() && directive != "KERNEL") {
        return emitErrorAt(lineLoc, "\\! kernel must appear first");
      }

      if (directive == "KERNEL") {
        if (!kernelName.empty()) {
          return emitErrorAt(lineLoc, "duplicate \\! kernel directive");
        }
        if (tokens.size() != 2) {
          return emitErrorAt(lineLoc,
                             "kernel directive expects: \\! kernel <name>");
        }
        kernelName = tokens[1].str();
      } else if (directive == "PARAM" || directive == "SHARED") {
        if (tokens.size() != 3) {
          return emitErrorAt(lineLoc,
                             "param/shared directive expects: \\! param "
                             "<name> <type>");
        }

        std::string nameUpper = toUpperCase(tokens[1]);
        for (const auto &param : paramDecls) {
          if (param.name == nameUpper) {
            return emitErrorAt(lineLoc, "duplicate name: " + nameUpper +
                                            " (already declared as param)");
          }
        }
        for (const auto &shared : sharedDecls) {
          if (shared.name == nameUpper) {
            return emitErrorAt(lineLoc, "duplicate name: " + nameUpper +
                                            " (already declared as shared)");
          }
        }

        llvm::StringRef typeToken = tokens[2];
        bool isArray = false;
        int64_t size = 0;
        BaseType baseType = BaseType::I64;
        size_t lbracket = typeToken.find('[');
        if (lbracket != llvm::StringRef::npos) {
          size_t rbracket = typeToken.find(']');
          if (rbracket == llvm::StringRef::npos ||
              rbracket != typeToken.size() - 1) {
            return emitErrorAt(lineLoc,
                               "array type must use suffix [N], e.g. i64[4]");
          }
          llvm::StringRef base = typeToken.substr(0, lbracket);
          llvm::StringRef sizeStr =
              typeToken.substr(lbracket + 1, rbracket - lbracket - 1);
          std::string baseUpper = toUpperCase(base);
          if (baseUpper == "I64") {
            baseType = BaseType::I64;
          } else if (baseUpper == "F64") {
            baseType = BaseType::F64;
          } else {
            return emitErrorAt(lineLoc, "unsupported base type: " + base.str());
          }
          if (sizeStr.empty())
            return emitErrorAt(lineLoc, "array type requires a size");
          if (sizeStr.getAsInteger(10, size) || size <= 0)
            return emitErrorAt(lineLoc,
                               "array size must be a positive integer");
          isArray = true;
        } else {
          std::string typeUpper = toUpperCase(typeToken);
          if (typeUpper == "I64") {
            baseType = BaseType::I64;
          } else if (typeUpper == "F64") {
            baseType = BaseType::F64;
          } else {
            return emitErrorAt(lineLoc,
                               "unsupported scalar type: " + typeToken.str());
          }
        }

        if (directive == "PARAM") {
          ParamDecl decl;
          decl.name = nameUpper;
          decl.isArray = isArray;
          decl.size = size;
          decl.baseType = baseType;
          paramDecls.push_back(decl);
        } else {
          SharedDecl decl;
          decl.name = nameUpper;
          decl.isArray = isArray;
          decl.size = size;
          decl.baseType = baseType;
          sharedDecls.push_back(decl);
        }
      } else {
        return emitErrorAt(lineLoc, "unknown header directive: " + directive);
      }

      headerEndPtr = lineEnd < bufEnd ? lineEnd + 1 : lineEnd;
    } else if (trimmed.starts_with("\\")) {
      if (trimmed.size() == 1 || trimmed[1] == ' ' || trimmed[1] == '\t') {
        if (!headerEnded)
          headerEndPtr = lineEnd < bufEnd ? lineEnd + 1 : lineEnd;
      } else {
        headerEnded = true;
      }
    } else {
      headerEnded = true;
    }

    if (lineEnd >= bufEnd)
      break;
    if (*lineEnd == '\r' && lineEnd + 1 < bufEnd && lineEnd[1] == '\n')
      lineStart = lineEnd + 2;
    else
      lineStart = lineEnd + 1;
  }

  if (kernelName.empty()) {
    auto loc = llvm::SMLoc::getFromPointer(bufStart);
    return emitErrorAt(loc, "\\! kernel <name> is required");
  }

  return success();
}

Value ForthParser::emitOperation(StringRef word, Value inputStack,
                                 Location loc) {
  Type stackType = forth::StackType::get(context);

  // Check if word is a local variable (only valid inside word definitions)
  if (inWordDefinition) {
    auto it = localVars.find(word);
    if (it != localVars.end()) {
      return builder
          .create<forth::PushValueOp>(loc, stackType, inputStack, it->second)
          .getOutputStack();
    }
  }

  // Check if word is a param name (only valid outside word definitions)
  if (!inWordDefinition) {
    for (const auto &param : paramDecls) {
      if (word == param.name) {
        return builder
            .create<forth::ParamRefOp>(loc, stackType, inputStack,
                                       builder.getStringAttr(param.name))
            .getResult();
      }
    }
  } else {
    for (const auto &param : paramDecls) {
      if (word == param.name) {
        (void)emitError("parameter '" + param.name +
                        "' cannot be referenced inside a word definition; "
                        "pass the address from the caller instead");
        return nullptr;
      }
    }
  }

  // Check if word is a shared memory name (only valid outside word definitions)
  if (!inWordDefinition) {
    auto it = sharedAllocs.find(word);
    if (it != sharedAllocs.end()) {
      Value alloca = it->second;
      Value ptrIndex =
          builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, alloca);
      Value ptrI64 = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), ptrIndex);
      return builder
          .create<forth::PushValueOp>(loc, stackType, inputStack, ptrI64)
          .getOutputStack();
    }
  } else {
    for (const auto &shared : sharedDecls) {
      if (word == shared.name) {
        (void)emitError("shared memory '" + shared.name +
                        "' cannot be referenced inside a word definition; "
                        "pass the address from the caller instead");
        return nullptr;
      }
    }
  }

  // Check user-defined words first
  std::string mangledWord = mangleForthName(word);
  if (wordDefs.count(mangledWord)) {
    Type stackType = forth::StackType::get(context);
    auto symbolRef = mlir::FlatSymbolRefAttr::get(context, mangledWord);
    return builder.create<func::CallOp>(loc, stackType, symbolRef, inputStack)
        .getResult(0);
  }

  // CELLS: multiply by 8 (sizeof i64 = sizeof f64) for byte addressing
  if (word == "CELLS") {
    Value lit8 = builder
                     .create<forth::ConstantOp>(loc, stackType, inputStack,
                                                builder.getI64IntegerAttr(8))
                     .getResult();
    return builder.create<forth::MulIOp>(loc, stackType, lit8).getResult();
  }

  // Built-in operations
  if (word == "DUP") {
    return builder.create<forth::DupOp>(loc, stackType, inputStack).getResult();
  } else if (word == "DROP") {
    return builder.create<forth::DropOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "SWAP") {
    return builder.create<forth::SwapOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "OVER") {
    return builder.create<forth::OverOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "ROT") {
    return builder.create<forth::RotOp>(loc, stackType, inputStack).getResult();
  } else if (word == "NIP") {
    return builder.create<forth::NipOp>(loc, stackType, inputStack).getResult();
  } else if (word == "TUCK") {
    return builder.create<forth::TuckOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "PICK") {
    return builder.create<forth::PickOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "ROLL") {
    return builder.create<forth::RollOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "+" || word == "ADD") {
    return builder.create<forth::AddIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "-" || word == "SUB") {
    return builder.create<forth::SubIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "*" || word == "MUL") {
    return builder.create<forth::MulIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "/" || word == "DIV") {
    return builder.create<forth::DivIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F+") {
    return builder.create<forth::AddFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F-") {
    return builder.create<forth::SubFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F*") {
    return builder.create<forth::MulFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F/") {
    return builder.create<forth::DivFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "MOD") {
    return builder.create<forth::ModOp>(loc, stackType, inputStack).getResult();
  } else if (word == "AND") {
    return builder.create<forth::AndOp>(loc, stackType, inputStack).getResult();
  } else if (word == "OR") {
    return builder.create<forth::OrOp>(loc, stackType, inputStack).getResult();
  } else if (word == "XOR") {
    return builder.create<forth::XorOp>(loc, stackType, inputStack).getResult();
  } else if (word == "NOT") {
    return builder.create<forth::NotOp>(loc, stackType, inputStack).getResult();
  } else if (word == "LSHIFT") {
    return builder.create<forth::LshiftOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "RSHIFT") {
    return builder.create<forth::RshiftOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "@") {
    return builder.create<forth::LoadIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "!") {
    return builder.create<forth::StoreIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F@") {
    return builder.create<forth::LoadFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F!") {
    return builder.create<forth::StoreFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "S@") {
    return builder.create<forth::SharedLoadIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "S!") {
    return builder.create<forth::SharedStoreIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "SF@") {
    return builder.create<forth::SharedLoadFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "SF!") {
    return builder.create<forth::SharedStoreFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "TID-X") {
    return builder.create<forth::ThreadIdXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "TID-Y") {
    return builder.create<forth::ThreadIdYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "TID-Z") {
    return builder.create<forth::ThreadIdZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BID-X") {
    return builder.create<forth::BlockIdXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BID-Y") {
    return builder.create<forth::BlockIdYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BID-Z") {
    return builder.create<forth::BlockIdZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BDIM-X") {
    return builder.create<forth::BlockDimXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BDIM-Y") {
    return builder.create<forth::BlockDimYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BDIM-Z") {
    return builder.create<forth::BlockDimZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "GDIM-X") {
    return builder.create<forth::GridDimXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "GDIM-Y") {
    return builder.create<forth::GridDimYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "GDIM-Z") {
    return builder.create<forth::GridDimZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "GLOBAL-ID") {
    return builder.create<forth::GlobalIdOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "BARRIER") {
    builder.create<forth::BarrierOp>(loc);
    return inputStack;
  } else if (word == "=") {
    return builder.create<forth::EqIOp>(loc, stackType, inputStack).getResult();
  } else if (word == "<") {
    return builder.create<forth::LtIOp>(loc, stackType, inputStack).getResult();
  } else if (word == ">") {
    return builder.create<forth::GtIOp>(loc, stackType, inputStack).getResult();
  } else if (word == "<>") {
    return builder.create<forth::NeIOp>(loc, stackType, inputStack).getResult();
  } else if (word == "<=") {
    return builder.create<forth::LeIOp>(loc, stackType, inputStack).getResult();
  } else if (word == ">=") {
    return builder.create<forth::GeIOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F=") {
    return builder.create<forth::EqFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F<") {
    return builder.create<forth::LtFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F>") {
    return builder.create<forth::GtFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F<>") {
    return builder.create<forth::NeFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F<=") {
    return builder.create<forth::LeFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "F>=") {
    return builder.create<forth::GeFOp>(loc, stackType, inputStack).getResult();
  } else if (word == "S>F") {
    return builder.create<forth::IToFOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "F>S") {
    return builder.create<forth::FToIOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "0=") {
    return builder.create<forth::ZeroEqOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "I" || word == "J" || word == "K") {
    int64_t depth = (word == "I") ? 0 : (word == "J") ? 1 : 2;
    if (static_cast<int64_t>(loopStack.size()) < depth + 1) {
      (void)emitError("'" + word.str() + "' requires " +
                      std::to_string(depth + 1) + " nested DO/LOOP(s)");
      return nullptr;
    }
    // Load counter from the appropriate loop context
    auto &ctx = loopStack[loopStack.size() - 1 - depth];
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value idx =
        builder.create<memref::LoadOp>(loc, ctx.counter, ValueRange{c0});
    return builder.create<forth::PushValueOp>(loc, stackType, inputStack, idx)
        .getOutputStack();
  }

  // Unknown word
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Body parsing - shared by word definitions and main.
// Control flow words are handled inline using cf.br/cf.cond_br.
//===----------------------------------------------------------------------===//

Block *ForthParser::createStackBlock(Region *region, Location loc) {
  auto *block = new Block();
  region->push_back(block);
  block->addArgument(forth::StackType::get(context), loc);
  return block;
}

std::pair<Value, Value> ForthParser::emitPopFlag(Location loc, Value stack) {
  auto popFlag = builder.create<forth::PopFlagOp>(
      loc, forth::StackType::get(context), builder.getI1Type(), stack);
  return {popFlag.getOutputStack(), popFlag.getFlag()};
}

void ForthParser::emitLoopEnd(Location loc, const LoopContext &ctx, Value step,
                              Value &stack) {
  auto i64Type = builder.getI64Type();

  // Load old counter, compute new = old + step, store.
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value oldIdx =
      builder.create<memref::LoadOp>(loc, ctx.counter, ValueRange{c0});
  Value newIdx = builder.create<arith::AddIOp>(loc, oldIdx, step);
  builder.create<memref::StoreOp>(loc, newIdx, ctx.counter, ValueRange{c0});

  // Crossing test: ((oldIdx - limit) XOR (newIdx - limit)) < 0
  // This correctly handles both positive and negative step values.
  Value oldDiff = builder.create<arith::SubIOp>(loc, oldIdx, ctx.limit);
  Value newDiff = builder.create<arith::SubIOp>(loc, newIdx, ctx.limit);
  Value xorVal = builder.create<arith::XOrIOp>(loc, oldDiff, newDiff);
  Value zero = builder.create<arith::ConstantOp>(loc, i64Type,
                                                 builder.getI64IntegerAttr(0));
  Value crossed = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                xorVal, zero);

  // If crossed → exit, otherwise → loop back to body.
  builder.create<cf::CondBranchOp>(loc, crossed, ctx.exit, ValueRange{stack},
                                   ctx.body, ValueRange{stack});

  // Continue after exit.
  builder.setInsertionPointToStart(ctx.exit);
  stack = ctx.exit->getArgument(0);
}

LogicalResult ForthParser::parseBody(Value &stack) {
  Type stackType = forth::StackType::get(context);

  while (currentToken.kind != Token::Kind::EndOfFile &&
         currentToken.kind != Token::Kind::Semicolon &&
         currentToken.kind != Token::Kind::Colon) {

    if (currentToken.kind == Token::Kind::Number) {
      Location tokenLoc = getLoc();
      int64_t value = std::stoll(currentToken.text);
      stack = builder
                  .create<forth::ConstantOp>(tokenLoc, stackType, stack,
                                             builder.getI64IntegerAttr(value))
                  .getResult();
      consume();
    } else if (currentToken.kind == Token::Kind::Float) {
      Location tokenLoc = getLoc();
      double value = std::stod(currentToken.text);
      stack = builder
                  .create<forth::ConstantOp>(tokenLoc, stackType, stack,
                                             builder.getF64FloatAttr(value))
                  .getResult();
      consume();
    } else if (currentToken.kind == Token::Kind::Word) {
      Location loc = getLoc();
      StringRef word = currentToken.text;

      //=== IF ===
      if (word == "IF") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();

        auto [s1, flag] = emitPopFlag(loc, stack);

        auto *thenBlock = createStackBlock(parentRegion, loc);
        auto *joinBlock = createStackBlock(parentRegion, loc);

        // Branch: true -> then, false -> join.
        builder.create<cf::CondBranchOp>(loc, flag, thenBlock, ValueRange{s1},
                                         joinBlock, ValueRange{s1});

        // Push join block for THEN/ELSE to pick up.
        cfStack.push_back({CFTag::Orig, joinBlock});

        // Continue parsing in then block.
        builder.setInsertionPointToStart(thenBlock);
        stack = thenBlock->getArgument(0);

        //=== ELSE ===
      } else if (word == "ELSE") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();

        auto *mergeBlock = createStackBlock(parentRegion, loc);

        // End of then-body: branch to merge.
        builder.create<cf::BranchOp>(loc, mergeBlock, ValueRange{stack});

        // Pop the false-path block (from IF) - this becomes else-body start.
        if (cfStack.empty())
          return emitError("ELSE without matching IF");
        auto [tag, joinBlock] = cfStack.pop_back_val();
        if (tag != CFTag::Orig)
          return emitError("ELSE without matching IF");

        // Push merge block for THEN to pick up.
        cfStack.push_back({CFTag::Orig, mergeBlock});

        // Continue parsing in the else (false-path) block.
        builder.setInsertionPointToStart(joinBlock);
        stack = joinBlock->getArgument(0);

        //=== THEN ===
      } else if (word == "THEN") {
        consume();

        // Pop the join/merge block.
        if (cfStack.empty())
          return emitError("THEN without matching IF");
        auto [tag, joinBlock] = cfStack.pop_back_val();
        if (tag != CFTag::Orig)
          return emitError("THEN without matching IF");

        // Branch from current block to join.
        builder.create<cf::BranchOp>(loc, joinBlock, ValueRange{stack});

        // Continue parsing after the join.
        builder.setInsertionPointToStart(joinBlock);
        stack = joinBlock->getArgument(0);

        //=== BEGIN ===
      } else if (word == "BEGIN") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();

        auto *loopBlock = createStackBlock(parentRegion, loc);

        // Branch to loop header.
        builder.create<cf::BranchOp>(loc, loopBlock, ValueRange{stack});

        // Push loop header as backward reference.
        cfStack.push_back({CFTag::Dest, loopBlock});

        // Continue parsing in loop body.
        builder.setInsertionPointToStart(loopBlock);
        stack = loopBlock->getArgument(0);

        //=== UNTIL ===
      } else if (word == "UNTIL") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();

        auto [s1, flag] = emitPopFlag(loc, stack);

        if (cfStack.empty())
          return emitError("UNTIL without matching BEGIN");
        auto [tag, loopBlock] = cfStack.pop_back_val();
        if (tag != CFTag::Dest)
          return emitError("UNTIL without matching BEGIN");

        auto *exitBlock = createStackBlock(parentRegion, loc);

        // true -> exit, false -> loop back.
        builder.create<cf::CondBranchOp>(loc, flag, exitBlock, ValueRange{s1},
                                         loopBlock, ValueRange{s1});

        // Continue after exit.
        builder.setInsertionPointToStart(exitBlock);
        stack = exitBlock->getArgument(0);

        //=== WHILE ===
      } else if (word == "WHILE") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();

        auto [s1, flag] = emitPopFlag(loc, stack);

        if (cfStack.empty())
          return emitError("WHILE without matching BEGIN");
        auto [tag, loopBlock] = cfStack.pop_back_val();
        if (tag != CFTag::Dest)
          return emitError("WHILE without matching BEGIN");

        auto *bodyBlock = createStackBlock(parentRegion, loc);
        auto *exitBlock = createStackBlock(parentRegion, loc);

        // true -> body, false -> exit.
        builder.create<cf::CondBranchOp>(loc, flag, bodyBlock, ValueRange{s1},
                                         exitBlock, ValueRange{s1});

        // Push exit (forward ref) then loop header (backward ref).
        cfStack.push_back({CFTag::Orig, exitBlock});
        cfStack.push_back({CFTag::Dest, loopBlock});

        // Continue parsing in body.
        builder.setInsertionPointToStart(bodyBlock);
        stack = bodyBlock->getArgument(0);

        //=== REPEAT ===
      } else if (word == "REPEAT") {
        consume();

        // Pop loop header (from WHILE's re-push).
        if (cfStack.empty())
          return emitError("REPEAT without matching WHILE");
        auto [destTag, loopBlock] = cfStack.pop_back_val();
        if (destTag != CFTag::Dest)
          return emitError("REPEAT without matching WHILE");

        // Branch back to loop header.
        builder.create<cf::BranchOp>(loc, loopBlock, ValueRange{stack});

        // Pop exit block (from WHILE).
        if (cfStack.empty())
          return emitError("REPEAT without matching WHILE");
        auto [origTag, exitBlock] = cfStack.pop_back_val();
        if (origTag != CFTag::Orig)
          return emitError("REPEAT without matching WHILE");

        // Continue after exit.
        builder.setInsertionPointToStart(exitBlock);
        stack = exitBlock->getArgument(0);

        //=== LEAVE ===
      } else if (word == "LEAVE") {
        consume();

        if (loopStack.empty()) {
          return emitError("LEAVE without matching DO");
        }

        Region *parentRegion = builder.getInsertionBlock()->getParent();
        auto &ctx = loopStack.back();

        // Continue parsing in a dead block to avoid inserting after a
        // terminator. Use a dummy cond_br to create a reachable dead block that
        // carries a stack argument, keeping cf->memref type conversion
        // consistent.
        auto *deadBlock = createStackBlock(parentRegion, loc);
        Value cond = builder.create<arith::ConstantOp>(
            loc, builder.getI1Type(), builder.getBoolAttr(true));
        builder.create<cf::CondBranchOp>(loc, cond, ctx.exit, ValueRange{stack},
                                         deadBlock, ValueRange{stack});
        builder.setInsertionPointToStart(deadBlock);
        stack = deadBlock->getArgument(0);

        //=== UNLOOP ===
      } else if (word == "UNLOOP") {
        consume();

        if (loopStack.empty()) {
          return emitError("UNLOOP without matching DO");
        }

        // No-op: loop control uses CFG blocks and a memref counter, not the
        // Forth stack, so there is nothing to discard. We keep the loopStack
        // intact so that LOOP can still find its matching DO.

        //=== EXIT ===
      } else if (word == "EXIT") {
        consume();

        if (!inWordDefinition) {
          return emitError("EXIT outside word definition");
        }

        Region *parentRegion = builder.getInsertionBlock()->getParent();

        // Create a block that performs the return.
        auto *returnBlock = createStackBlock(parentRegion, loc);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(returnBlock);
          builder.create<func::ReturnOp>(loc, returnBlock->getArgument(0));
        }

        // Use a dummy cond_br to keep the dead block structurally reachable,
        // matching the pattern used by LEAVE.
        auto *deadBlock = createStackBlock(parentRegion, loc);
        Value cond = builder.create<arith::ConstantOp>(
            loc, builder.getI1Type(), builder.getBoolAttr(true));
        builder.create<cf::CondBranchOp>(loc, cond, returnBlock,
                                         ValueRange{stack}, deadBlock,
                                         ValueRange{stack});
        builder.setInsertionPointToStart(deadBlock);
        stack = deadBlock->getArgument(0);

        //=== DO ===
      } else if (word == "DO") {
        consume();
        Region *parentRegion = builder.getInsertionBlock()->getParent();
        auto i64Type = builder.getI64Type();

        // Pop start and limit from the Forth stack.
        auto popStart =
            builder.create<forth::PopOp>(loc, stackType, i64Type, stack);
        Value s1 = popStart.getOutputStack();
        Value start = popStart.getValue();

        auto popLimit =
            builder.create<forth::PopOp>(loc, stackType, i64Type, s1);
        Value s2 = popLimit.getOutputStack();
        Value limit = popLimit.getValue();

        // Allocate counter storage.
        auto counterType = MemRefType::get({1}, i64Type);
        Value counter = builder.create<memref::AllocaOp>(loc, counterType);
        Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
        builder.create<memref::StoreOp>(loc, start, counter, ValueRange{c0});

        // Create body and exit blocks (post-test loop: always enters once).
        auto *bodyBlock = createStackBlock(parentRegion, loc);
        auto *exitBlock = createStackBlock(parentRegion, loc);

        // Branch directly to body.
        builder.create<cf::BranchOp>(loc, bodyBlock, ValueRange{s2});

        // Push loop context for I/J/K.
        loopStack.push_back({counter, limit, bodyBlock, exitBlock});

        // Continue parsing in body.
        builder.setInsertionPointToStart(bodyBlock);
        stack = bodyBlock->getArgument(0);

        //=== LOOP ===
      } else if (word == "LOOP") {
        consume();

        if (loopStack.empty()) {
          return emitError("LOOP without matching DO");
        }

        auto ctx = loopStack.pop_back_val();
        Value one = builder.create<arith::ConstantOp>(
            loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
        emitLoopEnd(loc, ctx, one, stack);

        //=== +LOOP ===
      } else if (word == "+LOOP") {
        consume();

        if (loopStack.empty()) {
          return emitError("+LOOP without matching DO");
        }

        auto ctx = loopStack.pop_back_val();

        // Pop step from data stack.
        auto popOp = builder.create<forth::PopOp>(
            loc, forth::StackType::get(context), builder.getI64Type(), stack);
        stack = popOp.getOutputStack();
        Value step = popOp.getValue();
        emitLoopEnd(loc, ctx, step, stack);

        //=== Normal word ===
      } else {
        Value newStack = emitOperation(currentToken.text, stack, loc);
        if (!newStack)
          return emitError("unknown word: " + currentToken.text);
        stack = newStack;
        consume();
      }
    } else {
      return emitError("unexpected token");
    }
  }

  if (!cfStack.empty()) {
    cfStack.clear();
    return emitError("unclosed control flow (missing THEN, REPEAT, or UNTIL?)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Word definition and top-level parsing.
//===----------------------------------------------------------------------===//

LogicalResult ForthParser::parseLocals(Value &stack) {
  // If current token is not '{', no locals to parse
  if (currentToken.kind != Token::Kind::Word || currentToken.text != "{")
    return success();

  Location loc = getLoc();
  consume(); // consume '{'

  // Collect local names until '--' or '}'
  SmallVector<std::string> names;
  while (currentToken.kind != Token::Kind::EndOfFile) {
    if (currentToken.kind == Token::Kind::Word && currentToken.text == "--")
      break;
    if (currentToken.kind == Token::Kind::Word && currentToken.text == "}")
      break;

    if (currentToken.kind != Token::Kind::Word)
      return emitError("expected local variable name in { ... }");

    std::string name = currentToken.text; // already uppercased by lexer

    // Check for duplicate local names
    for (const auto &existing : names) {
      if (existing == name)
        return emitError("duplicate local variable name: " + name);
    }

    // Check for conflicts with param names
    for (const auto &param : paramDecls) {
      if (param.name == name)
        return emitError("local variable name '" + name +
                         "' conflicts with parameter name");
    }

    // Check for conflicts with shared names
    for (const auto &shared : sharedDecls) {
      if (shared.name == name)
        return emitError("local variable name '" + name +
                         "' conflicts with shared memory name");
    }

    names.push_back(name);
    consume();
  }

  // Skip '--' and output names until '}'
  if (currentToken.kind == Token::Kind::Word && currentToken.text == "--") {
    consume(); // consume '--'
    while (currentToken.kind != Token::Kind::EndOfFile) {
      if (currentToken.kind == Token::Kind::Word && currentToken.text == "}")
        break;
      consume(); // skip output names (ignored)
    }
  }

  if (currentToken.kind != Token::Kind::Word || currentToken.text != "}")
    return emitError("expected '}' to close local variable declaration");

  consume(); // consume '}'

  if (names.empty())
    return success();

  // Pop values in reverse order: { a b c -- } with stack ( 1 2 3 )
  // pops 3->c, 2->b, 1->a
  Type i64Type = builder.getI64Type();
  Type stackType = forth::StackType::get(context);
  for (int i = names.size() - 1; i >= 0; --i) {
    auto popOp = builder.create<forth::PopOp>(loc, stackType, i64Type, stack);
    stack = popOp.getOutputStack();
    localVars[names[i]] = popOp.getValue();
  }

  return success();
}

LogicalResult ForthParser::parseWordDefinition() {
  Location loc = getLoc();
  auto savedInsertionPoint = builder.saveInsertionPoint();
  inWordDefinition = true;

  consume(); // consume ':'

  if (currentToken.kind != Token::Kind::Word) {
    return emitError("expected word name after ':'");
  }

  std::string wordName = mangleForthName(currentToken.text);
  consume(); // consume word name

  // Create function: !forth.stack -> !forth.stack
  Type stackType = forth::StackType::get(context);
  auto funcType = builder.getFunctionType({stackType}, {stackType});
  auto funcOp = builder.create<func::FuncOp>(loc, wordName, funcType);
  funcOp.setPrivate();

  // Create entry block
  Block *entryBlock = funcOp.addEntryBlock();
  Value resultStack = entryBlock->getArgument(0);
  builder.setInsertionPointToStart(entryBlock);

  // Parse local variable declarations (if any)
  if (failed(parseLocals(resultStack)))
    return failure();

  // Parse word body until ';'
  if (failed(parseBody(resultStack)))
    return failure();

  if (currentToken.kind != Token::Kind::Semicolon) {
    return emitError("unterminated word definition: missing ';'");
  }

  // Add return at current insertion point (may differ from entry block
  // if the word body contains control flow).
  builder.create<func::ReturnOp>(loc, resultStack);

  // Register the word
  wordDefs.insert(wordName);

  consume(); // consume ';'

  inWordDefinition = false;
  localVars.clear();

  // Restore insertion point
  builder.restoreInsertionPoint(savedInsertionPoint);
  return success();
}

LogicalResult ForthParser::parseOperations(Value &stack) {
  Type stackType = forth::StackType::get(context);
  Location loc = getLoc();

  // Initialize the stack.
  stack = builder.create<forth::StackOp>(loc, stackType);

  while (currentToken.kind != Token::Kind::EndOfFile) {
    if (currentToken.kind == Token::Kind::Colon) {
      // Skip entire word definition
      consume(); // consume ':'
      while (currentToken.kind != Token::Kind::Semicolon) {
        if (currentToken.kind == Token::Kind::EndOfFile) {
          return emitError("unterminated word definition: missing ';'");
        }
        consume();
      }
      consume(); // consume ';'
      continue;
    }

    // parseBody handles numbers, words, and IF/ELSE/THEN.
    if (failed(parseBody(stack)))
      return failure();
  }

  return success();
}

OwningOpRef<ModuleOp> ForthParser::parseModule() {
  // Create a module to hold the parsed operations
  Location loc = getLoc();
  OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToEnd(module->getBody());

  if (failed(parseHeader()))
    return nullptr;

  // First pass: parse all word definitions
  lexer.resetTo(headerEndPtr);
  consume();
  while (currentToken.kind != Token::Kind::EndOfFile) {
    if (currentToken.kind == Token::Kind::Colon) {
      if (failed(parseWordDefinition())) {
        return nullptr;
      }
    } else {
      consume();
    }
  }

  // Reset lexer for second pass
  lexer.resetTo(headerEndPtr);
  consume();

  // Reset insertion point to end of module for main function
  builder.setInsertionPointToEnd(module->getBody());

  // Build function argument types from param declarations
  SmallVector<Type> argTypes;
  for (const auto &param : paramDecls) {
    Type elemType = param.baseType == BaseType::F64
                        ? Type(builder.getF64Type())
                        : Type(builder.getI64Type());
    if (param.isArray) {
      argTypes.push_back(MemRefType::get({param.size}, elemType));
    } else {
      argTypes.push_back(elemType);
    }
  }

  auto funcType = builder.getFunctionType(argTypes, {});
  auto funcOp = builder.create<func::FuncOp>(loc, kernelName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("forth.kernel", builder.getUnitAttr());

  // Annotate arguments with param names
  for (size_t i = 0; i < paramDecls.size(); ++i) {
    funcOp.setArgAttr(i, "forth.param_name",
                      builder.getStringAttr(paramDecls[i].name));
  }

  // Create the entry block with arguments
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Emit shared memory allocations at kernel entry
  for (const auto &shared : sharedDecls) {
    int64_t size = shared.isArray ? shared.size : 1;
    Type elemType = shared.baseType == BaseType::F64
                        ? Type(builder.getF64Type())
                        : Type(builder.getI64Type());
    auto memrefType = MemRefType::get({size}, elemType);
    Value alloca = builder.create<memref::AllocaOp>(loc, memrefType);
    alloca.getDefiningOp()->setAttr("forth.shared_name",
                                    builder.getStringAttr(shared.name));
    sharedAllocs[shared.name] = alloca;
  }

  // Parse Forth operations
  Value finalStack;
  if (failed(parseOperations(finalStack))) {
    return nullptr;
  }

  // Add a return operation
  builder.create<func::ReturnOp>(loc);

  return module;
}

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> forth::parseForthSource(llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  // Ensure required dialects are loaded
  context->loadDialect<forth::ForthDialect>();
  context->loadDialect<func::FuncDialect>();
  context->loadDialect<cf::ControlFlowDialect>();
  context->loadDialect<arith::ArithDialect>();
  context->loadDialect<memref::MemRefDialect>();

  // Create parser and parse the module
  ForthParser parser(sourceMgr, context);
  return parser.parseModule();
}

void mlir::forth::registerForthToMLIRTranslation() {
  TranslateToMLIRRegistration registration(
      "forth-to-mlir", "Translate Forth source to MLIR",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        return forth::parseForthSource(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        registry.insert<forth::ForthDialect, func::FuncDialect,
                        cf::ControlFlowDialect, arith::ArithDialect,
                        memref::MemRefDialect>();
      });
}
