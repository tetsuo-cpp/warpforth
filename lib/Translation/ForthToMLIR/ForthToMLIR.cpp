//===- ForthToMLIR.cpp - Forth to MLIR translation -----------------------===//
//
// This file implements the Forth-to-MLIR translation.
//
//===----------------------------------------------------------------------===//

#include "ForthToMLIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"
#include "llvm/ADT/StringExtras.h"
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
  Token::Kind kind = isNumber(text) ? Token::Kind::Number : Token::Kind::Word;

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

void ForthParser::scanParamDeclarations() {
  lexer.reset();
  consume();
  while (currentToken.kind != Token::Kind::EndOfFile) {
    if (currentToken.kind == Token::Kind::Word &&
        currentToken.text == "param") {
      consume(); // consume "param"
      if (currentToken.kind != Token::Kind::Word)
        continue;
      std::string name = currentToken.text;
      consume(); // consume name
      if (currentToken.kind != Token::Kind::Number)
        continue;
      int64_t size = std::stoll(currentToken.text);
      consume(); // consume size
      paramDecls.push_back({name, size});
    } else {
      consume();
    }
  }
}

Value ForthParser::emitOperation(StringRef word, Value inputStack,
                                 Location loc) {
  Type stackType = forth::StackType::get(context);

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

  // Check user-defined words first
  std::string mangledWord = mangleForthName(word);
  if (wordDefs.count(mangledWord)) {
    Type stackType = forth::StackType::get(context);
    auto symbolRef = mlir::FlatSymbolRefAttr::get(context, mangledWord);
    return builder.create<func::CallOp>(loc, stackType, symbolRef, inputStack)
        .getResult(0);
  }

  // cells: multiply by 8 (sizeof i64) for byte addressing
  if (word == "cells") {
    Value lit8 = builder
                     .create<forth::LiteralOp>(loc, stackType, inputStack,
                                               builder.getI64IntegerAttr(8))
                     .getResult();
    return builder.create<forth::MulOp>(loc, stackType, lit8).getResult();
  }

  // Built-in operations
  if (word == "dup") {
    return builder.create<forth::DupOp>(loc, stackType, inputStack).getResult();
  } else if (word == "drop") {
    return builder.create<forth::DropOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "swap") {
    return builder.create<forth::SwapOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "over") {
    return builder.create<forth::OverOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "rot") {
    return builder.create<forth::RotOp>(loc, stackType, inputStack).getResult();
  } else if (word == "+" || word == "add") {
    return builder.create<forth::AddOp>(loc, stackType, inputStack).getResult();
  } else if (word == "-" || word == "sub") {
    return builder.create<forth::SubOp>(loc, stackType, inputStack).getResult();
  } else if (word == "*" || word == "mul") {
    return builder.create<forth::MulOp>(loc, stackType, inputStack).getResult();
  } else if (word == "/" || word == "div") {
    return builder.create<forth::DivOp>(loc, stackType, inputStack).getResult();
  } else if (word == "mod") {
    return builder.create<forth::ModOp>(loc, stackType, inputStack).getResult();
  } else if (word == "@") {
    return builder.create<forth::LoadOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "!") {
    return builder.create<forth::StoreOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "tid-x") {
    return builder.create<forth::ThreadIdXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "tid-y") {
    return builder.create<forth::ThreadIdYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "tid-z") {
    return builder.create<forth::ThreadIdZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bid-x") {
    return builder.create<forth::BlockIdXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bid-y") {
    return builder.create<forth::BlockIdYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bid-z") {
    return builder.create<forth::BlockIdZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bdim-x") {
    return builder.create<forth::BlockDimXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bdim-y") {
    return builder.create<forth::BlockDimYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "bdim-z") {
    return builder.create<forth::BlockDimZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "gdim-x") {
    return builder.create<forth::GridDimXOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "gdim-y") {
    return builder.create<forth::GridDimYOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "gdim-z") {
    return builder.create<forth::GridDimZOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "global-id") {
    return builder.create<forth::GlobalIdOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "=") {
    return builder.create<forth::EqOp>(loc, stackType, inputStack).getResult();
  } else if (word == "<") {
    return builder.create<forth::LtOp>(loc, stackType, inputStack).getResult();
  } else if (word == ">") {
    return builder.create<forth::GtOp>(loc, stackType, inputStack).getResult();
  } else if (word == "0=") {
    return builder.create<forth::ZeroEqOp>(loc, stackType, inputStack)
        .getResult();
  } else if (word == "I") {
    if (doLoopDepth == 0) {
      (void)emitError("'I' used outside of DO/LOOP");
      return nullptr;
    }
    return builder.create<forth::LoopIndexOp>(loc, stackType, inputStack)
        .getResult();
  }

  // Unknown word
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Body parsing — shared by word definitions, main, and control flow regions.
//===----------------------------------------------------------------------===//

LogicalResult
ForthParser::parseBody(Value &stack,
                       llvm::function_ref<bool(StringRef)> isStopWord) {
  Type stackType = forth::StackType::get(context);

  while (currentToken.kind != Token::Kind::EndOfFile &&
         currentToken.kind != Token::Kind::Semicolon &&
         currentToken.kind != Token::Kind::Colon) {

    // Check if current word is a stop word.
    if (currentToken.kind == Token::Kind::Word && isStopWord(currentToken.text))
      break;

    // Skip param declarations at top level.
    if (!inWordDefinition && currentToken.kind == Token::Kind::Word &&
        currentToken.text == "param") {
      consume(); // "param"
      consume(); // name
      consume(); // size
      continue;
    }

    if (currentToken.kind == Token::Kind::Number) {
      Location tokenLoc = getLoc();
      int64_t value = std::stoll(currentToken.text);
      stack = builder
                  .create<forth::LiteralOp>(tokenLoc, stackType, stack,
                                            builder.getI64IntegerAttr(value))
                  .getResult();
      consume();
    } else if (currentToken.kind == Token::Kind::Word) {
      if (currentToken.text == "IF") {
        Location tokenLoc = getLoc();
        consume(); // consume IF
        stack = parseIf(stack, tokenLoc);
        if (!stack)
          return failure();
      } else if (currentToken.text == "BEGIN") {
        Location tokenLoc = getLoc();
        consume(); // consume BEGIN
        stack = parseBeginUntil(stack, tokenLoc);
        if (!stack)
          return failure();
      } else if (currentToken.text == "DO") {
        Location tokenLoc = getLoc();
        consume(); // consume DO
        stack = parseDoLoop(stack, tokenLoc);
        if (!stack)
          return failure();
      } else {
        Location tokenLoc = getLoc();
        Value newStack = emitOperation(currentToken.text, stack, tokenLoc);
        if (!newStack)
          return emitError("unknown word: " + currentToken.text);
        stack = newStack;
        consume();
      }
    } else {
      return emitError("unexpected token");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IF / ELSE / THEN parsing.
//===----------------------------------------------------------------------===//

Value ForthParser::parseIf(Value inputStack, Location loc) {
  Type stackType = forth::StackType::get(context);

  // Create forth.if op. inputStack has the flag on top.
  // Regions capture inputStack from the enclosing scope (no block args).
  // Each region starts with forth.drop to pop the flag.
  auto ifOp = builder.create<forth::IfOp>(loc, stackType, inputStack);

  auto isElseOrThen = [](StringRef word) {
    return word == "ELSE" || word == "THEN";
  };
  auto isThen = [](StringRef word) { return word == "THEN"; };

  // --- Then region ---
  Block *thenBlock = new Block();
  thenBlock->addArgument(stackType, loc);
  ifOp.getThenRegion().push_back(thenBlock);

  builder.setInsertionPointToStart(thenBlock);
  Value thenArg = thenBlock->getArgument(0);
  // Drop the flag from the block arg.
  Value thenStack =
      builder.create<forth::DropOp>(loc, stackType, thenArg).getResult();
  if (failed(parseBody(thenStack, isElseOrThen)))
    return nullptr;
  builder.create<forth::YieldOp>(getLoc(), thenStack);

  // --- Else region ---
  Block *elseBlock = new Block();
  elseBlock->addArgument(stackType, loc);
  ifOp.getElseRegion().push_back(elseBlock);

  if (currentToken.kind == Token::Kind::Word && currentToken.text == "ELSE") {
    consume(); // consume ELSE
    builder.setInsertionPointToStart(elseBlock);
    Value elseArg = elseBlock->getArgument(0);
    Value elseStack =
        builder.create<forth::DropOp>(loc, stackType, elseArg).getResult();
    if (failed(parseBody(elseStack, isThen)))
      return nullptr;
    builder.create<forth::YieldOp>(getLoc(), elseStack);
  } else {
    // No ELSE clause — just drop the flag and yield (identity).
    builder.setInsertionPointToStart(elseBlock);
    Value elseArg = elseBlock->getArgument(0);
    Value elseStack =
        builder.create<forth::DropOp>(loc, stackType, elseArg).getResult();
    builder.create<forth::YieldOp>(loc, elseStack);
  }

  // Consume THEN.
  if (currentToken.kind != Token::Kind::Word || currentToken.text != "THEN") {
    (void)emitError("expected 'THEN'");
    return nullptr;
  }
  consume(); // consume THEN

  // Restore insertion point to after the forth.if op.
  builder.setInsertionPointAfter(ifOp);
  return ifOp.getOutputStack();
}

//===----------------------------------------------------------------------===//
// BEGIN / UNTIL parsing.
//===----------------------------------------------------------------------===//

Value ForthParser::parseBeginUntil(Value inputStack, Location loc) {
  Type stackType = forth::StackType::get(context);

  // Create forth.begin_until op.
  auto beginUntilOp =
      builder.create<forth::BeginUntilOp>(loc, stackType, inputStack);

  auto isUntil = [](StringRef word) { return word == "UNTIL"; };

  // --- Body region ---
  Block *bodyBlock = new Block();
  bodyBlock->addArgument(stackType, loc);
  beginUntilOp.getBodyRegion().push_back(bodyBlock);

  builder.setInsertionPointToStart(bodyBlock);
  Value bodyStack = bodyBlock->getArgument(0);
  if (failed(parseBody(bodyStack, isUntil)))
    return nullptr;
  builder.create<forth::YieldOp>(getLoc(), bodyStack);

  // Consume UNTIL.
  if (currentToken.kind != Token::Kind::Word || currentToken.text != "UNTIL") {
    (void)emitError("expected 'UNTIL'");
    return nullptr;
  }
  consume(); // consume UNTIL

  // Restore insertion point to after the forth.begin_until op.
  builder.setInsertionPointAfter(beginUntilOp);
  return beginUntilOp.getOutputStack();
}

//===----------------------------------------------------------------------===//
// DO / LOOP parsing.
//===----------------------------------------------------------------------===//

Value ForthParser::parseDoLoop(Value inputStack, Location loc) {
  Type stackType = forth::StackType::get(context);

  // Create forth.do_loop op.
  auto doLoopOp = builder.create<forth::DoLoopOp>(loc, stackType, inputStack);

  auto isLoop = [](StringRef word) { return word == "LOOP"; };

  // --- Body region ---
  Block *bodyBlock = new Block();
  bodyBlock->addArgument(stackType, loc);
  doLoopOp.getBodyRegion().push_back(bodyBlock);

  builder.setInsertionPointToStart(bodyBlock);
  Value bodyStack = bodyBlock->getArgument(0);
  ++doLoopDepth;
  if (failed(parseBody(bodyStack, isLoop))) {
    --doLoopDepth;
    return nullptr;
  }
  --doLoopDepth;
  builder.create<forth::YieldOp>(getLoc(), bodyStack);

  // Consume LOOP.
  if (currentToken.kind != Token::Kind::Word || currentToken.text != "LOOP") {
    (void)emitError("expected 'LOOP'");
    return nullptr;
  }
  consume(); // consume LOOP

  // Restore insertion point to after the forth.do_loop op.
  builder.setInsertionPointAfter(doLoopOp);
  return doLoopOp.getOutputStack();
}

//===----------------------------------------------------------------------===//
// Word definition and top-level parsing.
//===----------------------------------------------------------------------===//

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

  // Parse word body until ';'
  if (failed(parseBody(resultStack, [](StringRef) { return false; })))
    return failure();

  if (currentToken.kind != Token::Kind::Semicolon) {
    return emitError("unterminated word definition: missing ';'");
  }

  // Add return
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<func::ReturnOp>(loc, resultStack);

  // Register the word
  wordDefs.insert(wordName);

  consume(); // consume ';'

  inWordDefinition = false;

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
    if (failed(parseBody(stack, [](StringRef) { return false; })))
      return failure();
  }

  return success();
}

OwningOpRef<ModuleOp> ForthParser::parseModule() {
  // Create a module to hold the parsed operations
  Location loc = getLoc();
  OwningOpRef<ModuleOp> module = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToEnd(module->getBody());

  // Pre-pass: scan for param declarations
  scanParamDeclarations();

  // First pass: parse all word definitions
  lexer.reset();
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
  lexer.reset();
  consume();

  // Reset insertion point to end of module for main function
  builder.setInsertionPointToEnd(module->getBody());

  // Build function argument types from param declarations
  SmallVector<Type> argTypes;
  for (const auto &param : paramDecls) {
    argTypes.push_back(MemRefType::get({param.size}, builder.getI64Type()));
  }

  auto funcType = builder.getFunctionType(argTypes, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "main", funcType);
  funcOp.setPrivate();

  // Annotate arguments with param names
  for (size_t i = 0; i < paramDecls.size(); ++i) {
    funcOp.setArgAttr(i, "forth.param_name",
                      builder.getStringAttr(paramDecls[i].name));
  }

  // Create the entry block with arguments
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

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
  // Ensure the Forth dialect is loaded
  context->loadDialect<forth::ForthDialect>();
  context->loadDialect<func::FuncDialect>();

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
        registry.insert<forth::ForthDialect, func::FuncDialect>();
      });
}
