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
  while (curPtr < endPtr && isWhitespace(*curPtr)) {
    ++curPtr;
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

Value ForthParser::emitOperation(StringRef word, Value inputStack) {
  Location loc = builder.getUnknownLoc();
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
  }

  // Unknown word - this is where we'd check a symbol table in the future
  return nullptr;
}

LogicalResult ForthParser::parseWordDefinition() {
  // Expect ':'
  Location loc = builder.getUnknownLoc();

  // Save current insertion point
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
  Value inputStack = entryBlock->getArgument(0);
  builder.setInsertionPointToStart(entryBlock);

  // Parse word body until ';'
  Value resultStack = inputStack;
  while (currentToken.kind != Token::Kind::Semicolon) {
    if (currentToken.kind == Token::Kind::EndOfFile) {
      return emitError("unterminated word definition: missing ';'");
    }

    if (currentToken.kind == Token::Kind::Number) {
      int64_t value = std::stoll(currentToken.text);
      resultStack =
          builder
              .create<forth::LiteralOp>(loc, stackType, resultStack,
                                        builder.getI64IntegerAttr(value))
              .getResult();
      consume();
    } else if (currentToken.kind == Token::Kind::Word) {
      Value newStack = emitOperation(currentToken.text, resultStack);
      if (!newStack) {
        return emitError("unknown word in definition: " + currentToken.text);
      }
      resultStack = newStack;
      consume();
    } else {
      return emitError("unexpected token in word definition");
    }
  }

  // Add return - move to end of block to ensure it's the terminator
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
  Location loc = builder.getUnknownLoc();

  // Start with a null stack - first operation will initialize it
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
    } else if (currentToken.kind == Token::Kind::Word &&
               currentToken.text == "param") {
      // Skip param declarations (already processed in pre-pass)
      consume(); // consume "param"
      consume(); // consume name
      consume(); // consume size
      continue;
    } else if (currentToken.kind == Token::Kind::Number) {
      // Parse numeric literal
      int64_t value = std::stoll(currentToken.text);

      stack = builder
                  .create<forth::LiteralOp>(loc, stackType, stack,
                                            builder.getI64IntegerAttr(value))
                  .getResult();

      consume();
    } else if (currentToken.kind == Token::Kind::Word) {
      // Parse operation - these operations thread the stack through
      if (!stack) {
        return emitError("operation requires a value on the stack: " +
                         currentToken.text);
      }

      Value newStack = emitOperation(currentToken.text, stack);

      if (!newStack) {
        return emitError("unknown word: " + currentToken.text);
      }

      stack = newStack;
      consume();
    }
  }

  return success();
}

OwningOpRef<ModuleOp> ForthParser::parseModule() {
  // Create a module to hold the parsed operations
  Location loc = builder.getUnknownLoc();
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
