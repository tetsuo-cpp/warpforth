//===- ForthToMLIR.cpp - Forth to MLIR translation -----------------------===//
//
// This file implements the Forth-to-MLIR translation.
//
//===----------------------------------------------------------------------===//

#include "ForthToMLIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>

using namespace mlir;
using namespace mlir::forth;

//===----------------------------------------------------------------------===//
// ForthLexer implementation
//===----------------------------------------------------------------------===//

ForthLexer::ForthLexer(llvm::SourceMgr &sourceMgr, unsigned bufferID) {
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

  // Read until whitespace
  while (curPtr < endPtr && !isWhitespace(*curPtr)) {
    ++curPtr;
  }

  std::string text(tokenStart, curPtr - tokenStart);

  // Determine token kind
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

Value ForthParser::emitOperation(StringRef word, Value inputStack) {
  Location loc = builder.getUnknownLoc();
  Type stackType = forth::StackType::get(context);

  // Map Forth words to operations
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
  }

  // Unknown word - this is where we'd check a symbol table in the future
  return nullptr;
}

LogicalResult ForthParser::parseOperations(Value &stack) {
  Type stackType = forth::StackType::get(context);
  Location loc = builder.getUnknownLoc();

  // Start with a null stack - first operation will initialize it
  stack = builder.create<forth::StackOp>(loc, stackType);

  while (currentToken.kind != Token::Kind::EndOfFile) {
    if (currentToken.kind == Token::Kind::Number) {
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

  // Create a main function to hold the Forth code
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "main", funcType);
  funcOp.setPrivate();

  // Create the entry block
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

namespace {
/// Parse Forth source and convert to MLIR.
OwningOpRef<Operation *> parseForthSource(llvm::SourceMgr &sourceMgr,
                                          MLIRContext *context) {
  // Ensure the Forth dialect is loaded
  context->loadDialect<forth::ForthDialect>();
  context->loadDialect<func::FuncDialect>();

  // Create parser and parse the module
  ForthParser parser(sourceMgr, context);
  return parser.parseModule();
}
} // namespace

void mlir::forth::registerForthToMLIRTranslation() {
  TranslateToMLIRRegistration registration(
      "forth-to-mlir", "Translate Forth source to MLIR",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return parseForthSource(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        registry.insert<forth::ForthDialect, func::FuncDialect>();
      });
}
