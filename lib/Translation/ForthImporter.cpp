//===- ForthImporter.cpp - Forth source to MLIR importer -------*- C++ -*-===//
//
// This file implements the Forth source code importer that translates
// Forth programs into MLIR Forth dialect operations.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Translation/ForthImporter.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Dialect/Forth/ForthOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <stack>
#include <string>
#include <vector>

namespace mlir {
namespace forth {

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

enum class TokenKind {
  Number, // Integer literal
  Word,   // Forth word (identifier)
  Eof,    // End of file
  Error   // Lexical error
};

struct Token {
  TokenKind kind;
  llvm::StringRef value;
  unsigned line;
  unsigned column;
};

class ForthLexer {
public:
  ForthLexer(llvm::StringRef input)
      : input(input), pos(0), line(1), column(1) {}

  Token getNextToken() {
    skipWhitespaceAndComments();

    if (pos >= input.size())
      return {TokenKind::Eof, "", line, column};

    unsigned startLine = line;
    unsigned startColumn = column;

    // Check for number
    if (std::isdigit(input[pos]) ||
        (input[pos] == '-' && pos + 1 < input.size() &&
         std::isdigit(input[pos + 1]))) {
      return lexNumber(startLine, startColumn);
    }

    // Otherwise it's a word
    return lexWord(startLine, startColumn);
  }

private:
  void skipWhitespaceAndComments() {
    while (pos < input.size()) {
      // Skip whitespace
      if (std::isspace(input[pos])) {
        if (input[pos] == '\n') {
          line++;
          column = 1;
        } else {
          column++;
        }
        pos++;
        continue;
      }

      // Skip line comment (\ to end of line)
      if (input[pos] == '\\') {
        while (pos < input.size() && input[pos] != '\n')
          pos++;
        continue;
      }

      // Skip parenthetical comment
      if (input[pos] == '(') {
        pos++;
        column++;
        int depth = 1;
        while (pos < input.size() && depth > 0) {
          if (input[pos] == '(')
            depth++;
          else if (input[pos] == ')')
            depth--;

          if (input[pos] == '\n') {
            line++;
            column = 1;
          } else {
            column++;
          }
          pos++;
        }
        continue;
      }

      break;
    }
  }

  Token lexNumber(unsigned startLine, unsigned startColumn) {
    size_t start = pos;

    // Handle negative sign
    if (input[pos] == '-') {
      pos++;
      column++;
    }

    // Read digits
    while (pos < input.size() && std::isdigit(input[pos])) {
      pos++;
      column++;
    }

    return {TokenKind::Number, input.slice(start, pos), startLine, startColumn};
  }

  Token lexWord(unsigned startLine, unsigned startColumn) {
    size_t start = pos;

    // Read word characters (non-whitespace, non-comment)
    while (pos < input.size() && !std::isspace(input[pos]) &&
           input[pos] != '(' && input[pos] != '\\') {
      pos++;
      column++;
    }

    auto value = input.slice(start, pos);
    if (value.empty())
      return {TokenKind::Error, "Empty word", startLine, startColumn};

    return {TokenKind::Word, value, startLine, startColumn};
  }

  llvm::StringRef input;
  size_t pos;
  unsigned line;
  unsigned column;
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

class ForthParser {
public:
  ForthParser(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context)
      : sourceMgr(sourceMgr), context(context), builder(context),
        lexer(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()) {

    // Load the Forth dialect
    context->getOrLoadDialect<ForthDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();

    currentToken = lexer.getNextToken();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseModule() {
    auto loc = mlir::UnknownLoc::get(context);
    auto module = mlir::ModuleOp::create(loc);

    builder.setInsertionPointToEnd(module.getBody());

    // Create a main function to hold the Forth code
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "forth_main", funcType);
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Parse the Forth code
    if (!parseForthCode()) {
      return nullptr;
    }

    // Add return at end
    builder.create<mlir::func::ReturnOp>(loc);

    module.push_back(func);
    return module;
  }

private:
  bool parseForthCode() {
    std::stack<mlir::Value> valueStack;

    while (currentToken.kind != TokenKind::Eof) {
      if (currentToken.kind == TokenKind::Error) {
        emitError(("Lexical error: " + currentToken.value).str());
        return false;
      }

      if (currentToken.kind == TokenKind::Number) {
        // Push constant onto stack
        if (!parseNumber(valueStack)) {
          return false;
        }
      } else if (currentToken.kind == TokenKind::Word) {
        // Process Forth word
        if (!parseWord(valueStack)) {
          return false;
        }
      }

      currentToken = lexer.getNextToken();
    }

    return true;
  }

  bool parseNumber(std::stack<mlir::Value> &stack) {
    auto loc = getLocation();
    int64_t value;

    if (currentToken.value.getAsInteger(10, value)) {
      emitError("Invalid number: " + currentToken.value.str());
      return false;
    }

    auto constantOp = builder.create<ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(value));

    stack.push(constantOp.getResult());
    return true;
  }

  bool parseWord(std::stack<mlir::Value> &stack) {
    llvm::StringRef word = currentToken.value;

    auto loc = getLocation();

    // Arithmetic operations
    if (word == "+") {
      return parseBinaryOp<AddOp>(stack, loc);
    } else if (word == "-") {
      return parseBinaryOp<SubOp>(stack, loc);
    } else if (word == "*") {
      return parseBinaryOp<MulOp>(stack, loc);
    } else if (word == "/") {
      return parseBinaryOp<DivOp>(stack, loc);
    }
    // Stack operations (case-insensitive)
    else if (word.equals_insensitive("DUP")) {
      return parseDup(stack, loc);
    } else if (word.equals_insensitive("DROP")) {
      return parseDrop(stack, loc);
    } else if (word.equals_insensitive("SWAP")) {
      return parseSwap(stack, loc);
    } else {
      emitError(("Unknown Forth word: " + currentToken.value).str());
      return false;
    }
  }

  template <typename OpType>
  bool parseBinaryOp(std::stack<mlir::Value> &stack, mlir::Location loc) {
    if (stack.size() < 2) {
      emitError("Stack underflow: need 2 values for binary operation");
      return false;
    }

    mlir::Value rhs = stack.top();
    stack.pop();
    mlir::Value lhs = stack.top();
    stack.pop();

    // Binary operations require explicit result type
    auto result = builder.create<OpType>(loc, lhs.getType(), lhs, rhs);
    stack.push(result.getResult());
    return true;
  }

  bool parseDup(std::stack<mlir::Value> &stack, mlir::Location loc) {
    if (stack.empty()) {
      emitError("Stack underflow: need 1 value for DUP");
      return false;
    }

    mlir::Value top = stack.top();
    auto result = builder.create<DupOp>(loc, top.getType(), top);
    stack.push(result.getResult());
    return true;
  }

  bool parseDrop(std::stack<mlir::Value> &stack, mlir::Location loc) {
    if (stack.empty()) {
      emitError("Stack underflow: need 1 value for DROP");
      return false;
    }

    mlir::Value top = stack.top();
    stack.pop();
    builder.create<DropOp>(loc, top);
    return true;
  }

  bool parseSwap(std::stack<mlir::Value> &stack, mlir::Location loc) {
    if (stack.size() < 2) {
      emitError("Stack underflow: need 2 values for SWAP");
      return false;
    }

    mlir::Value second = stack.top();
    stack.pop();
    mlir::Value first = stack.top();
    stack.pop();

    // Swap operation requires explicit result types
    llvm::SmallVector<mlir::Type, 2> resultTypes = {second.getType(),
                                                    first.getType()};
    auto result = builder.create<SwapOp>(loc, resultTypes, first, second);
    stack.push(result.getResult(0)); // second becomes first
    stack.push(result.getResult(1)); // first becomes second
    return true;
  }

  mlir::Location getLocation() {
    return mlir::FileLineColLoc::get(
        builder.getStringAttr(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())
                ->getBufferIdentifier()),
        currentToken.line, currentToken.column);
  }

  void emitError(const std::string &message) {
    llvm::errs() << "Error at line " << currentToken.line << ", column "
                 << currentToken.column << ": " << message << "\n";
  }

  llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  ForthLexer lexer;
  Token currentToken;
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp> importForth(llvm::SourceMgr &sourceMgr,
                                              mlir::MLIRContext *context) {
  ForthParser parser(sourceMgr, context);
  return parser.parseModule();
}

} // namespace forth
} // namespace mlir
