//===- ForthToMLIR.h - Forth to MLIR translation (private) ------*- C++ -*-===//
//
// Private header for Forth-to-MLIR translation implementation.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include <string>
#include <vector>

namespace mlir {
namespace forth {

/// Simple token representing a Forth word or literal.
struct Token {
  enum class Kind { Number, Word, EndOfFile };

  Kind kind;
  std::string text;
  llvm::SMLoc location;

  Token(Kind k, std::string t, llvm::SMLoc loc)
      : kind(k), text(std::move(t)), location(loc) {}
};

/// Simple tokenizer for Forth source code.
/// Splits on whitespace and identifies numbers vs. words.
class ForthLexer {
public:
  ForthLexer(llvm::SourceMgr &sourceMgr, unsigned bufferID);

  /// Get the next token from the input.
  Token nextToken();

private:
  llvm::SourceMgr &sourceMgr;
  const char *curPtr;
  const char *endPtr;
  unsigned bufferID;

  /// Skip whitespace and comments.
  void skipWhitespace();

  /// Check if a character is whitespace.
  bool isWhitespace(char c) const;

  /// Check if a string is a number.
  bool isNumber(const std::string &str) const;
};

/// Parser and translator for Forth source code to MLIR.
class ForthParser {
public:
  ForthParser(llvm::SourceMgr &sourceMgr, MLIRContext *context);

  /// Parse Forth source and generate MLIR module.
  OwningOpRef<ModuleOp> parseModule();

private:
  llvm::SourceMgr &sourceMgr;
  MLIRContext *context;
  OpBuilder builder;
  ForthLexer lexer;
  Token currentToken;

  /// Advance to the next token.
  void consume();

  /// Emit an error at the current location.
  LogicalResult emitError(const llvm::Twine &message);

  /// Parse a sequence of Forth operations.
  LogicalResult parseOperations(Value &stack);

  /// Emit a Forth operation based on the current token.
  /// Returns the updated stack value or nullptr on error.
  Value emitOperation(StringRef word, Value inputStack);
};

} // namespace forth
} // namespace mlir
