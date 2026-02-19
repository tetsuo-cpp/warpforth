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
#include "llvm/ADT/StringMap.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace forth {

/// A declared kernel parameter: `param <name> <type>`.
struct ParamDecl {
  std::string name;
  bool isArray = false;
  int64_t size = 0;
};

/// A declared shared memory region: `shared <name> <type>`.
/// TODO: Not yet consumed — scaffolding for shared memory support.
struct SharedDecl {
  std::string name;
  bool isArray = false;
  int64_t size = 0;
};

/// Simple token representing a Forth word or literal.
struct Token {
  enum class Kind { Number, Word, Colon, Semicolon, EndOfFile };

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

  /// Reset lexer to beginning of buffer.
  void reset();

  /// Reset lexer to a specific position in the buffer.
  void resetTo(const char *ptr);

private:
  llvm::SourceMgr &sourceMgr;
  unsigned bufferID;
  const char *curPtr;
  const char *endPtr;

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
  std::unordered_set<std::string> wordDefs;
  std::vector<ParamDecl> paramDecls;
  std::vector<SharedDecl> sharedDecls;
  llvm::StringMap<Value> sharedAllocs;
  std::string kernelName;
  const char *headerEndPtr = nullptr;
  bool inWordDefinition = false;

  /// Control flow stack tag: Orig = forward reference (IF->THEN, WHILE->exit),
  /// Dest = backward reference (BEGIN->loop header).
  enum class CFTag { Orig, Dest };

  /// Control flow stack for IF/ELSE/THEN, BEGIN/UNTIL, BEGIN/WHILE/REPEAT.
  SmallVector<std::pair<CFTag, Block *>> cfStack;

  /// Loop context for DO/LOOP with I/J/K support.
  struct LoopContext {
    Value counter; // memref<1xi64> alloca for the loop counter
    Value limit;   // i64 loop limit
    Block *body;   // loop body block
    Block *exit;   // loop exit block
  };
  SmallVector<LoopContext> loopStack;

  /// Parse the kernel header (\\! directives) and record metadata.
  LogicalResult parseHeader();

  /// Advance to the next token.
  void consume();

  /// Emit an error at the current location.
  LogicalResult emitError(const llvm::Twine &message);

  /// Emit an error at a specific buffer location.
  LogicalResult emitErrorAt(llvm::SMLoc loc, const llvm::Twine &message);

  /// Parse a sequence of Forth operations.
  LogicalResult parseOperations(Value &stack);

  /// Convert the current token's SMLoc to an MLIR FileLineColLoc.
  Location getLoc();

  /// Emit a Forth operation based on the current token.
  /// Returns the updated stack value or nullptr on error.
  Value emitOperation(StringRef word, Value inputStack, Location loc);

  /// Create a new block with a single !forth.stack argument, appended to
  /// the given region.
  Block *createStackBlock(Region *region, Location loc);

  /// Pop a flag from the stack: emits forth.pop_flag and returns
  /// the updated stack and the i1 flag value.
  std::pair<Value, Value> emitPopFlag(Location loc, Value stack);

  /// Parse a sequence of Forth operations, handling control flow inline.
  LogicalResult parseBody(Value &stack);

  /// Emit the common loop-end logic for LOOP and +LOOP:
  /// load counter, add step, store, crossing test, cond_br to exit or body.
  void emitLoopEnd(Location loc, const LoopContext &ctx, Value step,
                   Value &stack);

  /// Parse a user-defined word definition.
  LogicalResult parseWordDefinition();
};

} // namespace forth
} // namespace mlir
