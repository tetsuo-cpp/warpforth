//===- warpforth-translate.cpp - WarpForth translation driver --*- C++ -*-===//
//
// This file implements the WarpForth translation tool, which translates
// Forth source code to MLIR Forth dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthImporter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "WarpForth translation tool\n");

  // Set up the input file
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up the output file
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up the MLIR context
  MLIRContext context;
  context.getOrLoadDialect<forth::ForthDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  // Set up source manager
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), SMLoc());

  // Import the Forth source code
  auto module = forth::importForth(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to import Forth source\n";
    return 1;
  }

  // Print the MLIR module
  module->print(output->os());
  output->keep();

  return 0;
}
