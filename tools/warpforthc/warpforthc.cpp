//===- warpforthc.cpp - WarpForth compiler driver -------------------------===//
//
// Compiles Forth source to PTX assembly in a single process.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "warpforth/Conversion/Passes.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"
#include "warpforth/Translation/MLIRToPTX/MLIRToPTX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input>"),
                                                llvm::cl::Required);

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "WarpForth compiler: Forth to PTX\n");

  // Set up source manager with input file
  auto inputFile = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = inputFile.getError()) {
    llvm::errs() << "error: could not open input file '" << inputFilename
                 << "': " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*inputFile), llvm::SMLoc());

  // Parse Forth source to MLIR
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<forth::ForthDialect>();

  MLIRContext context(registry);
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  auto module = forth::parseForthSource(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse Forth source\n";
    return 1;
  }

  // Run the compilation pipeline
  PassManager pm(&context);
  (void)applyPassManagerCLOptions(pm);
  warpforth::buildWarpForthPipeline(pm);
  if (failed(pm.run(*module))) {
    llvm::errs() << "error: compilation pipeline failed\n";
    return 1;
  }

  // Extract PTX and write to output
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: could not open output file '" << outputFilename
                 << "': " << ec.message() << "\n";
    return 1;
  }

  if (failed(warpforth::extractPTXFromModule(*module, outputFile.os()))) {
    return 1;
  }

  outputFile.keep();
  return 0;
}
