//===- warpforth-opt.cpp - MLIR Optimization Driver ------------*- C++ -*-===//
//
// This file implements the 'warpforth-opt' tool, which is the WarpForth analog
// of mlir-opt, used to drive MLIR passes and conversions.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "warpforth/Conversion/Passes.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::warpforth::registerConversionPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Register LLVM IR translations for GPU module serialization
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<mlir::forth::ForthDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "WarpForth optimizer driver\n", registry));
}
