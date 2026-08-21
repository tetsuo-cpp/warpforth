//===- warpforth-opt.cpp - MLIR Optimization Driver ------------*- C++ -*-===//
//
// This file implements the 'warpforth-opt' tool, which is the WarpForth analog
// of mlir-opt, used to drive MLIR passes and conversions.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "warpforth/Conversion/Passes.h"
#include "warpforth/Dialect/Forth/ForthDialect.h" // IWYU pragma: keep

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
