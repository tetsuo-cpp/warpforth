//===- warpforth-opt.cpp - WarpForth optimizer driver ----------*- C++ -*-===//
//
// This file implements the WarpForth optimizer tool, which is used to test
// and apply transformations and lowerings on Forth dialect operations.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Conversion/ForthToGPU/ForthToGPU.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Dialect/Forth/ForthOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::forth::ForthDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::gpu::GPUDialect>();

  // Register standard MLIR dialects
  mlir::registerAllDialects(registry);

  // Register Forth-specific passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::forth::createLowerForthToGPUPass();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "WarpForth optimizer driver\n", registry));
}
