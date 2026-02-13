//===- MLIRToPTX.cpp - MLIR to PTX translation ---------------------------===//
//
// This file implements extraction of PTX assembly from gpu.binary operations.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Translation/MLIRToPTX/MLIRToPTX.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

LogicalResult mlir::warpforth::extractPTXFromModule(ModuleOp module,
                                                    llvm::raw_ostream &output) {
  bool foundBinary = false;

  module.walk([&](gpu::BinaryOp binaryOp) {
    foundBinary = true;

    for (auto attr : binaryOp.getObjectsAttr().getValue()) {
      auto objectAttr = dyn_cast<gpu::ObjectAttr>(attr);
      if (!objectAttr)
        continue;

      // StringAttr stores actual bytes, not escaped representation
      output << objectAttr.getObject().getValue();
    }
  });

  if (!foundBinary) {
    llvm::errs() << "error: no gpu.binary operations found in module\n";
    llvm::errs() << "hint: run 'warpforth-opt --warpforth-pipeline' first\n";
    return failure();
  }

  return success();
}

void mlir::warpforth::registerMLIRToPTXTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-ptx", "Extract PTX assembly from gpu.binary operations",
      [](Operation *op, llvm::raw_ostream &output) {
        auto module = dyn_cast<ModuleOp>(op);
        if (!module) {
          llvm::errs() << "error: expected ModuleOp at top level\n";
          return failure();
        }
        return warpforth::extractPTXFromModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<gpu::GPUDialect, NVVM::NVVMDialect>();
      });
}
