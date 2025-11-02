//===- warpforth-translate.cpp -------------------------------------------===//
//
// This file implements the warpforth-translate tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllTranslations();

  // Register the Forth dialect
  registry.insert<mlir::forth::ForthDialect>();

  // Register Forth-to-MLIR translation
  mlir::forth::registerForthToMLIRTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "WarpForth Translation Tool"));
}
