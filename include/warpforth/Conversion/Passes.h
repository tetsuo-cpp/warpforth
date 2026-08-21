//===- Passes.h - Conversion Pass Registration -----------------*- C++ -*-===//
//
// This file declares the registration functions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_CONVERSION_PASSES_H
#define WARPFORTH_CONVERSION_PASSES_H

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CodeGen.h"
#include <string>

namespace mlir {
class OpPassManager;

namespace warpforth {

/// Command-line parser for a supported NVVM chip.
class NVVMChipParser : public llvm::cl::parser<std::string> {
public:
  using parser::parser;

  bool parse(llvm::cl::Option &option, StringRef argName, StringRef arg,
             std::string &value);
};

/// Target configuration for the WarpForth compilation pipeline.
struct WarpForthPipelineOptions
    : public PassPipelineOptions<WarpForthPipelineOptions> {
  WarpForthPipelineOptions();

  /// NVVM target chip, such as `sm_70`.
  PassOptions::Option<std::string, NVVMChipParser> chip;
  /// NVVM target features, such as `+ptx60`.
  PassOptions::Option<std::string> features;
  /// Path to the CUDA libdevice bitcode library.
  PassOptions::Option<std::string> libdevicePath;
  /// NVVM target optimization level.
  PassOptions::Option<llvm::CodeGenOptLevel> optLevel;
  /// GPU output format: LLVM bitcode, assembly, binary, or fat binary.
  PassOptions::Option<gpu::CompilationTarget> compilationTarget;
};

/// Register all conversion passes.
void registerConversionPasses();

/// Build the WarpForth compilation pipeline (Forth to PTX).
void buildWarpForthPipeline(OpPassManager &pm);
void buildWarpForthPipeline(OpPassManager &pm,
                            const WarpForthPipelineOptions &options);

} // namespace warpforth
} // namespace mlir

#endif
