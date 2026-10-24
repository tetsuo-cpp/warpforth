//===- warpforthc.cpp - WarpForth compiler driver -------------------------===//
//
// Compiles Forth source to PTX assembly in a single process.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "warpforth/Conversion/Passes.h"
#include "warpforth/Translation/ForthToMLIR/ForthToMLIR.h"
#include "warpforth/Translation/MLIRToPTX/MLIRToPTX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input>"),
                                                llvm::cl::Required);

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string, false, warpforth::NVVMChipParser>
    targetArchitecture("arch", llvm::cl::desc("GPU architecture"),
                       llvm::cl::value_desc("sm_XX"), llvm::cl::init("sm_70"));

static llvm::cl::opt<std::string>
    optLevel("O", llvm::cl::desc("Optimization level (0, 1, 2, or 3)"),
             llvm::cl::value_desc("level"), llvm::cl::Prefix,
             llvm::cl::init(""));
static llvm::cl::alias optLevelAlias("opt-level",
                                     llvm::cl::desc("Alias for -O"),
                                     llvm::cl::aliasopt(optLevel));

static llvm::cl::opt<std::string>
    targetFeatures("features", llvm::cl::desc("NVVM target features"),
                   llvm::cl::value_desc("+ptxNN"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    libdevicePath("libdevice", llvm::cl::desc("Path to the libdevice bitcode"),
                  llvm::cl::value_desc("path"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  warpforth::registerConversionPasses();
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "WarpForth compiler: Forth to PTX\n");

  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  DialectRegistry registry;
  arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  func::registerInlinerExtension(registry);

  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  NVVM::registerNVVMTargetInterfaceExternalModels(registry);

  MLIRContext context(registry);
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  auto module = forth::parseForthSource(sourceMgr, &context);
  if (!module)
    return 1;

  PassManager pm(&context);
  if (failed(applyPassManagerCLOptions(pm)))
    return 1;
  warpforth::WarpForthPipelineOptions pipelineOptions;
  std::string overrides;
  llvm::raw_string_ostream overridesOS(overrides);
  if (!optLevel.empty())
    overridesOS << " opt-level=" << optLevel;
  if (!targetFeatures.empty())
    overridesOS << " features=" << targetFeatures;
  if (!libdevicePath.empty())
    overridesOS << " libdevice=" << libdevicePath;
  if (failed(pipelineOptions.parseFromString(overrides)))
    return 1;
  pipelineOptions.chip = targetArchitecture.getValue();
  warpforth::buildWarpForthPipeline(pm, pipelineOptions);
  if (failed(pm.run(*module)))
    return 1;

  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << "error: " << errorMessage << "\n";
    return 1;
  }

  if (failed(warpforth::extractPTXFromModule(*module, outputFile->os())))
    return 1;

  outputFile->keep();
  return 0;
}
