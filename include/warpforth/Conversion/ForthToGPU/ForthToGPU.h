#ifndef WARPFORTH_CONVERSION_FORTHTOGPU_FORTHTOGPU_H
#define WARPFORTH_CONVERSION_FORTHTOGPU_FORTHTOGPU_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

namespace forth {

/// Create a pass to lower Forth operations to GPU dialect operations.
std::unique_ptr<Pass> createLowerForthToGPUPass();

/// Populate the given pattern set with patterns that lower Forth operations
/// to GPU dialect operations.
void populateForthToGPUConversionPatterns(RewritePatternSet &patterns);

} // namespace forth
} // namespace mlir

#endif // WARPFORTH_CONVERSION_FORTHTOGPU_FORTHTOGPU_H
