import os
import lit.formats

config.name = "WarpForth"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".forth", ".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Tool substitutions
config.substitutions.append(
    ("%warpforth-translate", os.path.join(config.warpforth_bin_root, "bin", "warpforth-translate"))
)
config.substitutions.append(
    ("%warpforth-opt", os.path.join(config.warpforth_bin_root, "bin", "warpforth-opt"))
)
config.substitutions.append(
    ("%FileCheck", config.filecheck_path)
)
config.substitutions.append(
    ("%not", os.path.join(os.path.dirname(config.filecheck_path), "not"))
)
