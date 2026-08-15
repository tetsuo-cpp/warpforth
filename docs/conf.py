project = "WarpForth"
author = "WarpForth contributors"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
master_doc = "index"
templates_path = ["_templates"]

myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

html_theme = "furo"
html_title = "WarpForth"
html_theme_options = {
    "source_repository": "https://github.com/tetsuo-cpp/warpforth/",
    "source_branch": "canon",
    "source_directory": "docs/",
}
