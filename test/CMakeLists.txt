configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

add_custom_target(check-warpforth
  COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS warpforth-translate warpforth-opt
  COMMENT "Running WarpForth lit tests"
)
