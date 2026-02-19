\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.thread_id_x
\ CHECK: forth.thread_id_y
\ CHECK: forth.thread_id_z
\ CHECK: forth.block_id_x
\ CHECK: forth.block_id_y
\ CHECK: forth.block_id_z
\ CHECK: forth.block_dim_x
\ CHECK: forth.block_dim_y
\ CHECK: forth.block_dim_z
\ CHECK: forth.grid_dim_x
\ CHECK: forth.grid_dim_y
\ CHECK: forth.grid_dim_z
\ CHECK: forth.global_id
\! kernel main
TID-X TID-Y TID-Z
BID-X BID-Y BID-Z
BDIM-X BDIM-Y BDIM-Z
GDIM-X GDIM-Y GDIM-Z
GLOBAL-ID
