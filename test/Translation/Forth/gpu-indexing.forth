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
tid-x tid-y tid-z
bid-x bid-y bid-z
bdim-x bdim-y bdim-z
gdim-x gdim-y gdim-z
global-id
