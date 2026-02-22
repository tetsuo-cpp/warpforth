\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test I8@ / I8!
\ CHECK: forth.load_i8
\ CHECK: forth.store_i8

\ Test SI8@ / SI8!
\ CHECK: forth.shared_load_i8
\ CHECK: forth.shared_store_i8

\ Test I16@ / I16!
\ CHECK: forth.load_i16
\ CHECK: forth.store_i16

\ Test SI16@ / SI16!
\ CHECK: forth.shared_load_i16
\ CHECK: forth.shared_store_i16

\ Test I32@ / I32!
\ CHECK: forth.load_i32
\ CHECK: forth.store_i32

\ Test SI32@ / SI32!
\ CHECK: forth.shared_load_i32
\ CHECK: forth.shared_store_i32

\ Test HF@ / HF!
\ CHECK: forth.load_f16
\ CHECK: forth.store_f16

\ Test SHF@ / SHF!
\ CHECK: forth.shared_load_f16
\ CHECK: forth.shared_store_f16

\ Test BF@ / BF!
\ CHECK: forth.load_bf16
\ CHECK: forth.store_bf16

\ Test SBF@ / SBF!
\ CHECK: forth.shared_load_bf16
\ CHECK: forth.shared_store_bf16

\ Test F32@ / F32!
\ CHECK: forth.load_f32
\ CHECK: forth.store_f32

\ Test SF32@ / SF32!
\ CHECK: forth.shared_load_f32
\ CHECK: forth.shared_store_f32

\! kernel main
1 I8@ 2 3 I8!
4 SI8@ 5 6 SI8!
1 I16@ 2 3 I16!
4 SI16@ 5 6 SI16!
1 I32@ 2 3 I32!
4 SI32@ 5 6 SI32!
1 HF@ 2 3 HF!
4 SHF@ 5 6 SHF!
1 BF@ 2 3 BF!
4 SBF@ 5 6 SBF!
1 F32@ 2 3 F32!
4 SF32@ 5 6 SF32!
