// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// `ADDR load ADDR store` for each width. A load widens the element to a cell:
// an integer sign-extends, and a float extends to f64 and travels as bits. A
// store narrows the cell back. Global words address !llvm.ptr<1>.
// CHECK-LABEL: func.func @global_memory(
// CHECK-SAME: %[[ADDR:[^:]*]]: i64
// CHECK: %[[I64_PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
// CHECK: %[[I64:.*]] = llvm.load %[[I64_PTR]] : !llvm.ptr<1> -> i64
// CHECK: llvm.store %[[I64]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: %[[F64:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> f64
// CHECK: %[[F64_CELL:.*]] = arith.bitcast %[[F64]] : f64 to i64
// CHECK: %[[F64_BACK:.*]] = arith.bitcast %[[F64_CELL]] : i64 to f64
// CHECK: llvm.store %[[F64_BACK]], %{{.*}} : f64, !llvm.ptr<1>
// CHECK: %[[I8:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i8
// CHECK: %[[I8_CELL:.*]] = arith.extsi %[[I8]] : i8 to i64
// CHECK: %[[I8_BACK:.*]] = arith.trunci %[[I8_CELL]] : i64 to i8
// CHECK: llvm.store %[[I8_BACK]], %{{.*}} : i8, !llvm.ptr<1>
// CHECK: %[[I16:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i16
// CHECK: %[[I16_CELL:.*]] = arith.extsi %[[I16]] : i16 to i64
// CHECK: %[[I16_BACK:.*]] = arith.trunci %[[I16_CELL]] : i64 to i16
// CHECK: llvm.store %[[I16_BACK]], %{{.*}} : i16, !llvm.ptr<1>
// CHECK: %[[I32:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i32
// CHECK: %[[I32_CELL:.*]] = arith.extsi %[[I32]] : i32 to i64
// CHECK: %[[I32_BACK:.*]] = arith.trunci %[[I32_CELL]] : i64 to i32
// CHECK: llvm.store %[[I32_BACK]], %{{.*}} : i32, !llvm.ptr<1>
// CHECK: %[[F16:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> f16
// CHECK: %[[F16_WIDE:.*]] = arith.extf %[[F16]] : f16 to f64
// CHECK: %[[F16_CELL:.*]] = arith.bitcast %[[F16_WIDE]] : f64 to i64
// CHECK: %[[F16_F64:.*]] = arith.bitcast %[[F16_CELL]] : i64 to f64
// CHECK: %[[F16_BACK:.*]] = arith.truncf %[[F16_F64]] : f64 to f16
// CHECK: llvm.store %[[F16_BACK]], %{{.*}} : f16, !llvm.ptr<1>
// CHECK: %[[BF16:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> bf16
// CHECK: %[[BF16_WIDE:.*]] = arith.extf %[[BF16]] : bf16 to f64
// CHECK: %[[BF16_CELL:.*]] = arith.bitcast %[[BF16_WIDE]] : f64 to i64
// CHECK: %[[BF16_F64:.*]] = arith.bitcast %[[BF16_CELL]] : i64 to f64
// CHECK: %[[BF16_BACK:.*]] = arith.truncf %[[BF16_F64]] : f64 to bf16
// CHECK: llvm.store %[[BF16_BACK]], %{{.*}} : bf16, !llvm.ptr<1>
// CHECK: %[[F32:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> f32
// CHECK: %[[F32_WIDE:.*]] = arith.extf %[[F32]] : f32 to f64
// CHECK: %[[F32_CELL:.*]] = arith.bitcast %[[F32_WIDE]] : f64 to i64
// CHECK: %[[F32_F64:.*]] = arith.bitcast %[[F32_CELL]] : i64 to f64
// CHECK: %[[F32_BACK:.*]] = arith.truncf %[[F32_F64]] : f64 to f32
// CHECK: llvm.store %[[F32_BACK]], %{{.*}} : f32, !llvm.ptr<1>
func.func @global_memory(%addr: i64) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %addr : !forth.stack, i64 -> !forth.stack
  %s2 = forth.loadi %s1 : !forth.stack -> !forth.stack
  %s3 = forth.param_ref %s2 %addr : !forth.stack, i64 -> !forth.stack
  %s4 = forth.storei %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %addr : !forth.stack, i64 -> !forth.stack
  %s6 = forth.loadf %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %addr : !forth.stack, i64 -> !forth.stack
  %s8 = forth.storef %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %addr : !forth.stack, i64 -> !forth.stack
  %s10 = forth.load_i8 %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %addr : !forth.stack, i64 -> !forth.stack
  %s12 = forth.store_i8 %s11 : !forth.stack -> !forth.stack
  %s13 = forth.param_ref %s12 %addr : !forth.stack, i64 -> !forth.stack
  %s14 = forth.load_i16 %s13 : !forth.stack -> !forth.stack
  %s15 = forth.param_ref %s14 %addr : !forth.stack, i64 -> !forth.stack
  %s16 = forth.store_i16 %s15 : !forth.stack -> !forth.stack
  %s17 = forth.param_ref %s16 %addr : !forth.stack, i64 -> !forth.stack
  %s18 = forth.load_i32 %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %addr : !forth.stack, i64 -> !forth.stack
  %s20 = forth.store_i32 %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %addr : !forth.stack, i64 -> !forth.stack
  %s22 = forth.load_f16 %s21 : !forth.stack -> !forth.stack
  %s23 = forth.param_ref %s22 %addr : !forth.stack, i64 -> !forth.stack
  %s24 = forth.store_f16 %s23 : !forth.stack -> !forth.stack
  %s25 = forth.param_ref %s24 %addr : !forth.stack, i64 -> !forth.stack
  %s26 = forth.load_bf16 %s25 : !forth.stack -> !forth.stack
  %s27 = forth.param_ref %s26 %addr : !forth.stack, i64 -> !forth.stack
  %s28 = forth.store_bf16 %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %addr : !forth.stack, i64 -> !forth.stack
  %s30 = forth.load_f32 %s29 : !forth.stack -> !forth.stack
  %s31 = forth.param_ref %s30 %addr : !forth.stack, i64 -> !forth.stack
  %s32 = forth.store_f32 %s31 : !forth.stack -> !forth.stack
  return
}

// Shared words address !llvm.ptr<3> and convert cells like the global ones.
// CHECK-LABEL: func.func @shared_memory(
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i64
// CHECK: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> f64
// CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i8
// CHECK: llvm.store %{{.*}}, %{{.*}} : i8, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i16
// CHECK: llvm.store %{{.*}}, %{{.*}} : i16, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> f16
// CHECK: llvm.store %{{.*}}, %{{.*}} : f16, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> bf16
// CHECK: llvm.store %{{.*}}, %{{.*}} : bf16, !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> f32
// CHECK: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<3>
func.func @shared_memory(%addr: i64) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %addr : !forth.stack, i64 -> !forth.stack
  %s2 = forth.shared_loadi %s1 : !forth.stack -> !forth.stack
  %s3 = forth.param_ref %s2 %addr : !forth.stack, i64 -> !forth.stack
  %s4 = forth.shared_storei %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %addr : !forth.stack, i64 -> !forth.stack
  %s6 = forth.shared_loadf %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %addr : !forth.stack, i64 -> !forth.stack
  %s8 = forth.shared_storef %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %addr : !forth.stack, i64 -> !forth.stack
  %s10 = forth.shared_load_i8 %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %addr : !forth.stack, i64 -> !forth.stack
  %s12 = forth.shared_store_i8 %s11 : !forth.stack -> !forth.stack
  %s13 = forth.param_ref %s12 %addr : !forth.stack, i64 -> !forth.stack
  %s14 = forth.shared_load_i16 %s13 : !forth.stack -> !forth.stack
  %s15 = forth.param_ref %s14 %addr : !forth.stack, i64 -> !forth.stack
  %s16 = forth.shared_store_i16 %s15 : !forth.stack -> !forth.stack
  %s17 = forth.param_ref %s16 %addr : !forth.stack, i64 -> !forth.stack
  %s18 = forth.shared_load_i32 %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %addr : !forth.stack, i64 -> !forth.stack
  %s20 = forth.shared_store_i32 %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %addr : !forth.stack, i64 -> !forth.stack
  %s22 = forth.shared_load_f16 %s21 : !forth.stack -> !forth.stack
  %s23 = forth.param_ref %s22 %addr : !forth.stack, i64 -> !forth.stack
  %s24 = forth.shared_store_f16 %s23 : !forth.stack -> !forth.stack
  %s25 = forth.param_ref %s24 %addr : !forth.stack, i64 -> !forth.stack
  %s26 = forth.shared_load_bf16 %s25 : !forth.stack -> !forth.stack
  %s27 = forth.param_ref %s26 %addr : !forth.stack, i64 -> !forth.stack
  %s28 = forth.shared_store_bf16 %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %addr : !forth.stack, i64 -> !forth.stack
  %s30 = forth.shared_load_f32 %s29 : !forth.stack -> !forth.stack
  %s31 = forth.param_ref %s30 %addr : !forth.stack, i64 -> !forth.stack
  %s32 = forth.shared_store_f32 %s31 : !forth.stack -> !forth.stack
  return
}
