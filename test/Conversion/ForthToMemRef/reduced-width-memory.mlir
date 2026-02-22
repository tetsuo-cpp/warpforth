// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// --- i8 load: llvm.load i8, extsi to i64 ---
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK-NEXT: llvm.load %{{.*}} : !llvm.ptr -> i8
// CHECK-NEXT: arith.extsi %{{.*}} : i8 to i64

// --- i8 store: trunci i64 to i8, llvm.store ---
// CHECK: arith.trunci %{{.*}} : i64 to i8
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : i8, !llvm.ptr

// --- shared i8 load: ptr<3> ---
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<3>
// CHECK-NEXT: llvm.load %{{.*}} : !llvm.ptr<3> -> i8
// CHECK-NEXT: arith.extsi %{{.*}} : i8 to i64

// --- shared i8 store: ptr<3> ---
// CHECK: arith.trunci %{{.*}} : i64 to i8
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : i8, !llvm.ptr<3>

// --- i16 load ---
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i16
// CHECK-NEXT: arith.extsi %{{.*}} : i16 to i64

// --- i16 store ---
// CHECK: arith.trunci %{{.*}} : i64 to i16
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : i16, !llvm.ptr

// --- i32 load ---
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT: arith.extsi %{{.*}} : i32 to i64

// --- i32 store ---
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr

// --- f16 load: llvm.load f16, extf to f64, bitcast to i64 ---
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> f16
// CHECK-NEXT: arith.extf %{{.*}} : f16 to f64
// CHECK-NEXT: arith.bitcast %{{.*}} : f64 to i64

// --- f16 store: bitcast i64 to f64, truncf to f16, llvm.store ---
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT: arith.truncf %{{.*}} : f64 to f16
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : f16, !llvm.ptr

// --- bf16 load ---
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> bf16
// CHECK-NEXT: arith.extf %{{.*}} : bf16 to f64
// CHECK-NEXT: arith.bitcast %{{.*}} : f64 to i64

// --- bf16 store ---
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT: arith.truncf %{{.*}} : f64 to bf16
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : bf16, !llvm.ptr

// --- f32 load ---
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> f32
// CHECK-NEXT: arith.extf %{{.*}} : f32 to f64
// CHECK-NEXT: arith.bitcast %{{.*}} : f64 to i64

// --- f32 store ---
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT: arith.truncf %{{.*}} : f64 to f32
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    // i8 load
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %2 = forth.load_i8 %1 : !forth.stack -> !forth.stack
    // i8 store
    %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(100 : i64) : !forth.stack -> !forth.stack
    %5 = forth.store_i8 %4 : !forth.stack -> !forth.stack
    // shared i8 load
    %6 = forth.constant %5(2 : i64) : !forth.stack -> !forth.stack
    %7 = forth.shared_load_i8 %6 : !forth.stack -> !forth.stack
    // shared i8 store
    %8 = forth.constant %7(9 : i64) : !forth.stack -> !forth.stack
    %9 = forth.constant %8(3 : i64) : !forth.stack -> !forth.stack
    %10 = forth.shared_store_i8 %9 : !forth.stack -> !forth.stack
    // i16
    %11 = forth.constant %10(1 : i64) : !forth.stack -> !forth.stack
    %12 = forth.load_i16 %11 : !forth.stack -> !forth.stack
    %13 = forth.constant %12(42 : i64) : !forth.stack -> !forth.stack
    %14 = forth.constant %13(100 : i64) : !forth.stack -> !forth.stack
    %15 = forth.store_i16 %14 : !forth.stack -> !forth.stack
    // i32
    %16 = forth.constant %15(1 : i64) : !forth.stack -> !forth.stack
    %17 = forth.load_i32 %16 : !forth.stack -> !forth.stack
    %18 = forth.constant %17(42 : i64) : !forth.stack -> !forth.stack
    %19 = forth.constant %18(100 : i64) : !forth.stack -> !forth.stack
    %20 = forth.store_i32 %19 : !forth.stack -> !forth.stack
    // f16
    %21 = forth.constant %20(1 : i64) : !forth.stack -> !forth.stack
    %22 = forth.load_f16 %21 : !forth.stack -> !forth.stack
    %23 = forth.constant %22(42 : i64) : !forth.stack -> !forth.stack
    %24 = forth.constant %23(100 : i64) : !forth.stack -> !forth.stack
    %25 = forth.store_f16 %24 : !forth.stack -> !forth.stack
    // bf16
    %26 = forth.constant %25(1 : i64) : !forth.stack -> !forth.stack
    %27 = forth.load_bf16 %26 : !forth.stack -> !forth.stack
    %28 = forth.constant %27(42 : i64) : !forth.stack -> !forth.stack
    %29 = forth.constant %28(100 : i64) : !forth.stack -> !forth.stack
    %30 = forth.store_bf16 %29 : !forth.stack -> !forth.stack
    // f32
    %31 = forth.constant %30(1 : i64) : !forth.stack -> !forth.stack
    %32 = forth.load_f32 %31 : !forth.stack -> !forth.stack
    %33 = forth.constant %32(42 : i64) : !forth.stack -> !forth.stack
    %34 = forth.constant %33(100 : i64) : !forth.stack -> !forth.stack
    %35 = forth.store_f32 %34 : !forth.stack -> !forth.stack
    return
  }
}
