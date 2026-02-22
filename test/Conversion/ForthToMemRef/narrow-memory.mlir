// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// HF@ (loadhf): load addr, inttoptr, llvm.load f16, extf f16->f32, bitcast f32->i32, extsi i32->i64, store
// CHECK: memref.load
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> f16
// CHECK: arith.extf %{{.*}} : f16 to f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// HF! (storehf): pop addr, pop value, trunci i64->i32, inttoptr, bitcast i32->f32, truncf f32->f16, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.truncf %{{.*}} : f32 to f16
// CHECK: llvm.store %{{.*}}, %{{.*}} : f16

// BF@ (loadbf): load addr, inttoptr, llvm.load bf16, extf bf16->f32, bitcast f32->i32, extsi i32->i64
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> bf16
// CHECK: arith.extf %{{.*}} : bf16 to f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// BF! (storebf): trunci i64->i32, bitcast i32->f32, truncf f32->bf16, store
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK: llvm.store %{{.*}}, %{{.*}} : bf16

// I8@ (loadi8): load addr, inttoptr, llvm.load i8, extsi i8->i32, extsi i32->i64
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i8
// CHECK: arith.extsi %{{.*}} : i8 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// I8! (storei8): trunci i64->i32, trunci i32->i8, store
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr
// CHECK: arith.trunci %{{.*}} : i32 to i8
// CHECK: llvm.store %{{.*}}, %{{.*}} : i8

// I16@ (loadi16): llvm.load i16, extsi i16->i32, extsi i32->i64
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i16
// CHECK: arith.extsi %{{.*}} : i16 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// I16! (storei16): trunci i64->i32, trunci i32->i16, store
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr
// CHECK: arith.trunci %{{.*}} : i32 to i16
// CHECK: llvm.store %{{.*}}, %{{.*}} : i16

// SHF@ (shared_loadhf): inttoptr to shared ptr, llvm.load f16
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> f16
// CHECK: arith.extf %{{.*}} : f16 to f32

// SHF! (shared_storehf): truncf f32->f16, store to shared
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.truncf %{{.*}} : f32 to f16
// CHECK: llvm.store %{{.*}}, %{{.*}} : f16, !llvm.ptr<3>

// SBF@ (shared_loadbf): inttoptr shared, llvm.load bf16
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> bf16
// CHECK: arith.extf %{{.*}} : bf16 to f32

// SBF! (shared_storebf): truncf f32->bf16, store to shared
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK: llvm.store %{{.*}}, %{{.*}} : bf16, !llvm.ptr<3>

// SI8@ (shared_loadi8): llvm.load i8 from shared, extsi i8->i32
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i8
// CHECK: arith.extsi %{{.*}} : i8 to i32

// SI8! (shared_storei8): trunci i32->i8, store to shared
// CHECK: arith.trunci %{{.*}} : i32 to i8
// CHECK: llvm.store %{{.*}}, %{{.*}} : i8, !llvm.ptr<3>

// SI16@ (shared_loadi16): llvm.load i16 from shared, extsi
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i16
// CHECK: arith.extsi %{{.*}} : i16 to i32

// SI16! (shared_storei16): trunci i32->i16, store to shared
// CHECK: arith.trunci %{{.*}} : i32 to i16
// CHECK: llvm.store %{{.*}}, %{{.*}} : i16, !llvm.ptr<3>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    // HF@ test: push address, load f16
    %1 = forth.constant %0(1000 : i32) : !forth.stack -> !forth.stack
    %2 = forth.loadhf %1 : !forth.stack -> !forth.stack
    // HF! test: push value, push address, store f16
    %3 = forth.constant %2(3.140000e+00 : f32) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(2000 : i32) : !forth.stack -> !forth.stack
    %5 = forth.storehf %4 : !forth.stack -> !forth.stack
    // BF@ test
    %6 = forth.constant %5(3000 : i32) : !forth.stack -> !forth.stack
    %7 = forth.loadbf %6 : !forth.stack -> !forth.stack
    // BF! test
    %8 = forth.constant %7(2.710000e+00 : f32) : !forth.stack -> !forth.stack
    %9 = forth.constant %8(4000 : i32) : !forth.stack -> !forth.stack
    %10 = forth.storebf %9 : !forth.stack -> !forth.stack
    // I8@ test
    %11 = forth.constant %10(5000 : i32) : !forth.stack -> !forth.stack
    %12 = forth.loadi8 %11 : !forth.stack -> !forth.stack
    // I8! test
    %13 = forth.constant %12(42 : i32) : !forth.stack -> !forth.stack
    %14 = forth.constant %13(6000 : i32) : !forth.stack -> !forth.stack
    %15 = forth.storei8 %14 : !forth.stack -> !forth.stack
    // I16@ test
    %16 = forth.constant %15(7000 : i32) : !forth.stack -> !forth.stack
    %17 = forth.loadi16 %16 : !forth.stack -> !forth.stack
    // I16! test
    %18 = forth.constant %17(999 : i32) : !forth.stack -> !forth.stack
    %19 = forth.constant %18(8000 : i32) : !forth.stack -> !forth.stack
    %20 = forth.storei16 %19 : !forth.stack -> !forth.stack
    // SHF@ test (shared)
    %21 = forth.constant %20(100 : i32) : !forth.stack -> !forth.stack
    %22 = forth.shared_loadhf %21 : !forth.stack -> !forth.stack
    // SHF! test (shared)
    %23 = forth.constant %22(1.500000e+00 : f32) : !forth.stack -> !forth.stack
    %24 = forth.constant %23(200 : i32) : !forth.stack -> !forth.stack
    %25 = forth.shared_storehf %24 : !forth.stack -> !forth.stack
    // SBF@ test (shared)
    %26 = forth.constant %25(300 : i32) : !forth.stack -> !forth.stack
    %27 = forth.shared_loadbf %26 : !forth.stack -> !forth.stack
    // SBF! test (shared)
    %28 = forth.constant %27(2.500000e+00 : f32) : !forth.stack -> !forth.stack
    %29 = forth.constant %28(400 : i32) : !forth.stack -> !forth.stack
    %30 = forth.shared_storebf %29 : !forth.stack -> !forth.stack
    // SI8@ test (shared)
    %31 = forth.constant %30(500 : i32) : !forth.stack -> !forth.stack
    %32 = forth.shared_loadi8 %31 : !forth.stack -> !forth.stack
    // SI8! test (shared)
    %33 = forth.constant %32(7 : i32) : !forth.stack -> !forth.stack
    %34 = forth.constant %33(600 : i32) : !forth.stack -> !forth.stack
    %35 = forth.shared_storei8 %34 : !forth.stack -> !forth.stack
    // SI16@ test (shared)
    %36 = forth.constant %35(700 : i32) : !forth.stack -> !forth.stack
    %37 = forth.shared_loadi16 %36 : !forth.stack -> !forth.stack
    // SI16! test (shared)
    %38 = forth.constant %37(123 : i32) : !forth.stack -> !forth.stack
    %39 = forth.constant %38(800 : i32) : !forth.stack -> !forth.stack
    %40 = forth.shared_storei16 %39 : !forth.stack -> !forth.stack
    return
  }
}
