\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: %[[X:.*]] = forth.constant %{{.*}}(1 : i64)
\ CHECK-NEXT: %[[B0:.*]] = forth.constant %[[X]](2 : i64)
\ CHECK-NEXT: %[[S0:.*]] = forth.shuffle down %[[B0]] : !forth.stack -> !forth.stack
\ CHECK-NEXT: %[[B1:.*]] = forth.constant %[[S0]](3 : i64)
\ CHECK-NEXT: %[[S1:.*]] = forth.shuffle up %[[B1]] : !forth.stack -> !forth.stack
\ CHECK-NEXT: %[[B2:.*]] = forth.constant %[[S1]](4 : i64)
\ CHECK-NEXT: %[[S2:.*]] = forth.shuffle xor %[[B2]] : !forth.stack -> !forth.stack
\ CHECK-NEXT: %[[B3:.*]] = forth.constant %[[S2]](5 : i64)
\ CHECK-NEXT: %[[S3:.*]] = forth.shuffle idx %[[B3]] : !forth.stack -> !forth.stack
\ CHECK-NEXT: %[[L:.*]] = forth.lane_id %[[S3]] : !forth.stack -> !forth.stack
\ CHECK-NEXT: forth.constant %[[L]](32 : i64) : !forth.stack -> !forth.stack
\ CHECK-NEXT: return
\! kernel main
1 2 shfl-down 3 Shfl-Up 4 SHFL-xor 5 sHfL-iDx Lane-Id warp-size
