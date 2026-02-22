\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test HF@ produces forth.loadhf
\ CHECK: forth.loadhf %{{.*}} : !forth.stack -> !forth.stack

\ Test HF! produces forth.storehf
\ CHECK: forth.storehf %{{.*}} : !forth.stack -> !forth.stack

\ Test BF@ produces forth.loadbf
\ CHECK: forth.loadbf %{{.*}} : !forth.stack -> !forth.stack

\ Test BF! produces forth.storebf
\ CHECK: forth.storebf %{{.*}} : !forth.stack -> !forth.stack

\ Test I8@ produces forth.loadi8
\ CHECK: forth.loadi8 %{{.*}} : !forth.stack -> !forth.stack

\ Test I8! produces forth.storei8
\ CHECK: forth.storei8 %{{.*}} : !forth.stack -> !forth.stack

\ Test I16@ produces forth.loadi16
\ CHECK: forth.loadi16 %{{.*}} : !forth.stack -> !forth.stack

\ Test I16! produces forth.storei16
\ CHECK: forth.storei16 %{{.*}} : !forth.stack -> !forth.stack

\ Test SHF@ produces forth.shared_loadhf
\ CHECK: forth.shared_loadhf %{{.*}} : !forth.stack -> !forth.stack

\ Test SHF! produces forth.shared_storehf
\ CHECK: forth.shared_storehf %{{.*}} : !forth.stack -> !forth.stack

\ Test SBF@ produces forth.shared_loadbf
\ CHECK: forth.shared_loadbf %{{.*}} : !forth.stack -> !forth.stack

\ Test SBF! produces forth.shared_storebf
\ CHECK: forth.shared_storebf %{{.*}} : !forth.stack -> !forth.stack

\ Test SI8@ produces forth.shared_loadi8
\ CHECK: forth.shared_loadi8 %{{.*}} : !forth.stack -> !forth.stack

\ Test SI8! produces forth.shared_storei8
\ CHECK: forth.shared_storei8 %{{.*}} : !forth.stack -> !forth.stack

\ Test SI16@ produces forth.shared_loadi16
\ CHECK: forth.shared_loadi16 %{{.*}} : !forth.stack -> !forth.stack

\ Test SI16! produces forth.shared_storei16
\ CHECK: forth.shared_storei16 %{{.*}} : !forth.stack -> !forth.stack
\! kernel main
1 HF@ 2.0 3 HF!
4 BF@ 5.0 6 BF!
7 I8@ 8 9 I8!
10 I16@ 11 12 I16!
13 SHF@ 14.0 15 SHF!
16 SBF@ 17.0 18 SBF!
19 SI8@ 20 21 SI8!
22 SI16@ 23 24 SI16!
