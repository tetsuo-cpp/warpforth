: compute-2d-index
    bid-y bdim-y * tid-y +
    bid-x bdim-x * tid-x +
;

: compute-3d-index
    bid-z bdim-z * tid-z +
    bdim-y bdim-x * *
    bid-y bdim-y * tid-y +
    bdim-x * +
    bid-x bdim-x * tid-x + +
;

: store-dimensions
    tid-x 0 !
    tid-y 1 !
    tid-z 2 !
    bid-x 3 !
    bid-y 4 !
    bid-z 5 !
;

: store-block-dims
    bdim-x 6 !
    bdim-y 7 !
    bdim-z 8 !
;

: store-grid-dims
    gdim-x 9 !
    gdim-y 10 !
    gdim-z 11 !
;

store-dimensions
store-block-dims
store-grid-dims
compute-2d-index 12 !
global-id 13 !
