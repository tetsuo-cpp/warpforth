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

: get-dimensions
    tid-x tid-y tid-z bid-x bid-y bid-z
;

: get-block-dims
    bdim-x bdim-y bdim-z
;

: get-grid-dims
    gdim-x gdim-y gdim-z
;

: get-global-id
    global-id
;

get-dimensions
5 ! 4 ! 3 ! 2 ! 1 ! 0 !

get-block-dims
8 ! 7 ! 6 !

get-grid-dims
11 ! 10 ! 9 !

compute-2d-index
12 !

get-global-id
13 !
