#!/bin/sh

set -x  # set verbose

# square, tall, wide
export size="--dim 2:24:1 --dim 4:48:2x2:24:1 --dim 2:24:1x4:48:2"
export settings='--nb 6 --print-edgeitems 4 --matrix ij --matrixB ij --matrixC ij --ref n'
export settings1='--nb 6 --print-edgeitems 4 --matrix ij --matrixB ij --ref n'
export sep='--------------------------------------------------------------------------------'

echo "settings ${settings}"
echo "size ${size}"

# #gemm prints slate::Matrix A
# echo $sep;  ./tester ${size} ${settings}             gemm
# echo $sep;  ./tester ${size} ${settings} --verbose 0 gemm
# echo $sep;  ./tester ${size} ${settings} --verbose 1 gemm
# echo $sep;  ./tester ${size} ${settings} --verbose 2 gemm
# echo $sep;  ./tester ${size} ${settings} --verbose 3 gemm
# echo $sep;  ./tester ${size} ${settings} --verbose 4 gemm
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 gemm
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 gemm
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 gemm
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 gemm

# #her2k prints slate::HermitianMatrix A
# # echo $sep;  ./tester ${size} ${settings}             her2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 0 her2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 1 her2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 2 her2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 3 her2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 4 her2k
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 her2k
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 her2k
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 her2k
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 her2k

# #trsm prints slate::TriangularMatrix A
# # echo $sep;  ./tester ${size} ${settings1}             trsm
# # echo $sep;  ./tester ${size} ${settings1} --verbose 0 trsm
# # echo $sep;  ./tester ${size} ${settings1} --verbose 1 trsm
# # echo $sep;  ./tester ${size} ${settings1} --verbose 2 trsm
# # echo $sep;  ./tester ${size} ${settings1} --verbose 3 trsm
# # echo $sep;  ./tester ${size} ${settings1} --verbose 4 trsm
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 trsm
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 trsm
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 trsm
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 trsm

# #syr2k prints slate::SymmetricMatrix C, Cref
# # echo $sep;  ./tester ${size} ${settings}             syr2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 0 syr2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 1 syr2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 2 syr2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 3 syr2k
# # echo $sep;  ./tester ${size} ${settings} --verbose 4 syr2k
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 syr2k
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 syr2k
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 syr2k
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 syr2k

# #gels prints slate::TrapezoidMatrix DR
# # echo $sep;  ./tester ${size} ${settings}             gels
# # echo $sep;  ./tester ${size} ${settings} --verbose 0 gels
# # echo $sep;  ./tester ${size} ${settings} --verbose 1 gels
# # echo $sep;  ./tester ${size} ${settings} --verbose 2 --check y gels
# # echo $sep;  ./tester ${size} ${settings} --verbose 3 gels
# # echo $sep;  ./tester ${size} ${settings} --verbose 4 --check y gels

# echo $sep;  ./tester ${size} ${settings} --origin s             gbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 0 gbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 1 gbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 2 --check y gbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 3 gbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 4 --check y gbmm
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 --origin s gbmm
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 --origin s gbmm
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 --origin s gbmm
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 --origin s gbmm

# mpirun -np 4 ./tester --dim 100:105:1 --nb 20 --verbose 2 --print-edgeitems 4 --origin s gbmm

# echo $sep;  ./tester ${size} ${settings} --origin s             hbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 0 hbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 1 hbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 2 --check y hbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 3 hbmm
# echo $sep;  ./tester ${size} ${settings} --origin s --verbose 4 --check y hbmm
./tester --dim 12x6 --verbose 2 --print-edgeitems 4 --origin s hbmm
./tester --dim 6x12 --verbose 2 --print-edgeitems 4 --origin s hbmm
./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 --origin s hbmm
./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 --origin s hbmm

mpirun -np 4 ./tester --dim 100:105:1 --nb 20 --verbose 4 --print-edgeitems 4 --origin s hbmm

# # echo $sep;  ./tester ${size} ${settings} --origin s             tbsm
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 0 tbsm
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 1 tbsm
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 2 --check y tbsm
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 3 tbsm
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 4 --check y tbsm
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 --origin s tbsm
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 --origin s tbsm
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 --origin s tbsm
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 --origin s tbsm

# mpirun -np 4 ./tester --dim 100:105:1 --nb 20 --verbose 2 --print-edgeitems 4 --origin s tbsm

# # echo $sep;  ./tester ${size} ${settings} --origin s             gbtrf
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 0 gbtrf
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 1 gbtrf
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 2 --check y gbtrf
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 3 gbtrf
# # echo $sep;  ./tester ${size} ${settings} --origin s --verbose 4 --check y gbtrf
# ./tester --dim 12x6 --verbose 2 --print-edgeitems 4 --origin s gbtrf
# ./tester --dim 6x12 --verbose 2 --print-edgeitems 4 --origin s gbtrf
# ./tester --dim 14x7 --verbose 2 --nb 6 --print-edgeitems 4 --origin s gbtrf
# ./tester --dim 7x14 --verbose 2 --nb 6 --print-edgeitems 4 --origin s gbtrf

# mpirun -np 4 ./tester --dim 100:105:1 --nb 20 --verbose 2 --print-edgeitems 4 --origin s gbtrf

# ./tester --dim 7x14 --uplo u --nb 6 --print-edgeitems 4 --verbose 2 trnorm

set +x  # unset verbose
