This will compile the files in the source as well as in the test directory.
Check the GNUMakefile for CXXFLAGS or LIBS that may need to be altered.
The Makefile is initially hardcoded for MKL+OPENMP.

CXXFLAGS += -DSLATE_WITH_MKL
CXXFLAGS += -DSLATE_WITH_OPENMP

Running the SLATE only tester
make; 
./test potrf  --type d --align 32 --nt 5,10,20 --nb 10,100,268 --uplo l --p 1 --q 1
mpirun -n 4 ./test potrf  --type d --align 32 --nb 200 --nt 6 --lookahead 1 --p 2 --q 2
salloc -N 4 -w b[01-04] srun ./test potrf  --type d --align 32 --nb 200 --nt 6 --lookahead 1 --p 2 --q 2

Running the ScaLAPACK pdpotrf tester
make
./test pdpotrf --nb 128,256 --dim 2000 --dim 1200 --p 1 --q 1
mpirun -np 4 ./test pdpotrf --nb 128,256 --dim 10000 --dim 12000 --p 2 --q 2
salloc -N 4 -w b[01-04] srun -n 4 ./test pdpotrf --nb 128,256 --dim 10000 --dim 12000 --p 2 --q 2
