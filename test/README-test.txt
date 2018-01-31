This will compile the files in the source as well as in the test directory.
Check the GNUMakefile for CXXFLAGS or LIBS that may need to be altered.
The Makefile is initially hardcoded for MKL+OPENMP.

CXXFLAGS += -DSLATE_WITH_MKL
CXXFLAGS += -DSLATE_WITH_OPENMP

make; 
./test potrf  --type d --align 32 --nt 5,10,20 --nb 10,100,268 --uplo l --p 1 --q 1
