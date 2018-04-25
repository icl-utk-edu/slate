This will compile the files in the source as well as in the test directory.
Check the GNUMakefile for CXXFLAGS or LIBS that may need to be altered.
The Makefile is initially hardcoded for MKL+OPENMP.

CXXFLAGS += -DSLATE_WITH_MKL
CXXFLAGS += -DSLATE_WITH_OPENMP


Example of running the tester 

salloc -N 4 -w b[01-04] mpirun -n 4 ./test symm   --type s,d,c,z --side l,r --uplo l,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50  --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test syr2k  --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2 

salloc -N 4 -w b[01-04] mpirun -n 4 ./test syrk   --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50  --p 2 --q 2 

salloc -N 4 -w b[01-04] mpirun -n 4 ./test trmm   --type s,d,c,z --side l,r --uplo l,u --transA n,t,c --diag n,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2 

salloc -N 4 -w b[01-04] mpirun -n 4 ./test trsm   --type s,d,c,z --side l,r --uplo l,u --transA n,t,c --diag n,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50  --p 2 --q 2 
  The trsm has some FAILED output because errors are not in 5*eps range.
