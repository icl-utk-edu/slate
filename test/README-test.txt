Examples of running the tester

salloc -N 4 -w b[01-04] mpirun -n 4  ./test gemm  --type s,d,c,z --transA n,t,c --transB n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --dim 100x300x600 --dim 300x100x600 --dim 100x600x300 --dim 300x600x100 --dim 600x100x300 --dim 600x300x100 --nb 10,64 --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test symm   --type s,d,c,z --side l,r --uplo l,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64  --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test syr2k  --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64  --p 2 --q 2
salloc -N 4 -w b[01-04] mpirun -n 4 ./test syr2k  --type c,z --uplo l,u --trans n,t --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64  --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test syrk   --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64   --p 2 --q 2
salloc -N 4 -w b[01-04] mpirun -n 4 ./test syrk   --type c,z --uplo l,u --trans n,t --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64   --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test trmm   --type s,d,c,z --side l,r --uplo l,u --transA n,t,c --diag n,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64 --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test trsm   --type s,d,c,z --side l,r --uplo l,u --transA n,t,c --diag n,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,64  --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test potrf  --type s,d,c,z --dim 100:500:100 --uplo l,u --nb 10,64 --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test hemm   --type s,d,c,z --side l,r --uplo l,u --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --p 1 --q 1 --norm 1 --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test her2k  --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2
salloc -N 4 -w b[01-04] mpirun -n 4 ./test her2k  --type c,z --uplo l,u --trans n,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2

salloc -N 4 -w b[01-04] mpirun -n 4 ./test herk   --type s,d --uplo l,u --trans n,t,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2
salloc -N 4 -w b[01-04] mpirun -n 4 ./test herk   --type c,z --uplo l,u --trans n,c --dim 100:500:100 --dim 200:1000:200x100:500:100 --dim 100:500:100x200:1000:200 --nb 10,50 --p 2 --q 2

----

salloc -N 4 -w b[01-04] env OMP_NUM_THREADS=20  mpirun -n 4 test/test genorm  --type d --dim $[256*4],$[256*4]:$[256*1000]:$[256*10]  --nb 256 --p 2 --q 2 --target t --norm m

----

Single runs

salloc -N 4 -w b[01-04] mpirun -n 4 ./test gemm  --type s --dim 100 --nb 64 --p 2 --q 2
salloc -N 4 -w b[01-04] mpirun -n 4 ./test potrf  --type s --dim 100 --uplo u --nb 64 --p 2 --q 2

salloc -w b[01-04] --tasks-per-node 1 env OMP_NUM_THREADS=20 OMP_NESTED=TRUE OMP_PROC_BIND=TRUE OMP_PLACES=cores OMP_DISPLAY_ENV=TRUE  mpirun -n 4 --print-rank-map test gemm --type d --dim 1000:50000:1000 --nb 256 --p 2 --q 2
