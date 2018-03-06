# This will get the compiler and flags
top ?= ..
testdir_target: ${top}/test/test

include ../GNUmakefile

pwd := ${shell pwd}
cdir := ${notdir ${pwd}}

CXXFLAGS += -I$(top)/test -I$(top) -I$(top)/libtest -I$(top)/lapackpp/include -I$(top)/blaspp/include 
CXXFLAGS += -I$(top)/blaspp/test # for blas_flops.hh

LIBS += -L${top}/libtest -Wl,-rpath,${top}/libtest -ltest -L${top}/lapackpp/lib -Wl,-rpath,${top}/lapackpp/lib -llapackpp 

test_src = ${top}/test/test.cc ${top}/test/scalapack_support_routines.cc ${top}/test/test_potrf.cc ${top}/test/test_potrf_lapack.cc ${top}/test/test_gemm.cc 
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@ 

${top}/test/test: ${test_obj} ${top_obj} 
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${top_obj} -L${top}/lib -lslate $(LIB) 

clean:
		rm -f $(test_obj) test trace_*.svg
