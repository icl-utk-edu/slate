# This will get the compiler and flags
top ?= ..
testdir_target: ${top}/test/test
include ${top}/GNUmakefile

pwd := ${shell pwd}

CXXFLAGS += -I$(top)/libtest
LIB += -L${top}/libtest -Wl,-rpath,${top}/libtest -ltest 

CXXFLAGS += -I$(top)/blaspp/test # for blas_flops.hh

# ScaLAPACK
ifeq (${mkl},1)
LIB := -L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 ${LIB}
endif
ifeq (${essl},1)
LIB := -lscalapack ${LIB}
endif


test_src = ${top}/test/test.cc ${top}/test/test_gemm.cc ${top}/test/test_trmm.cc ${top}/test/test_symm.cc ${top}/test/test_syrk.cc ${top}/test/test_syr2k.cc ${top}/test/test_trsm.cc ${top}/test/test_potrf.cc 
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@ 

${top}/test/test: ${test_obj} ${top_obj} 
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${top_obj} -L${top}/lib -Wl,-rpath,${pwd}/${top}/lib -lslate $(LIB) 

clean_test:
	rm -f $(test_obj) test trace_*.svg

clean: clean_test
