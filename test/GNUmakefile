# This will get the compiler and flags
top ?= ..
cdir = ${top}/test
cdir_target: ${cdir}/test

include ${top}/GNUmakefile

pwd := ${shell pwd}

CXXFLAGS += -I${cdir} -I$(top) -I$(top)/libtest -I$(top)/lapackpp/include -I$(top)/blaspp/include
CXXFLAGS += -I$(top)/blaspp/test # for blas_flops.hh

LIB += -L${top}/libtest -Wl,-rpath,${pwd}/${top}/libtest -ltest -L${top}/lapackpp/lib -Wl,-rpath,${pwd}/${top}/lapackpp/lib -llapackpp 
LIB += -L${top}/lib -Wl,-rpath,${pwd}/${top}/lib -lslate 
LIB += -lscalapack 

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@ 

%.o: %.cc
	echo "hello"
	echo ${CXXFLAGS}
	$(CXX) ${CXXFLAGS} -c $< -o $@

test_src = ${cdir}/test.cc ${cdir}/scalapack_support_routines.cc ${cdir}/test_potrf.cc ${cdir}/test_potrf_lapack.cc ${cdir}/test_gemm.cc ${cdir}/test_trmm.cc ${cdir}/test_symm.cc ${cdir}/test_syr2k.cc ${cdir}/test_syrk.cc ${cdir}/test_trsm.cc
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

${cdir}/test: ${test_obj} ${top_obj} 
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${top_obj} $(LIB)

clean:
		rm -f $(test_obj) test trace_*.svg
