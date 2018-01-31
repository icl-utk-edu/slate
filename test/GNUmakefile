.SUFFIXES:

top = ..

testdir_default: ${top}/test/test

# This will get the compiler and flags
include ${top}/GNUmakefile

pwd := ${shell pwd}
cdir := ${notdir ${pwd}}

CXXFLAGS += -I$(top)/libtest -I$(top)/include -I$(top) -I$(top)/blaspp/include -I$(top)/lapackpp/include -DHAVE_MKL -fopenmp 
CXXFLAGS += -I$(top)/blaspp/test # for blas_flops.hh
CXXFLAGS += -DSLATE_WITH_MKL -DSLATE_WITH_MPI

LIBS += -L${top}/lapackpp/lib -Wl,-rpath,${top}/lapackpp/lib -llapackpp 
LIBS += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -ldl
#LIBS += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt 
LIBS += -L${top}/libtest -Wl,-rpath,${top}/libtest -ltest
LIBS += -fopenmp 
LIBS += -lpthread -lm -ldl 

top_src = ${top}/slate_Debug.cc ${top}/slate_Matrix_syrk.cc ${top}/slate_NoCuda.cc ${top}/slate_potrf.cc ${top}/slate_Matrix.cc ${top}/slate_Matrix_trsm.cc ${top}/slate_Matrix_gemm.cc ${top}/slate_Memory.cc ${top}/slate_Matrix_potrf.cc ${top}/slate_NoCublas.cc ${top}/slate_Tile.cc ${top}/trace/trace.c
#top_src = ${wildcard ${top}/*.cc} ${top}/trace/trace.c
top_obj = ${addsuffix .o, ${basename ${top_src}}}
top_dep = ${addsuffix .d, ${basename ${top_src}}}

test_src = ${wildcard ${top}/test/*.cc}
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} -E -o $@ $<

${top}/test/test: ${test_obj} ${top_obj} ${libtest_so}
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${top_obj} ${LIBS}

clean:
	rm -f $(test_obj) test trace_*.svg
