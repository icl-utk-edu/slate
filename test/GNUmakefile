top = ..

testdir_default: ${top}/test/test

# This will get the compiler and flags
include ${top}/GNUmakefile

pwd := ${shell pwd}
cdir := ${notdir ${pwd}}

CXXFLAGS += -I$(top)/libtest -I$(top)/include -I$(top) -I$(top)/blaspp/include -I$(top)/lapackpp/include 
CXXFLAGS += -DSLATE_WITH_MKL 
CXXFLAGS += -DHAVE_MKL -DMKL_Complex8="std::complex<float>" -DMKL_Complex16="std::complex<double>"
CXXFLAGS += -I$(top)/blaspp/test # for blas_flops.hh
CXXFLAGS += -DSLATE_WITH_MPI -I${I_MPI_ROOT}/2017.3.196/intel64/include
CXXFLAGS += -fopenmp 

CFLAGS += -I$(top)/lapackpp/include  -I${I_MPI_ROOT}/2017.3.196/intel64/include -I.

LIBS += -L${top}/lapackpp/lib -Wl,-rpath,${top}/lapackpp/lib -llapackpp 
LIBS += -L${top}/libtest -Wl,-rpath,${top}/libtest -ltest
LIBS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 
#  LIBS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_lapack
#  LIBS += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -ldl
LIBS += -L${I_MPI_ROOT}/2017.3.196/intel64/lib -lmpi 
LIBS += -fopenmp 
LIBS += -lpthread -lm -ldl 

top_src = ${top}/slate_Debug.cc ${top}/slate_Matrix_syrk.cc ${top}/slate_NoCuda.cc ${top}/slate_potrf.cc ${top}/slate_Matrix.cc ${top}/slate_Matrix_trsm.cc ${top}/slate_Matrix_gemm.cc ${top}/slate_Memory.cc ${top}/slate_Matrix_potrf.cc ${top}/slate_NoCublas.cc ${top}/slate_Tile.cc ${top}/slate_trace_Trace.cc
#top_src = ${wildcard ${top}/*.cc} ${top}/trace/trace.c
top_obj = ${addsuffix .o, ${basename ${top_src}}}
top_dep = ${addsuffix .d, ${basename ${top_src}}}

test_src = ${wildcard ${top}/test/*.cc} ${top}/test/myscalapack_common.c
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} -E -o $@ $<

%.o: %.c
	${CC} ${CFLAGS} -c -o $@ $<

${top}/test/test: ${test_obj} ${top_obj} ${libtest_so}
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${top_obj} ${LIBS}

clean:
	rm -f $(test_obj) test trace_*.svg
