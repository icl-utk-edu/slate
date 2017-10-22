
CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++11

#---------------------------------------------
# if OpenMP
ifeq (openmp,$(filter openmp,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_OPENMP
	CCFLAGS += -fopenmp
endif

#------------------------------------------------------
# if MPI
ifeq (mpi,$(filter mpi,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi
# if Spectrum MPI
else ifeq (spectrum,$(filter spectrum,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi_ibm
endif

#-----------------------------------------------------------------------------
# if MKL 
ifeq (mkl,$(filter mkl,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MKL
	# if Linux
	ifeq (linux,$(filter linux,$(MAKECMDGOALS)))
		LIB += -L${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	# if MacOS
	else ifeq (macos,$(filter macos,$(MAKECMDGOALS)))
		LIB += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	endif
# if ESSL
else ifeq (essl,$(filter essl,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_ESSL
	LIB += -lessl -llapack
endif

#-----------------------------------------
# if CUDA
ifeq (cuda,$(filter cuda,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_CUDA
	LIB += -lcublas -lcudart
endif

#-------------------------------------------------------------------------------
openmp:
	@echo built with OpenMP

mpi:
	@echo built with MPI

spectrum:
	@echo built with Spectrum MPI

mkl:
	@echo built with MKL

essl:
	@echo built with ESSL

cuda:
	@echo built with CUDA

linux macos:
	$(CC) $(CFLAGS) -c -DMPI trace/trace.c -o trace/trace.o
	$(CXX) $(CCFLAGS) app.cc trace/trace.o $(LIB) -o app

clean:
	rm -rf app trace_*.svg
