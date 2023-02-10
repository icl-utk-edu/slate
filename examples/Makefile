# Queries pkg-config for all libraries to link with.
# User should not generally need to put anything in make.inc.
# LIBS can be used if something like -lgfortran is missing.

-include make.inc

pkg_exists := ${shell pkg-config --exists slate; echo $$?}
ifneq (${pkg_exists},0)
    ${error pkg-config could not find slate. Install SLATE, then add /prefix/lib/pkgconfig with your prefix to $$PKG_CONFIG_PATH.}
endif

CXX            = ${shell pkg-config --variable=CXX       slate}
CXXFLAGS      += ${shell pkg-config --cflags             slate}
LDFLAGS       += ${shell pkg-config --libs-only-L        slate}
LDFLAGS       += ${shell pkg-config --libs-only-other    slate}
slate_libs     = ${shell pkg-config --libs-only-l        slate}
scalapack_libs = ${shell pkg-config --variable=scalapack slate}

#-------------------------------------------------------------------------------
# SLATE's ScaLAPACK API example
# Link with -lslate_scalapack_api -lscalapack -lslate -lblaspp -llapackpp.
# If RUNPATH was set on libslate_scalapack_api.so when it was installed,
# here we could implicitly link with -lslate -lblaspp -llapackpp.
# Implicitly links with your BLAS library.

scalapack_src = ${wildcard *scalapack*.cc}
scalapack_exe = ${basename ${scalapack_src}}
exe += ${scalapack_exe}

scalapack: ${scalapack_exe}

${scalapack_exe}: %: %.o
	${CXX} -o $@ $^ \
		${LDFLAGS} -lslate_scalapack_api ${slate_libs} ${scalapack_libs} ${LIBS}

#-------------------------------------------------------------------------------
# SLATE examples
# Link with -lslate -llapackpp -lblaspp.
# Implicitly links with your BLAS library.

slate_src = ${filter-out ${scalapack_src}, ${wildcard *.cc}}
slate_exe = ${basename ${slate_src}}
exe += ${slate_exe}

slate: ${slate_exe}

${slate_exe}: %: %.o
	${CXX} -o $@ $^ \
		${LDFLAGS} ${slate_libs} ${LIBS}

check: ${exe}
	./run_tests.py

#-------------------------------------------------------------------------------
# Generic rules.

.SUFFIXES:
.DEFAULT_GOAL := all

all: ${exe}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c -o $@ $<

clean:
	${RM} ${exe} *.o *.d

clean_exe:
	${RM} ${exe}

distclean: clean

-include *.d

#-------------------------------------------------------------------------------
# Debugging

echo:
	@echo "CXX            = ${CXX}"
	@echo "CXXFLAGS       = ${CXXFLAGS}"
	@echo "LDFLAGS        = ${LDFLAGS}"
	@echo "LIBS           = ${LIBS}"
	@echo "slate_libs     = ${slate_libs}"
	@echo "scalapack_libs = ${scalapack_libs}"
	@echo
	@echo "slate_src      = ${slate_src}"
	@echo "slate_exe      = ${slate_exe}"
	@echo
	@echo "scalapack_src  = ${scalapack_src}"
	@echo "scalapack_exe  = ${scalapack_exe}"
	@echo
	@echo "exe            = ${exe}"
