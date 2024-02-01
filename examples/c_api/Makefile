# Queries pkg-config for all libraries to link with.

-include make.inc

pkg_exists := ${shell pkg-config --exists slate; echo $$?}
ifneq (${pkg_exists},0)
    ${error pkg-config could not find slate. Install SLATE, then add /prefix/lib/pkgconfig with your prefix to $$PKG_CONFIG_PATH.}
endif

CC             = mpicc
CFLAGS        += ${shell pkg-config --cflags             slate}
LDFLAGS       += ${shell pkg-config --libs-only-L        slate}
LDFLAGS       += ${shell pkg-config --libs-only-other    slate}
slate_libs     = ${shell pkg-config --libs-only-l        slate}

CFLAGS := ${filter-out -std=c++%, ${CFLAGS}}

#-------------------------------------------------------------------------------
# SLATE examples
# Link with -lslate -llapackpp -lblaspp.
# Implicitly links with your BLAS library.

slate_src = ${wildcard *.c}
slate_exe = ${basename ${slate_src}}
exe += ${slate_exe}

slate: ${slate_exe}

${slate_exe}: %: %.o
	${CC} -o $@ $^ \
		${LDFLAGS} ${slate_libs} ${LIBS}

#-------------------------------------------------------------------------------
# Generic rules.

.SUFFIXES:
.DEFAULT_GOAL := all

all: ${exe}

%.o: %.c
	${CC} ${CFLAGS} -c -o $@ $<

clean:
	${RM} ${exe} *.o *.d

clean_exe:
	${RM} ${exe}

distclean: clean

-include *.d

#-------------------------------------------------------------------------------
# Debugging

echo:
	@echo "CC             = ${CC}"
	@echo "CFLAGS         = ${CFLAGS}"
	@echo "LDFLAGS        = ${LDFLAGS}"
	@echo "LIBS           = ${LIBS}"
	@echo "slate_libs     = ${slate_libs}"
	@echo
	@echo "slate_src      = ${slate_src}"
	@echo "slate_exe      = ${slate_exe}"
	@echo
	@echo "exe            = ${exe}"

