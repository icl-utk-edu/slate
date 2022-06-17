#!/bin/bash -e

if [[ "x$maker" = "x" || "x$gpu" = "x" ]]; then
   echo "$0 <make|cmake> <amd|nvidia|nogpu>"
   exit 1
fi

shopt -s expand_aliases
set +x
source /etc/profile
export top=`pwd`

# Suppress echo (-x) output of commands executed with `quiet`.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

print_section() {
    builtin echo "$*"
    date
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias section='{ save_flags="$-"; set +x; } 2> /dev/null; print_section'

module load gcc@7.3.0
module load intel-mkl

section "======================================== maker=$maker"

# For simplicity, create make.inc regardless of ${maker}
export color=no

cat > make.inc << END
CXX    = mpicxx
FC     = gfortran
blas   = mkl
prefix = ${top}/install
END

mydir=`basedir $0`
source ${mydir}/setup_${gpu}.sh

section "======================================== Verify MPI"

which mpicxx
which mpif90
mpicxx --version
mpif90 --version

echo "MKLROOT=${MKLROOT}"

section "======================================== Environment"
env

if [ "${maker}" = "cmake" ]; then
    cd build
fi

