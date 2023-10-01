#!/usr/bin/perl -pi

s/hip(Float|Double)Complex/rocblas_\l$1_complex/g;
s/make_(rocblas_(float|double)_complex)/$1/g;
s/\.cuh/.hip.hh/g;

# Fix spaces that violate style guide and are caught by git commit hook.
s/ +(,|;|$)/$1/g;
