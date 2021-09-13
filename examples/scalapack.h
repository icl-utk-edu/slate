#ifndef SCALAPACK_H
#define SCALAPACK_H

//==============================================================================
// ScaLAPACK prototypes
#ifdef __cplusplus
extern "C" {
#endif

// Set according to Fortran compiler's name mangling convention.
// Commonly, append underscore.
#define FORTRAN_NAME( lower, upper ) lower##_

//------------------------------------------------------------------------------
void Cblacs_pinfo(int* mypnum, int* nprocs);
void Cblacs_get(int context, int request, int* value);
int  Cblacs_gridinit(int* context, const char* order, int np_row, int np_col);
void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
void Cblacs_gridexit(int context);
void Cblacs_exit(int error_code);
void Cblacs_abort(int context, int error_code);

//------------------------------------------------------------------------------
#define descinit FORTRAN_NAME(descinit,DESCINIT)
void descinit(int* desc, int* m, int* n, int* mb, int* nb,
              int* irsrc, int* icsrc, int* ictxt, int* lld, int* info);

#define numroc FORTRAN_NAME(numroc,NUMROC)
int numroc(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

//------------------------------------------------------------------------------
#define psgemm FORTRAN_NAME( psgemm, PSGEMM )
#define pdgemm FORTRAN_NAME( pdgemm, PDGEMM )
#define pcgemm FORTRAN_NAME( pcgemm, PCGEMM )
#define pzgemm FORTRAN_NAME( pzgemm, PZGEMM )

void psgemm(const char* transa, const char* transb, int* m, int* n, int* k, float*                alpha, float*                A, int* ia, int* ja, int* descA, float*                B, int* ib, int* jb, int* descB, float*                beta, float*                C, int* ic, int* jc, int* descC);
void pdgemm(const char* transa, const char* transb, int* m, int* n, int* k, double*               alpha, double*               A, int* ia, int* ja, int* descA, double*               B, int* ib, int* jb, int* descB, double*               beta, double*               C, int* ic, int* jc, int* descC);
void pcgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>*  alpha, std::complex<float>*  A, int* ia, int* ja, int* descA, std::complex<float>*  B, int* ib, int* jb, int* descB, std::complex<float>*  beta, std::complex<float>*  C, int* ic, int* jc, int* descC);
void pzgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* A, int* ia, int* ja, int* descA, std::complex<double>* B, int* ib, int* jb, int* descB, std::complex<double>* beta, std::complex<double>* C, int* ic, int* jc, int* descC);

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef SCALAPACK_H
