/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef _SCALAPACK_COMMON_H_
#define _SCALAPACK_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void scalapack_pdplrnt( double *A, int m, int n, int mb, int nb, int myrow, int mycol, int nprow, int npcol, int mloc, int seed );

void scalapack_pdplghe( double *A, int m, int n, int mb, int nb, int myrow, int mycol, int nprow, int npcol, int mloc, int seed );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _SCALAPACK_COMMON_H_ */
