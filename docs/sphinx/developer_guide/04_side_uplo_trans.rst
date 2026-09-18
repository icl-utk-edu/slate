Handling of Side, Uplo, Trans, etc.
===================================

The classical BLAS take parameters such as side, uplo, trans (named "op" in SLATE), and diag to specify operation variants. Traditionally, this has meant that implementations have numerous cases. The reference BLAS has nine cases in ``zgemm`` and eight cases in ``ztrmm`` (times several sub-cases). ScaLAPACK and PLASMA likewise have eight cases in ``ztrmm``. In contrast, by storing both uplo and op within the matrix object itself, and supporting inexpensive shallow copy transposition, SLATE can implement just one or two cases and map all the other cases to that implementation by appropriate transpositions.

For instance, at the high level, ``gemm`` can ignore the operations on :math:`A` and :math:`B`. If transposed, the matrix object itself handles swapping indices to obtain the correct tiles during the algorithm. At the low level, the transposition operation is set on the tiles, and is passed on to the underlying node-level BLAS ``gemm`` routine.

Similarly, the Cholesky factorization implements only the lower case; the upper case is handled by a shallow copy transposition to map it to the lower case. The data is not physically transposed in memory; only the transpose op flag is set so that the matrix is logically lower.

Note that for the shallow copy to work correctly, matrices must be passed by value, rather than by reference. For instance, if ``potrf`` used pass-by-reference (see erroneous code example below), a user calling ``potrf`` would have the unintended side effect of transposing the matrix :math:`A` in the user's code:

.. code-block:: cpp

   // Erroneous code: passing A by reference and transposing it,
   // unintentionally transposing it in caller's code.
   template <Target target, typename scalar_t>
   int64_t potrf(
       slate::internal::TargetType<target>,
       HermitianMatrix<scalar_t>& A, ...)
   {
       // If upper, change to lower.
       // Since A is passed by reference (HermitianMatrix<scalar_t>& A),
       // this inadvertently transposes the matrix in the user's code -- a bug!
       if (A.uplo() == Uplo::Upper) {
           A = conj_transpose( A );
       }

       // Continue with code that assumes A is logically lower ...
   }

Instead, the matrix ``A`` is passed by value into ``potrf`` (see correct code example below), so transposition within the computational routine doesn't affect transposition in the user's code. (Though some wrappers may pass it by reference.) This results in the correct output with no unintended side effects.

.. code-block:: cpp

   // Correct code: passing A by value and transposing it,
   // without transposing it in caller's code.
   template <Target target, typename scalar_t>
   int64_t potrf(
       slate::internal::TargetType<target>,
       HermitianMatrix<scalar_t> A, ...)
   {
       // If upper, change to lower.
       // Since A is passed by value (HermitianMatrix<scalar_t> A),
       // with shallow-copy semantics,
       // this doesn't transpose the matrix A in the user's code.
       if (A.uplo() == Uplo::Upper) {
           A = conj_transpose( A );
       }

       // Continue with code that assumes A is logically lower ...
   }
