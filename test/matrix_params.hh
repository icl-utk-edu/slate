#ifndef SLATE_MATRIX_PARAMS_HH
#define SLATE_MATRIX_PARAMS_HH

#include "testsweeper.hh"

// =============================================================================
class MatrixParams
{
public:
    MatrixParams();

    void mark();

    int64_t verbose;
    int64_t iseed[4];

    // ---- test matrix generation parameters
    testsweeper::ParamString kind;
    testsweeper::ParamScientific cond, cond_used;
    testsweeper::ParamScientific condD;
};

#endif // SLATE_MATRIX_PARAMS_HH
