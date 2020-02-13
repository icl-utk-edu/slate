#include "test.hh"

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegst_work(Params& params, bool run)
{
}

// -----------------------------------------------------------------------------
void test_hegst(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegst_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegst_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegst_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegst_work<std::complex<double>>(params, run);
            break;
    }
}
