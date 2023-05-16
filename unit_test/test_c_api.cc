#include <slate/slate.hh>
#include <slate/c_api/slate.h>

#include "unit_test.hh"

namespace test {

//------------------------------------------------------------------------------
/// Test that C enums match C++ enums.
///
void test_enums()
{
    //----------
    assert( slate_TileKind_Workspace  == int( slate::TileKind::Workspace  ) );
    assert( slate_TileKind_SlateOwned == int( slate::TileKind::SlateOwned ) );
    assert( slate_TileKind_UserOwned  == int( slate::TileKind::UserOwned  ) );

    //----------
    assert( slate_Target_Host      == int( slate::Target::Host      ) );
    assert( slate_Target_HostTask  == int( slate::Target::HostTask  ) );
    assert( slate_Target_HostNest  == int( slate::Target::HostNest  ) );
    assert( slate_Target_HostBatch == int( slate::Target::HostBatch ) );
    assert( slate_Target_Devices   == int( slate::Target::Devices   ) );

    //----------
    assert( slate_TileReleaseStrategy_None     == int( slate::TileReleaseStrategy::None     ) );
    assert( slate_TileReleaseStrategy_Internal == int( slate::TileReleaseStrategy::Internal ) );
    assert( slate_TileReleaseStrategy_Slate    == int( slate::TileReleaseStrategy::Slate    ) );
    assert( slate_TileReleaseStrategy_All      == int( slate::TileReleaseStrategy::All      ) );

    //----------
    assert( slate_MethodEig_QR == int( slate::MethodEig::QR ) );
    assert( slate_MethodEig_DC == int( slate::MethodEig::DC ) );

    //----------
    assert( slate_Option_ChunkSize           == int( slate::Option::ChunkSize           ) );
    assert( slate_Option_Lookahead           == int( slate::Option::Lookahead           ) );
    assert( slate_Option_BlockSize           == int( slate::Option::BlockSize           ) );
    assert( slate_Option_InnerBlocking       == int( slate::Option::InnerBlocking       ) );
    assert( slate_Option_MaxPanelThreads     == int( slate::Option::MaxPanelThreads     ) );
    assert( slate_Option_Tolerance           == int( slate::Option::Tolerance           ) );
    assert( slate_Option_Target              == int( slate::Option::Target              ) );
    assert( slate_Option_TileReleaseStrategy == int( slate::Option::TileReleaseStrategy ) );
    assert( slate_Option_HoldLocalWorkspace  == int( slate::Option::HoldLocalWorkspace  ) );

    assert( slate_Option_PrintVerbose        == int( slate::Option::PrintVerbose        ) );
    assert( slate_Option_PrintEdgeItems      == int( slate::Option::PrintEdgeItems      ) );
    assert( slate_Option_PrintWidth          == int( slate::Option::PrintWidth          ) );
    assert( slate_Option_PrintPrecision      == int( slate::Option::PrintPrecision      ) );
    assert( slate_Option_PivotThreshold      == int( slate::Option::PivotThreshold      ) );

    assert( slate_Option_MethodCholQR        == int( slate::Option::MethodCholQR        ) );
    assert( slate_Option_MethodEig           == int( slate::Option::MethodEig           ) );
    assert( slate_Option_MethodGels          == int( slate::Option::MethodGels          ) );
    assert( slate_Option_MethodGemm          == int( slate::Option::MethodGemm          ) );
    assert( slate_Option_MethodHemm          == int( slate::Option::MethodHemm          ) );
    assert( slate_Option_MethodLU            == int( slate::Option::MethodLU            ) );
    assert( slate_Option_MethodTrsm          == int( slate::Option::MethodTrsm          ) );

    //----------
    assert( slate_Op_NoTrans   == int( slate::Op::NoTrans   ) );
    assert( slate_Op_Trans     == int( slate::Op::Trans     ) );
    assert( slate_Op_ConjTrans == int( slate::Op::ConjTrans ) );

    //----------
    assert( slate_Uplo_Upper   == int( slate::Uplo::Upper   ) );
    assert( slate_Uplo_Lower   == int( slate::Uplo::Lower   ) );
    assert( slate_Uplo_General == int( slate::Uplo::General ) );

    //----------
    assert( slate_Diag_NonUnit == int( slate::Diag::NonUnit ) );
    assert( slate_Diag_Unit    == int( slate::Diag::Unit    ) );

    //----------
    assert( slate_Side_Left  == int( slate::Side::Left  ) );
    assert( slate_Side_Right == int( slate::Side::Right ) );

    //----------
    assert( slate_Layout_ColMajor == int( slate::Layout::ColMajor ) );
    assert( slate_Layout_RowMajor == int( slate::Layout::RowMajor ) );

    //----------
    assert( slate_Norm_One == int( slate::Norm::One ) );
    assert( slate_Norm_Two == int( slate::Norm::Two ) );
    assert( slate_Norm_Inf == int( slate::Norm::Inf ) );
    assert( slate_Norm_Fro == int( slate::Norm::Fro ) );
    assert( slate_Norm_Max == int( slate::Norm::Max ) );

    //----------
    assert( slate_Direction_Forward  == int( slate::Direction::Forward  ) );
    assert( slate_Direction_Backward == int( slate::Direction::Backward ) );

    //----------
    assert( slate_Job_NoVec        == int( slate::Job::NoVec        ) );
    assert( slate_Job_Vec          == int( slate::Job::Vec          ) );
    assert( slate_Job_UpdateVec    == int( slate::Job::UpdateVec    ) );
    assert( slate_Job_AllVec       == int( slate::Job::AllVec       ) );
    assert( slate_Job_SomeVec      == int( slate::Job::SomeVec      ) );
    assert( slate_Job_OverwriteVec == int( slate::Job::OverwriteVec ) );
    assert( slate_Job_CompactVec   == int( slate::Job::CompactVec   ) );
    assert( slate_Job_SomeVecTol   == int( slate::Job::SomeVecTol   ) );
    assert( slate_Job_VecJacobi    == int( slate::Job::VecJacobi    ) );
    assert( slate_Job_Workspace    == int( slate::Job::Workspace    ) );
}

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test( test_enums, "enums" );
}

}  // namespace test

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int err = unit_test_main();  // which calls run_tests()
    return err;
}
