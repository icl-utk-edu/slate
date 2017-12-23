
#include "slate_Debug.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
void Debug::checkTilesLives(Matrix<FloatType> &a)
{
    for (auto it = a.tiles_->begin(); it != a.tiles_->end(); ++it) {

        int64_t i = std::get<0>(it->first);
        int64_t j = std::get<1>(it->first);       

        if (!a.tileIsLocal(i, j))
            if ((*a.lives_)[{i, j}] != 0 || it->second->data_ != nullptr)

                std::cout << "P" << a.mpi_rank_
                          << " TILE " << std::get<0>(it->first)
                          << " " << std::get<1>(it->first)
                          << " LIFE " << (*a.lives_)[{i, j}]
                          << " data_ " << it->second->data_ 
                          << " DEV " << std::get<2>(it->first) << std::endl;
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Debug::printTilesLives(Matrix<FloatType> &a)
{
    if (a.mpi_rank_ == 0) {
        for (int64_t i = 0; i < a.mt_; ++i) {
            for (int64_t j = 0; j < a.nt_; j++) {
                if (a.tiles_->find({i, j, a.host_num_}) == a.tiles_->end())
                    printf("  .");
                else
                    printf("%3ld", (*a.lives_)[{a.it_+i, a.jt_+j}]);
            }
            printf("\n");
        }
    }
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Debug::printTilesMaps(Matrix<FloatType> &a)
{
    for (int64_t i = 0; i < a.mt_; ++i) {
        for (int64_t j = 0; j <= i && j < a.nt_; ++j) {
            auto it = a.tiles_->find({i, j, a.host_num_});
            if (it != a.tiles_->end()) {
                auto tile = it->second;
                if (tile->origin_ == true)
                    printf("o");
                else
                    printf("x");
            }
            else {
                printf(".");
            }
        }
        printf("\n");
    }
    for (int device = 0; device < a.num_devices_; ++device) {
        for (int64_t i = 0; i < a.mt_; ++i) {
            for (int64_t j = 0; j <= i && j < a.nt_; ++j) {
                auto it = a.tiles_->find({i, j, device});
                if (it != a.tiles_->end()) {
                    auto tile = it->second;
                    if (tile->origin_ == true)
                        printf("o");
                    else
                        printf("x");
                }
                else {
                    printf(".");
                }
            }
            printf("\n");
        }
    }
}

//------------------------------------------------------------------------------
void Debug::printNumFreeMemBlocks(Memory &m)
{
    printf("\n");
    for (auto it = m.free_blocks_.begin(); it != m.free_blocks_.end(); ++it)
        printf("\tdevice: %d\tfree blocks: %d\n", it->first, it->second.size());
}

//------------------------------------------------------------------------------
template 
void Debug::checkTilesLives(Matrix<double> &a);

template
void Debug::printTilesLives(Matrix<double> &a);

template
void Debug::printTilesMaps(Matrix<double> &a);

} // namespace slate
