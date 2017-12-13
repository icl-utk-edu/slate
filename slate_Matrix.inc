
//-----------------------------------------
template <Target target = Target::HostTask>
static void gemm(blas::Op opa, blas::Op opb,
                 FloatType alpha, Matrix &&a,
                                  Matrix &&b,
                 FloatType beta,  Matrix &&c);

static void gemm(internal::TargetType<Target::HostTask>,
                 blas::Op opa, blas::Op opb,
                 FloatType alpha, Matrix &a,
                                  Matrix &b,
                 FloatType beta,  Matrix &c);

//-----------------------------------------
template <Target target = Target::HostTask>
static void potrf(blas::Uplo uplo, Matrix &&a);

static void potrf(internal::TargetType<Target::HostTask>,
                  blas::Uplo uplo, Matrix &a);

//-----------------------------------------
template <Target target = Target::HostTask>
static void syrk(blas::Uplo uplo, blas::Op op,
                 FloatType alpha, Matrix &&a,
                 FloatType beta,  Matrix &&c);

static void syrk(internal::TargetType<Target::HostTask>,
                 blas::Uplo uplo, blas::Op op,
                 FloatType alpha, Matrix &a,
                 FloatType beta,  Matrix &c);

static void syrk(internal::TargetType<Target::HostNest>,
                 blas::Uplo uplo, blas::Op op,
                 FloatType alpha, Matrix &a,
                 FloatType beta,  Matrix &c);

static void syrk(internal::TargetType<Target::HostBatch>,
                 blas::Uplo uplo, blas::Op op,
                 FloatType alpha, Matrix &a,
                 FloatType beta,  Matrix &c);

static void syrk(internal::TargetType<Target::Devices>,
                 blas::Uplo uplo, blas::Op op,
                 FloatType alpha, Matrix &a,
                 FloatType beta,  Matrix &c);

//-----------------------------------------
template <Target target = Target::HostTask>
static void trsm(blas::Side side, blas::Uplo uplo,
                 blas::Op op, blas::Diag diag,
                 FloatType alpha, Matrix &&a,
                                  Matrix &&b);

static void trsm(internal::TargetType<Target::HostTask>,
                 blas::Side side, blas::Uplo uplo,
                 blas::Op op, blas::Diag diag,
                 FloatType alpha, Matrix &a,
                                  Matrix &b);
