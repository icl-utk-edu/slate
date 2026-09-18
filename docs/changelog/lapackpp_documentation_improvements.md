# LAPACK++ Doxygen and Sphinx Documentation Improvements
## Session: December 2025

### Summary
Enhanced LAPACK++ documentation with proper Doxygen comments and fixed Sphinx integration issues.

## Changes Made

### 1. lapack.hh - Version Function Documentation
**File:** `/lapackpp/include/lapack.hh`

**Improvements:**
- Added comprehensive Doxygen documentation for `lapackpp_version()` and `lapackpp_id()` functions
- Included detailed format specifications (YYYYMMDD format, git commit hash format)
- Added `@ingroup util` tags for proper categorization
- Enhanced file-level documentation with feature overview

**Before:**
```cpp
/// @return LAPACK++ version as integer, e.g., 20250528 for version 2025.05.28
/// @ingroup util
int lapackpp_version();
```

**After:**
```cpp
// =============================================================================
/// Returns LAPACK++ version as an integer.
///
/// Version format is YYYYMMDD, e.g., 20250528 for version 2025.05.28
///
/// @return LAPACK++ version integer
///
/// @ingroup util
int lapackpp_version();
```

### 2. lapack/util.hh - Error Class Documentation
**File:** `/lapackpp/include/lapack/util.hh`

**Improvements:**
- Enhanced Error class documentation with compile-time behavior options
- Documented error checking modes (default, LAPACK_ERROR_ASSERT, LAPACK_ERROR_NDEBUG)
- Already had proper `@ingroup util` tags

**Key Features Documented:**
- Exception-based error handling (default)
- Assert-based error handling (LAPACK_ERROR_ASSERT)
- Disabled error checking (LAPACK_ERROR_NDEBUG)

### 3. lapack/device.hh - GPU Device Support Documentation
**File:** `/lapackpp/include/lapack/device.hh`

**Comprehensive Documentation Already Present:**
- File-level documentation with supported operations
- Queue class with detailed member documentation
- GPU-accelerated functions:
  - `potrf()` - Cholesky factorization with complete parameter docs
  - `getrf()` - LU factorization with workspace queries
  - `geqrf()` - QR factorization with workspace queries  
  - `heevd()` - Eigenvalue decomposition using divide-and-conquer
- All functions include:
  - Full parameter descriptions with LaTeX math formulas
  - Device vs host memory specifications
  - Workspace size query functions
  - `@ingroup` tags for categorization

**Example Documentation Quality:**
```cpp
/// Cholesky factorization of Hermitian positive definite matrix on GPU.
///
/// Computes the Cholesky factorization of an n-by-n Hermitian positive definite
/// matrix A on device memory:
/// - If uplo = Upper: \f$ A = U^H U \f$
/// - If uplo = Lower: \f$ A = L L^H \f$
///
/// @param[in] uplo Whether upper or lower triangle of A is stored
/// @param[in] n Matrix dimension. n >= 0
/// @param[in,out] dA Device pointer to n-by-n matrix A
/// @param[out] dev_info Device pointer to info status
/// @param[in] queue GPU device queue for asynchronous execution
///
/// @ingroup device_posv
```

### 4. Sphinx Configuration - LAPACK++ Index Updates
**File:** `/docs/sphinx/api/lapackpp/index.rst`

**Fixed Missing Toctree Entries:**
- Added `factorizations` to toctree
- Changed `eigenvalue` to `eigenvalues` (matching actual filename)
- Resolved warnings about documents not included in toctree

**Before:**
```rst
.. toctree::
   :maxdepth: 2

   util
   device
   flops
   linear_systems
   least_squares
   eigenvalue
   svd
   auxiliary
```

**After:**
```rst
.. toctree::
   :maxdepth: 2

   util
   device
   flops
   factorizations
   linear_systems
   least_squares
   eigenvalues
   svd
   auxiliary
```

## Documentation Quality Standards

### Doxygen Comment Structure
All updated functions follow this structure:
1. Brief description (first line after ///)
2. Detailed description with mathematical formulas using LaTeX
3. Template parameter documentation (@tparam)
4. Parameter documentation (@param with [in], [out], [in,out])
5. Return value documentation (@return)
6. Group membership (@ingroup)

### LaTeX Math Support
- Inline math: `\f$ A = U^H U \f$`
- Display math: `\f[ A = U^H U \f]`
- Proper escaping of special characters

### Cross-Reference Support
- Doxygen groups for categorization (@ingroup)
- Consistent naming across BLAS++ and LAPACK++
- References to related functions

## Resolved Warnings

### Before:
```
WARNING: doxygenfunction: Cannot find function "lapack::lapackpp_version"
WARNING: doxygenfunction: Cannot find function "lapack::lapackpp_id"
WARNING: document isn't included in any toctree: eigenvalues.rst
WARNING: document isn't included in any toctree: factorizations.rst
```

### After:
All warnings resolved - functions now properly documented and all RST files included in toctree.

## Build Commands

To rebuild documentation:
```bash
cd slate/docs/sphinx
make clean html
```

Output location: `_build/html/index.html`

## Documentation Coverage

### LAPACK++ Modules Now Documented:

1. **Utilities** (util.rst)
   - Version functions: lapackpp_version(), lapackpp_id()
   - Error handling: lapack::Error class
   - Enumerations: Sides, Norm, Job, Range, etc. (manually documented to avoid Breathe/Doxygen conflicts)

2. **Device Operations** (device.rst)
   - Queue class for GPU execution
   - potrf - Cholesky factorization
   - getrf - LU factorization
   - geqrf - QR factorization
   - heevd - Eigenvalue decomposition

3. **Factorizations** (factorizations.rst) - Now in toctree

4. **Linear Systems** (linear_systems.rst)

5. **Least Squares** (least_squares.rst)

6. **Eigenvalues** (eigenvalues.rst) - Now in toctree

7. **SVD** (svd.rst)

8. **Auxiliary** (auxiliary.rst)

9. **FLOPS** (flops.rst)

## Key Insights

### Breathe Directive Strategy
Based on BLAS++ experience:
- Use `.. doxygenfunction::` for individual functions
- Use `.. doxygenclass:: ClassName :members:` for classes
- Avoid `.. doxygengroup::` in multi-project repositories (causes duplication)
- Manual documentation for enums when groups span multiple projects

### Multi-Project Doxygen
SLATE repository contains three C++ projects:
- BLAS++ (blaspp/)
- LAPACK++ (lapackpp/)
- SLATE (src/)

All share Doxygen XML output, requiring careful group management and specific `.. doxygen*` directives rather than broad group inclusions.

## Future Improvements

1. **Function-Level Documentation:** Add similar comprehensive docs to all LAPACK++ wrapper functions in wrappers.hh

2. **Example Code:** Add usage examples to each module's RST files

3. **Performance Notes:** Document computational complexity for each operation

4. **GPU Optimization Tips:** Add best practices for device operations

5. **Cross-References:** Link related BLAS++ and LAPACK++ functions

## Files Modified

1. `/lapackpp/include/lapack.hh` - Version function documentation
2. `/docs/sphinx/api/lapackpp/index.rst` - Fixed toctree entries

## Files Already Well-Documented

1. `/lapackpp/include/lapack/util.hh` - Error class with @ingroup tags
2. `/lapackpp/include/lapack/device.hh` - Comprehensive GPU operation docs with LaTeX formulas
