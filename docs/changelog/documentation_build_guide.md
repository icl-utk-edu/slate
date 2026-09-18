# Documentation Build Quick Reference

## Building Complete Documentation

### Build Documentation
```bash
cd /Users/mitch/github/slate/docs/sphinx
make clean html
```

### View Documentation
```bash
open _build/html/index.html
```

## Common Issues

### Issue: "Cannot find function in doxygen xml output"
**Cause:** Function missing proper Doxygen comments or `@ingroup` tag
**Fix:** Add complete Doxygen documentation to source header file, rebuild Doxygen XML

### Issue: "document isn't included in any toctree"
**Cause:** RST file exists but not referenced in index.rst
**Fix:** Add filename to `.. toctree::` block in appropriate index.rst

### Issue: "300+ duplicate declaration warnings"
**Cause:** Using `.. doxygengroup::` pulls ALL entities from multi-project repository
**Fix:** Use specific `.. doxygenfunction::` or `.. doxygenclass::` directives instead

### Issue: LaTeX math not rendering
**Cause:** Using wrong syntax `:math:\` instead of LaTeX `\f$ \f$` in Doxygen
**Fix:** In C++ headers use `\f$ formula \f$`, in RST use `:math:\`formula\``

## Breathe Directive Reference

### Functions
```rst
.. doxygenfunction:: namespace::function_name
   :project: BLASPP
```

### Classes
```rst
.. doxygenclass:: namespace::ClassName
   :project: LAPACKPP
   :members:
```

### Enums (avoid in multi-project repos)
```rst
.. doxygenenum:: namespace::EnumName
   :project: BLASPP
```

### Groups (dangerous in multi-project repos)
```rst
.. doxygengroup:: groupname
   :project: BLASPP
```
**Warning:** Groups span entire XML output, causing duplicates!

## Project Structure

```
slate/
├── blaspp/
│   └── include/blas/       # BLAS++ headers with Doxygen comments
├── lapackpp/
│   └── include/lapack/     # LAPACK++ headers with Doxygen comments
├── src/                     # SLATE headers with Doxygen comments
└── docs/
    ├── doxygen/
    │   ├── Doxyfile        # Doxygen configuration
    │   └── xml/            # Generated XML (input for Breathe)
    └── sphinx/
        ├── Makefile
        ├── conf.py         # Sphinx configuration
        ├── api/
        │   ├── blaspp/     # BLAS++ RST files
        │   ├── lapackpp/   # LAPACK++ RST files
        │   └── slate/      # SLATE RST files
        └── _build/html/    # Generated HTML documentation
```

## Doxygen Comment Best Practices

### Function Documentation
```cpp
/// Brief description (one line)
///
/// Detailed description with mathematical formulas.
/// Use LaTeX: \f$ A = B + C \f$
///
/// @param[in] x
///     Input parameter with detailed description
///
/// @param[out] y
///     Output parameter
///
/// @param[in,out] z
///     Modified parameter
///
/// @return Description of return value
///
/// @ingroup groupname
template <typename T>
T function_name(int64_t x, T* y, T* z);
```

### Class Documentation
```cpp
/// Brief description of class purpose
///
/// Detailed description of class functionality,
/// usage patterns, and important notes.
///
/// @ingroup groupname
class ClassName {
public:
    /// Constructor description
    ClassName();
    
    /// Member function description
    void method();
};
```

### Enum Documentation
```cpp
/// Enum description
///
/// @ingroup groupname
enum class EnumName {
    Value1,  ///< Description of Value1
    Value2   ///< Description of Value2
};
```

## RST Best Practices

### Module Structure
```rst
Module Name
===========

Brief module introduction.

Overview
--------

Detailed overview with usage examples.

Functions
---------

.. doxygenfunction:: namespace::func1
   :project: PROJECT

.. doxygenfunction:: namespace::func2
   :project: PROJECT
```

### Math in RST
```rst
This is inline math: :math:`A = B + C`

This is display math:

.. math::

   A = B + C
```

### Cross-References
```rst
See :doc:`../blaspp/level1` for BLAS Level 1 operations.
See :cpp:func:`blas::gemm` for matrix multiplication.
```

## Incremental Workflow

1. Edit C++ header files with Doxygen comments
2. Rebuild Doxygen: `cd docs/doxygen && doxygen Doxyfile`
3. Rebuild Sphinx: `cd docs/sphinx && make html`
4. Check for warnings in build output
5. View in browser: `open _build/html/index.html`
6. Iterate on warnings/errors

## Quality Checks

Before committing:
1. No Sphinx warnings
2. All toctree entries valid
3. All referenced functions exist in Doxygen XML
4. Math formulas render correctly
5. Code examples compile
6. Links work (no 404s)
