# Eigen 3.4.0 (vendored)

This directory is an unmodified copy of the Eigen 3.4.0 headers, trimmed of the
upstream test suite, documentation and benchmarks (which this project does not
build). The library code under `Eigen/` and `unsupported/Eigen/` is untouched.

* Upstream: https://gitlab.com/libeigen/eigen
* Version: 3.4.0 (release tarball)
* License: MPL2 (see `COPYING.MPL2`), with the other upstream licenses alongside

CMake uses this copy only when no system Eigen is found:

```bash
cmake -S . -B build            # uses vendored Eigen if Eigen3 is not installed
```

To upgrade, replace this directory with a newer Eigen release and keep the same
trim: `Eigen/`, `unsupported/`, `signature_of_eigen3_matrix_library`, the
`COPYING.*` files and `README.md`.
