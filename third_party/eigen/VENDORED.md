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

**Delete Eigen's own `.gitignore` when you do.** It is written for upstream's
repository, and one of its rules is the bare word `core`. Git matches that
case-sensitively, so it excludes `Eigen/Core` and `Eigen/src/Core/` — the heart
of the library — from any Linux checkout. The developer machine that vendored
the code does not notice (macOS filesystems are case-insensitive, so the files
are still on disk and the build still works); CI notices immediately, because a
fresh checkout on Linux genuinely does not have them. The top-level
`.gitignore` carries `!third_party/eigen/Eigen/**` exceptions as a second line
of defence, but the vendored `.gitignore` should simply not be kept.
