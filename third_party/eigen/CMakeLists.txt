# Placeholder so that nothing walks into this directory expecting a buildable
# project: this is a *vendored header set*, used by the parent project through
# target_include_directories() (see ../../CMakeLists.txt), not built on its own.
#
# The upstream CMakeLists.txt is intentionally not kept: it defines targets that
# collide with the parent project (Eigen3::Eigen, install rules, BLAS/LAPACK
# detection) and pulls in the test suite and documentation, which are not part
# of this vendored copy. See VENDORED.md.
message(STATUS "third_party/eigen is a vendored header set; nothing to build here")
