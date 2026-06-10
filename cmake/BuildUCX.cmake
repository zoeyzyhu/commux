# BuildUCX.cmake -- build UCX (and, optionally, gdrcopy's userspace library)
# from source via ExternalProject when no system UCX is available. This wraps
# their autotools/Makefile builds inside CMake so consumers never have to
# install UCX/gdrcopy by hand.
#
# Sets (parent scope): UCX_INCLUDE_DIR, UCX_LIBRARIES, UCX_LIBRARY_DIR, and the
# target `commux_ucx_ext` that the commux library must depend on.
#
# NOTE: builds from release tarballs (pre-generated ./configure), so the target
# only needs a C compiler + make + (for GPU) the CUDA toolkit -- no autotools.
# The gdrdrv kernel module is a separate host/driver prerequisite for gdr_copy
# to actually engage at runtime; without it UCX falls back to cuda_copy/cuda_ipc.

include(ExternalProject)

set(_ucx_prefix "${CMAKE_BINARY_DIR}/_deps/commux-ucx")
set(_ucx_install "${_ucx_prefix}/install")
set(UCX_INCLUDE_DIR "${_ucx_install}/include")
set(UCX_LIBRARY_DIR "${_ucx_install}/lib")
set(UCX_LIBRARIES
    "${UCX_LIBRARY_DIR}/libucp.so"
    "${UCX_LIBRARY_DIR}/libucs.so"
    "${UCX_LIBRARY_DIR}/libuct.so")

# PUBLIC include dirs must exist at generate time even though ExternalProject
# populates them at build time.
file(MAKE_DIRECTORY "${UCX_INCLUDE_DIR}")

# Resolve a CUDA home for the configure/make of UCX + gdrcopy.
set(_cuda_home "")
if(DEFINED CUDAToolkit_ROOT)
  set(_cuda_home "${CUDAToolkit_ROOT}")
elseif(DEFINED ENV{CUDA_HOME})
  set(_cuda_home "$ENV{CUDA_HOME}")
elseif(EXISTS "/usr/local/cuda")
  set(_cuda_home "/usr/local/cuda")
endif()

# --- gdrcopy userspace lib (optional) --------------------------------------
set(_ucx_gdr_flag "--without-gdrcopy")
set(_ucx_depends "")
set(_want_gdrcopy OFF)
if(_cuda_home AND NOT COMMUX_WITH_GDRCOPY STREQUAL "off")
  set(_want_gdrcopy ON)
endif()

if(_want_gdrcopy)
  set(_gdr_install "${_ucx_install}")  # install gdrcopy into the same prefix
  ExternalProject_Add(commux_gdrcopy_ext
    URL "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${COMMUX_GDRCOPY_VERSION}.tar.gz"
    PREFIX "${_ucx_prefix}/gdrcopy"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make CUDA=${_cuda_home} lib
    INSTALL_COMMAND make prefix=${_gdr_install} CUDA=${_cuda_home} lib_install
    BUILD_BYPRODUCTS "${_gdr_install}/lib/libgdrapi.so"
    LOG_DOWNLOAD 1 LOG_BUILD 1 LOG_INSTALL 1)
  set(_ucx_gdr_flag "--with-gdrcopy=${_gdr_install}")
  list(APPEND _ucx_depends commux_gdrcopy_ext)
  message(STATUS "commux: bundled UCX will be built --with-gdrcopy (v${COMMUX_GDRCOPY_VERSION})")
endif()

# --- UCX --------------------------------------------------------------------
set(_ucx_cuda_flag "--without-cuda")
if(_cuda_home)
  set(_ucx_cuda_flag "--with-cuda=${_cuda_home}")
endif()

ExternalProject_Add(commux_ucx_build
  URL "https://github.com/openucx/ucx/releases/download/v${COMMUX_UCX_VERSION}/ucx-${COMMUX_UCX_VERSION}.tar.gz"
  PREFIX "${_ucx_prefix}/ucx"
  DEPENDS ${_ucx_depends}
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND <SOURCE_DIR>/configure
      --prefix=${_ucx_install}
      --enable-mt
      ${_ucx_cuda_flag}
      ${_ucx_gdr_flag}
      --without-java
      --disable-doxygen-doc
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS ${UCX_LIBRARIES}
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1)

# Single target the commux library depends on.
add_custom_target(commux_ucx_ext DEPENDS commux_ucx_build)

message(STATUS "commux: bundling UCX v${COMMUX_UCX_VERSION} -> ${_ucx_install}")
