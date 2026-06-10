# FindUCX.cmake -- locate a system/module UCX installation.
#
# Honors -DUCX_ROOT=/path (or env UCX_ROOT), then pkg-config, then default
# prefixes. Set UCX_ROOT to a custom build (e.g. ~/ucx-install).
#
# Provides: UCX_FOUND, UCX_INCLUDE_DIR, UCX_LIBRARIES (ucp;ucs;uct),
# UCX_LIBRARY_DIR, and imported target UCX::ucx.

if(NOT UCX_ROOT AND DEFINED ENV{UCX_ROOT})
  set(UCX_ROOT "$ENV{UCX_ROOT}")
endif()

# pkg-config gives a good hint on HPC systems (module load ucx).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND AND NOT UCX_ROOT)
  pkg_check_modules(_PC_UCX QUIET ucx)
endif()

find_path(UCX_INCLUDE_DIR
  NAMES ucp/api/ucp.h
  HINTS "${UCX_ROOT}/include" ${_PC_UCX_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(UCX_UCP_LIBRARY ucp
  HINTS "${UCX_ROOT}/lib" ${_PC_UCX_LIBRARY_DIRS} PATH_SUFFIXES lib lib64)
find_library(UCX_UCS_LIBRARY ucs
  HINTS "${UCX_ROOT}/lib" ${_PC_UCX_LIBRARY_DIRS} PATH_SUFFIXES lib lib64)
find_library(UCX_UCT_LIBRARY uct
  HINTS "${UCX_ROOT}/lib" ${_PC_UCX_LIBRARY_DIRS} PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCX
  REQUIRED_VARS UCX_INCLUDE_DIR UCX_UCP_LIBRARY UCX_UCS_LIBRARY UCX_UCT_LIBRARY)

if(UCX_FOUND)
  set(UCX_LIBRARIES ${UCX_UCP_LIBRARY} ${UCX_UCS_LIBRARY} ${UCX_UCT_LIBRARY})
  get_filename_component(UCX_LIBRARY_DIR "${UCX_UCP_LIBRARY}" DIRECTORY)
  if(NOT TARGET UCX::ucx)
    add_library(UCX::ucx INTERFACE IMPORTED)
    set_target_properties(UCX::ucx PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${UCX_LIBRARIES}")
  endif()
  message(STATUS "commux: found system UCX at ${UCX_LIBRARY_DIR}")
endif()

mark_as_advanced(UCX_INCLUDE_DIR UCX_UCP_LIBRARY UCX_UCS_LIBRARY UCX_UCT_LIBRARY)
