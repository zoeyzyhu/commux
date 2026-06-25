cmake_minimum_required(VERSION 3.18)
project(commux LANGUAGES C CXX
        DESCRIPTION "UCX tag-matching c10d backend for PyTorch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(COMMUX_UCX_PROVIDER "auto" CACHE STRING
    "How to obtain UCX: auto (system, else build) | system | bundled")
set(COMMUX_WITH_GDRCOPY "auto" CACHE STRING
    "Build gdrcopy into bundled UCX: auto | on | off")
set(COMMUX_UCX_VERSION "1.18.0" CACHE STRING "UCX release to bundle")
set(COMMUX_GDRCOPY_VERSION "2.4.1" CACHE STRING "gdrcopy release to bundle")
option(COMMUX_CUDA "Enable CUDA (GPU) support if available" ON)
option(COMMUX_BUILD_TESTS "Build C++ tests" OFF)
option(COMMUX_BUILD_PYTHON "Build the Python extension module" OFF)

# ---------------------------------------------------------------------------
# Torch: discover libs/includes directly from the active interpreter rather than
# find_package(Torch). This always matches the torch the consumer uses, and --
# crucially -- avoids clashing with consumers (e.g. snapy) that already define
# an imported `torch` target via their own FindTorch.cmake (find_package(Torch)
# would error with "targets already defined").
# ---------------------------------------------------------------------------
if(NOT Python3_EXECUTABLE)
  find_program(Python3_EXECUTABLE NAMES python3 python REQUIRED)
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import torch,os;print(os.path.dirname(torch.__file__));print(int(torch._C._GLIBCXX_USE_CXX11_ABI));print(int(torch.version.cuda is not None))"
  OUTPUT_VARIABLE _torch_info OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_probe)
if(NOT _torch_probe EQUAL 0)
  message(FATAL_ERROR "commux: could not import torch with ${Python3_EXECUTABLE}")
endif()
string(REPLACE "\n" ";" _torch_lines "${_torch_info}")
list(GET _torch_lines 0 TORCH_INSTALL_PREFIX)
list(GET _torch_lines 1 _torch_abi)
list(GET _torch_lines 2 _torch_has_cuda)

set(TORCH_INCLUDE_DIRS
  "${TORCH_INSTALL_PREFIX}/include"
  "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
find_library(_COMMUX_TORCH     torch     HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(_COMMUX_TORCH_CPU torch_cpu HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(_COMMUX_C10       c10       HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
set(TORCH_LIBRARIES ${_COMMUX_TORCH} ${_COMMUX_TORCH_CPU} ${_COMMUX_C10})
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_torch_abi})
message(STATUS "commux: torch at ${TORCH_INSTALL_PREFIX} (ABI=${_torch_abi}, cuda=${_torch_has_cuda})")

# ---------------------------------------------------------------------------
# CUDA support (optional): only need c10_cuda for stream sync + the CUDA include.
# ---------------------------------------------------------------------------
set(_commux_cuda OFF)
set(COMMUX_CUDA_INCLUDE_DIRS "")
if(COMMUX_CUDA AND _torch_has_cuda)
  find_library(C10_CUDA_LIB c10_cuda HINTS "${TORCH_INSTALL_PREFIX}/lib")
  if(C10_CUDA_LIB)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(COMMUX_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include")
      set(COMMUX_CUDA_INCLUDE_DIRS "$ENV{CUDA_HOME}/include")
    elseif(EXISTS "/usr/local/cuda/include")
      set(COMMUX_CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    endif()
    set(_commux_cuda ON)
    message(STATUS "commux: CUDA support ON (${C10_CUDA_LIB}; inc=${COMMUX_CUDA_INCLUDE_DIRS})")
  else()
    message(STATUS "commux: CUDA support OFF (no c10_cuda in this torch)")
  endif()
endif()

# ---------------------------------------------------------------------------
# UCX provider: system, bundled, or auto (system first, else build).
# ---------------------------------------------------------------------------
set(_ucx_bundled OFF)
if(COMMUX_UCX_PROVIDER STREQUAL "system")
  find_package(UCX REQUIRED)
elseif(COMMUX_UCX_PROVIDER STREQUAL "bundled")
  set(_ucx_bundled ON)
else()  # auto
  find_package(UCX QUIET)
  if(NOT UCX_FOUND)
    message(STATUS "commux: no system UCX found -> bundling from source")
    set(_ucx_bundled ON)
  endif()
endif()
if(_ucx_bundled)
  include(BuildUCX)
endif()

# ---------------------------------------------------------------------------
# commux library (SHARED so it carries DT_NEEDED + rpath to UCX for consumers)
# ---------------------------------------------------------------------------
add_library(commux SHARED src/process_group_ucx.cpp)
add_library(commux::commux ALIAS commux)
set_target_properties(commux PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${UCX_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN:${UCX_LIBRARY_DIR}")
target_include_directories(commux PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  "${UCX_INCLUDE_DIR}")
target_include_directories(commux SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(commux PUBLIC ${TORCH_LIBRARIES} ${UCX_LIBRARIES})
if(_ucx_bundled)
  add_dependencies(commux commux_ucx_ext)
endif()
if(_commux_cuda)
  target_compile_definitions(commux PRIVATE COMMUX_WITH_CUDA=1)
  target_link_libraries(commux PUBLIC ${C10_CUDA_LIB})
  target_include_directories(commux SYSTEM PRIVATE ${COMMUX_CUDA_INCLUDE_DIRS})
endif()

# ---------------------------------------------------------------------------
# C++ test
# ---------------------------------------------------------------------------
if(COMMUX_BUILD_TESTS)
  add_executable(test_commux tests/test_commux.cpp)
  target_include_directories(test_commux SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(test_commux PRIVATE commux ${TORCH_LIBRARIES})
  if(_commux_cuda)
    target_compile_definitions(test_commux PRIVATE COMMUX_WITH_CUDA=1)
    target_link_libraries(test_commux PRIVATE ${C10_CUDA_LIB})
    target_include_directories(test_commux SYSTEM PRIVATE ${COMMUX_CUDA_INCLUDE_DIRS})
  endif()
  set_target_properties(test_commux PROPERTIES BUILD_RPATH "${UCX_LIBRARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# Python extension module (commux._C) -- built for the wheel.
# ---------------------------------------------------------------------------
if(COMMUX_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_library(TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")

  add_library(commux_pybind MODULE python/csrc/commux_pybind.cpp)
  set_target_properties(commux_pybind PROPERTIES
    PREFIX "" OUTPUT_NAME "_C")
  target_link_libraries(commux_pybind PRIVATE
    commux ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Module)
  target_include_directories(commux_pybind SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
  set_target_properties(commux_pybind PROPERTIES
    BUILD_RPATH "$ORIGIN/lib:${UCX_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN:${UCX_LIBRARY_DIR}")

  # Self-contained wheel layout -- a C++ consumer (e.g. snapy) needs no separate
  # UCX install:
  #   commux/_C.so                           python extension
  #   commux/lib/libcommux.so                C++ backend
  #   commux/lib/libuc*.so*  libgdrapi*      vendored UCX + gdrcopy runtime libs
  #   commux/include/commux/*.hpp            commux headers
  #   commux/include/{ucp,ucs,uct,ucm}/*.h   vendored UCX headers
  install(TARGETS commux_pybind LIBRARY DESTINATION commux)
  install(TARGETS commux LIBRARY DESTINATION commux/lib)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
          DESTINATION commux/include
          FILES_MATCHING PATTERN "*.hpp")
  # Vendor UCX (libuc*) + gdrcopy (libgdrapi) next to libcommux so its $ORIGIN
  # rpath resolves them, plus the UCX C headers so a consumer's
  # <ucp/api/ucp.h> resolves -- whether UCX was bundled or taken from UCX_ROOT.
  install(DIRECTORY "${UCX_LIBRARY_DIR}/" DESTINATION commux/lib
          FILES_MATCHING PATTERN "libuc*.so*" PATTERN "libgdrapi*.so*")
  foreach(_ucx_hdr ucp ucs uct ucm)
    if(EXISTS "${UCX_INCLUDE_DIR}/${_ucx_hdr}")
      install(DIRECTORY "${UCX_INCLUDE_DIR}/${_ucx_hdr}"
              DESTINATION commux/include FILES_MATCHING PATTERN "*.h")
    endif()
  endforeach()

  # Relocatability: the vendored UCX libs bake in an absolute build-tree rpath
  # that dangles once the wheel is installed elsewhere. Repoint each vendored
  # lib's rpath at $ORIGIN (and the UCX transport modules in lib/ucx/ at
  # $ORIGIN/..) so they resolve their siblings inside commux/lib regardless of
  # install location. Requires patchelf (a build-time dependency).
  find_program(PATCHELF_EXECUTABLE patchelf)
  if(PATCHELF_EXECUTABLE)
    install(CODE "
      file(GLOB _vend \"\${CMAKE_INSTALL_PREFIX}/commux/lib/lib*.so*\")
      foreach(_l \${_vend})
        if(NOT IS_SYMLINK \"\${_l}\")
          execute_process(COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"\$ORIGIN\" \"\${_l}\" ERROR_QUIET)
        endif()
      endforeach()
      file(GLOB _mods \"\${CMAKE_INSTALL_PREFIX}/commux/lib/ucx/*.so*\")
      foreach(_m \${_mods})
        if(NOT IS_SYMLINK \"\${_m}\")
          execute_process(COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"\$ORIGIN/..\" \"\${_m}\" ERROR_QUIET)
        endif()
      endforeach()
    ")
  else()
    message(WARNING "patchelf not found: vendored UCX libs keep their build-tree "
                    "rpath; the wheel may not be relocatable. 'pip install patchelf'.")
  endif()
endif()
