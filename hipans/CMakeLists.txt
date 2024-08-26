execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX_PATH}")

hip_add_library(hipans SHARED
  HipANS.cpp
)
add_dependencies(hipans
  gpu_float_compress
)
target_link_libraries(hipans PRIVATE
	gpu_float_compress
  glog::glog
  "${TORCH_LIBRARIES}"
  )
target_include_directories(hipans PRIVATE
	$<BUILD_INTERFACE:${hipANS_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)

hip_add_executable(benchmark benchmark.cpp)
add_dependencies(benchmark
  gpu_float_compress
)
target_link_libraries(benchmark 
	gpu_float_compress
	${TORCH_LIBRARIES})
target_include_directories(benchmark PRIVATE
	$<BUILD_INTERFACE:${hipANS_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)
