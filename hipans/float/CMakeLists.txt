set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

hip_add_library(gpu_float_compress SHARED
  GpuFloatCompress.cpp
  GpuFloatDecompress.cpp
  GpuFloatInfo.cpp
)
add_dependencies(gpu_float_compress
  gpu_ans
  hipans_utils
)

target_include_directories(gpu_float_compress PUBLIC
 $<BUILD_INTERFACE:${hipans_SOURCE_DIR}>
)
target_link_libraries(gpu_float_compress PUBLIC
  gpu_ans
  hipans_utils
)
target_link_libraries(gpu_float_compress PRIVATE
 glog::glog
)
target_compile_options(gpu_float_compress PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
  --generate-line-info
  #--device-debug
>)

enable_testing()
include(GoogleTest)

hip_add_executable(float_test FloatTest.cpp)
target_link_libraries(float_test
  gpu_float_compress
 gtest_main
)

hip_add_executable(fpcompress fpcompress.cpp)
target_link_libraries(fpcompress
  gpu_float_compress
 gtest_main
)

hip_add_executable(fp16compress fp16compress.cpp)
target_link_libraries(fp16compress
  gpu_float_compress
 gtest_main
)

hip_add_executable(fpB16compress fpB16compress.cpp)
target_link_libraries(fpB16compress
  gpu_float_compress
 gtest_main
)


gtest_discover_tests(float_test)

