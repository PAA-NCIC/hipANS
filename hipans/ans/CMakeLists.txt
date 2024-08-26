hip_add_library(gpu_ans SHARED
  GpuANSDecode.cpp
  GpuANSEncode.cpp
  GpuANSInfo.cpp
)
add_dependencies(gpu_ans
  hipans_utils
)
target_include_directories(gpu_ans PUBLIC
	$<BUILD_INTERFACE:${hipANS_SOURCE_DIR}>
)
target_link_libraries(gpu_ans PUBLIC
  hipans_utils
)
target_link_libraries(gpu_ans PRIVATE
 glog::glog
)
target_compile_options(gpu_ans PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
  --generate-line-info
>)


enable_testing()
include(GoogleTest)

hip_add_executable(ans_test ANSTest.cpp)
target_link_libraries(ans_test
 gpu_ans
 gtest_main
)
gtest_discover_tests(ans_test)

hip_add_executable(ans_statistics_test ANSStatisticsTest.cpp)
target_link_libraries(ans_statistics_test
  gpu_ans
  gtest_main
  hipans_utils
)
gtest_discover_tests(ans_statistics_test)

hip_add_executable(batch_prefix_sum_test BatchPrefixSumTest.cpp)
target_link_libraries(batch_prefix_sum_test
  gpu_ans
  gtest_main
)
gtest_discover_tests(batch_prefix_sum_test)

hip_add_executable(compress compress.cpp)
target_link_libraries(compress
	hipans_utils
	gpu_ans
        ${TORCH_LIBRARIES})
target_include_directories(compress PRIVATE
  $<BUILD_INTERFACE:${hipans_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)

