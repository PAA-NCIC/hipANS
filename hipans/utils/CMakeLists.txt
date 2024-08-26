hip_add_library(hipans_utils SHARED
  DeviceUtils.cpp
  StackDeviceMemory.cpp
)

target_include_directories(hipans_utils PUBLIC
	$<BUILD_INTERFACE:${hipANS_SOURCE_DIR}>
 "${HIP_INCLUDE_DIRS}"
)
target_link_libraries(hipans_utils PUBLIC
 ${HIP_LIBRARIES}
 glog::glog
)
