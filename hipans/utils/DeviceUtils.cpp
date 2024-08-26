#include "hipans/utils/DeviceUtils.h"
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <unordered_map>
#include <cassert> 
#include <iostream>

namespace hipans {

//std::string errorToString(hipError_t err) {
//  return std::string(hipGetErrorString(err));
//}

//std::string errorToName(hipError_t err) {
//  return std::string(hipGetErrorName(err));
//}

int getCurrentDevice() {
  int dev = -1;
  HIP_VERIFY(hipGetDevice(&dev));
  assert(dev != -1 && "dev should not be equal to -1");
  return dev;
}

void setCurrentDevice(int device) {
  HIP_VERIFY(hipSetDevice(device));
}

int getNumDevices() {
  int numDev = -1;
  hipError_t err = hipGetDeviceCount(&numDev);
  if (hipErrorNoDevice == err) {
    numDev = 0;
  } else {
    HIP_VERIFY(err);
  }
  assert(numDev != -1 && "numDev should not be equal to -1");
  return numDev;
}

void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    DeviceScope scope(i);

    HIP_VERIFY(hipDeviceSynchronize());
  }
}

const hipDeviceProp_t& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, hipDeviceProp_t> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    hipDeviceProp_t prop;
    HIP_VERIFY(hipGetDeviceProperties(&prop, device));

    properties[device] = prop;
    it = properties.find(device);
  }

  return it->second;
}

const hipDeviceProp_t& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }

  hipPointerAttribute_t att;
  hipError_t err = hipPointerGetAttributes(&att, p);
  assert(err == hipSuccess || err == hipErrorInvalidValue && "unknown error");
  if (!(err == hipSuccess || err == hipErrorInvalidValue)) {
    std::cerr << "unknown error " << static_cast<int>(err) << std::endl;
  }

  if (err == hipErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = hipGetLastError();
    assert(err == hipErrorInvalidValue && "Expected hipErrorInvalidValue, but got a different error");
    if (err != hipErrorInvalidValue) {
    std::cerr << "unknown error " << static_cast<int>(err) << std::endl;
    }
    return -1;
  }

  // memoryType is deprecated for high version
  if (att.memoryType == hipMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
}

bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

DeviceScope::DeviceScope(int device) {
  if (device >= 0) {
    int curDevice = getCurrentDevice();

    if (curDevice != device) {
      prevDevice_ = curDevice;
      setCurrentDevice(device);
      return;
    }
  }

  // Otherwise, we keep the current device
  prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
  if (prevDevice_ != -1) {
    setCurrentDevice(prevDevice_);
  }
}

HipEvent::HipEvent(hipStream_t stream, bool timer) : event_(nullptr) {
  HIP_VERIFY(hipEventCreateWithFlags(
      &event_, timer ? hipEventDefault : hipEventDisableTiming));
  HIP_VERIFY(hipEventRecord(event_, stream));
}

HipEvent::HipEvent(HipEvent&& event) noexcept
    : event_(std::move(event.event_)) {
  event.event_ = nullptr;
}

HipEvent::~HipEvent() {
  if (event_) {
    HIP_VERIFY(hipEventDestroy(event_));
  }
}

HipEvent& HipEvent::operator=(HipEvent&& event) noexcept {
  event_ = std::move(event.event_);
  event.event_ = nullptr;

  return *this;
}

void HipEvent::streamWaitOnEvent(hipStream_t stream) {
  HIP_VERIFY(hipStreamWaitEvent(stream, event_, 0));
}

void HipEvent::cpuWaitOnEvent() {
  HIP_VERIFY(hipEventSynchronize(event_));
}

float HipEvent::timeFrom(HipEvent& from) {
  cpuWaitOnEvent();
  float ms = 0;
  HIP_VERIFY(hipEventElapsedTime(&ms, from.event_, event_));

  return ms;
}

HipStream::HipStream(int flags) : stream_(nullptr) {
  HIP_VERIFY(hipStreamCreateWithFlags(&stream_, flags));
}

HipStream::HipStream(HipStream&& stream) noexcept
    : stream_(std::move(stream.stream_)) {
  stream.stream_ = nullptr;
}

HipStream::~HipStream() {
  if (stream_) {
    HIP_VERIFY(hipStreamDestroy(stream_));
  }
}

HipStream& HipStream::operator=(HipStream&& stream) noexcept {
  stream_ = std::move(stream.stream_);
  stream.stream_ = nullptr;

  return *this;
}

HipStream HipStream::make() {
  return HipStream();
}

HipStream HipStream::makeNonBlocking() {
  return HipStream(hipStreamNonBlocking);
}

} // namespace hipans
