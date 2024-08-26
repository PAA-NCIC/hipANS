#include "hipans/ans/BatchProvider.h"
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/float/GpuFloatCompress.h"
#include "hipans/float/GpuFloatUtils.h"
#include "hipans/utils/DeviceUtils.h"
#include "hipans/utils/StackDeviceMemory.h"
#include "hipans/utils/StaticUtils.h"

#include <cmath>
#include <memory>
#include <vector>
#include <cassert>  
#include <iostream>

namespace hipans {

uint32_t getMaxFloatCompressedSize(FloatType floatType, uint32_t size) {
  // kNotCompressed bytes per float are simply stored uncompressed
  // rounded up to 16 bytes to ensure alignment of the following ANS data
  // portion
  uint32_t baseSize = sizeof(GpuFloatHeader) + getMaxCompressedSize(size);

  switch (floatType) {
    case FloatType::kFloat16:
      baseSize += FloatTypeInfo<FloatType::kFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kBFloat16:
      baseSize += FloatTypeInfo<FloatType::kBFloat16>::getUncompDataSize(size);
      break;
    case FloatType::kFloat32:
      baseSize += FloatTypeInfo<FloatType::kFloat32>::getUncompDataSize(size);
      break;
    default:
      assert(false);
      break;
  }

  return baseSize;
}

void floatCompress(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    const uint32_t* inSize,
    void** out,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  // Get the total and maximum input size
  uint32_t maxSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxSize = std::max(maxSize, inSize[i]);
  }

  // Copy data to device
  // To reduce latency, we prefer to coalesce all data together and copy as one
  // contiguous chunk
  static_assert(sizeof(void*) == sizeof(uintptr_t), "");
  static_assert(sizeof(uint32_t) <= sizeof(uintptr_t), "");

  // in, inSize, out
  auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
  auto params_host =
      std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

  memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  memcpy(&params_host[numInBatch], inSize, numInBatch * sizeof(uint32_t));
  memcpy(&params_host[2 * numInBatch], out, numInBatch * sizeof(void*));

  HIP_VERIFY(hipMemcpyAsync(
      params_dev.data(),
      params_host.get(),
      3 * numInBatch * sizeof(uintptr_t),
      hipMemcpyHostToDevice,
      stream));

  auto in_dev = (const void**)params_dev.data();
  auto inSize_dev = (const uint32_t*)(params_dev.data() + numInBatch);
  auto out_dev = (void**)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev, inSize_dev);
  auto outProvider = BatchProviderPointer(out_dev);
  floatCompressDevice(
      res,
      config,
      numInBatch,
      inProvider,
      maxSize,
      outProvider,
      outSize_dev,
      stream);
}

void floatCompressSplitSize(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    const void* in_dev,
    const uint32_t* inSplitSizes,
    void* out_dev,
    uint32_t outStride,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  auto floatWordSize = getWordSizeFromFloatType(config.floatType);

  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = inSplitSizes[i];

    splitSize[i] = size;
    if (i > 0) {
      splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
    }

    maxSplitSize = std::max(size, maxSplitSize);
  }

  // Copy data to device
  // splitSize, splitSizePrefix
  auto sizes_dev = res.alloc<uint32_t>(stream, splitSizeHost.size());

  HIP_VERIFY(hipMemcpyAsync(
      sizes_dev.data(),
      splitSizeHost.data(),
      splitSizeHost.size() * sizeof(uint32_t),
      hipMemcpyHostToDevice,
      stream));

  auto inProvider = BatchProviderSplitSize(
      (void*)in_dev,
      sizes_dev.data(),
      sizes_dev.data() + numInBatch,
      floatWordSize);

  auto outProvider = BatchProviderStride(out_dev, outStride);

  floatCompressDevice(
      res,
      config,
      numInBatch,
      inProvider,
      maxSplitSize,
      outProvider,
      outSize_dev,
      stream);
}

} // namespace hipans