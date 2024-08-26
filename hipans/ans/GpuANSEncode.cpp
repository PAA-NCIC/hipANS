#include <cassert>
#include "hipans/ans/BatchProvider.h"
#include "hipans/ans/GpuANSEncode.h"
#include "hipans/ans/GpuANSUtils.h"

namespace hipans {

uint32_t getMaxCompressedSize(uint32_t uncompressedBytes) {
  uint32_t blocks = divUp(uncompressedBytes, kDefaultBlockSize);

  size_t rawSize = ANSCoalescedHeader::getCompressedOverhead(kDefaultBlockSize);
  rawSize += (size_t)getMaxBlockSizeCoalesced(kDefaultBlockSize) * blocks;

  // When used in batches, we must align everything to 16 byte boundaries (due
  // to uint4 read/writes)
  rawSize = roundUp(rawSize, sizeof(uint4));
  assert(rawSize <= std::numeric_limits<int32_t>::max());

  return rawSize;
}

void ansEncodeBatchStride(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void* in_dev,
    uint32_t inPerBatchSize,
    uint32_t inPerBatchStride,
    const uint32_t* histogram_dev,
    void* out_dev,
    uint32_t outPerBatchStride,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  auto inProvider =
      BatchProviderStride((void*)in_dev, inPerBatchStride, inPerBatchSize);
  auto outProvider = BatchProviderStride(out_dev, outPerBatchStride);

  ansEncodeBatchDevice(
      res,
      config,
      numInBatch,
      inProvider,
      histogram_dev,
      inPerBatchSize, // max size
      outProvider,
      outSize_dev,
      stream);
}

void ansEncodeBatchPointer(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void** in,
    const uint32_t* inSize,
    const uint32_t* histogram_dev,
    void** out,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  // Get the total and maximum input size
  uint32_t maxSize = 0;

  for (uint32_t i = 0; i < numInBatch; ++i) {
    uint32_t curSize = inSize[i] / sizeof(ANSDecodedT);
    maxSize = std::max(maxSize, curSize);
  }
  
  // Copy data to device
  auto in_dev = res.alloc<void*>(stream, numInBatch);
  auto inSize_dev = res.alloc<uint32_t>(stream, numInBatch);
  auto out_dev = res.alloc<void*>(stream, numInBatch);
  
  HIP_VERIFY(hipMemcpyAsync(
      in_dev.data(),
      in,
      numInBatch * sizeof(void*),
      hipMemcpyHostToDevice,
      stream));

  HIP_VERIFY(hipMemcpyAsync(
      inSize_dev.data(),
      inSize,
      numInBatch * sizeof(uint32_t),
      hipMemcpyHostToDevice,
      stream));

  HIP_VERIFY(hipMemcpyAsync(
      out_dev.data(),
      out,
      numInBatch * sizeof(void*),
      hipMemcpyHostToDevice,
      stream));

  auto inProvider =
      BatchProviderPointer((void**)in_dev.data(), inSize_dev.data());
  auto outProvider = BatchProviderPointer(out_dev.data());
 
  ansEncodeBatchDevice(
      res,
      config,
      numInBatch,
      inProvider,
      histogram_dev,
      maxSize,
      outProvider,
      outSize_dev,
      stream);
}

void ansEncodeBatchSplitSize(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    const void* in_dev,
    const uint32_t* inSplitSizes,
    const uint32_t* histogram_dev,
    void* out_dev,
    uint32_t outStride,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  // check alignment
  assert(uintptr_t(in_dev) % kANSRequiredAlignment == 0);

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = inSplitSizes[i];

    if (i != (numInBatch - 1)) {
	assert(size % kANSRequiredAlignment == 0);
      // check alignment (internal splits affect alignment of things after it)
    }

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
      sizeof(uint8_t));

  auto outProvider = BatchProviderStride(out_dev, outStride);

  ansEncodeBatchDevice(
      res,
      config,
      numInBatch,
      inProvider,
      histogram_dev,
      maxSplitSize,
      outProvider,
      outSize_dev,
      stream);
}

} // namespace hipans
