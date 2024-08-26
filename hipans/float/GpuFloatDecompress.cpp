#include "hipans/ans/BatchProvider.h"
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/float/GpuFloatDecompress.h"
#include "hipans/utils/DeviceUtils.h"
#include "hipans/utils/StackDeviceMemory.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace hipans {

FloatDecompressStatus floatDecompress(
    StackDeviceMemory& res,
    const FloatDecompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    void** out,
    const uint32_t* outCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  // If the batch size is <= kBSLimit, we avoid hipMemcpy and send all data at
  // kernel launch
  constexpr int kLimit = 128;

  // Investigate all of the output pointers; are they 16 byte aligned? If so, we
  // can do everything in a single pass
  bool is16ByteAligned = true;
  for (int i = 0; i < numInBatch; ++i) {
    if (reinterpret_cast<uintptr_t>(out[i]) % 16 != 0) {
      is16ByteAligned = false;
      break;
    }
  }

  auto updatedConfig = config;
  updatedConfig.is16ByteAligned = is16ByteAligned;

  // We need a max capacity estimate before proceeding, for temporary memory
  // allocations
  uint32_t maxCapacity = 0;
  for (uint32_t i = 0; i < numInBatch; ++i) {
    maxCapacity = std::max(maxCapacity, outCapacity[i]);
  }

  if (numInBatch <= kLimit) {
    // We can do everything in a single pass without a h2d memcpy
    auto inProvider =
        BatchProviderInlinePointer<kLimit>(numInBatch, (void**)in);
    auto outProvider = BatchProviderInlinePointerCapacity<kLimit>(
        numInBatch, out, outCapacity);

    return floatDecompressDevice(
        res,
        updatedConfig,
        numInBatch,
        inProvider,
        outProvider,
        maxCapacity,
        outSuccess_dev,
        outSize_dev,
        stream);
  }

  // Copy data to device
  // To reduce latency, we prefer to coalesce all data together and copy as one
  // contiguous chunk
  assert(sizeof(void*) == sizeof(uintptr_t));
  assert(sizeof(uint32_t) <= sizeof(uintptr_t));

  // in, out, outCapacity
  auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
  auto params_host =
      std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

  std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
  std::memcpy(&params_host[numInBatch], out, numInBatch * sizeof(void*));
  std::memcpy(
      &params_host[2 * numInBatch], outCapacity, numInBatch * sizeof(uint32_t));

  HIP_VERIFY(hipMemcpyAsync(
      params_dev.data(),
      params_host.get(),
      3 * numInBatch * sizeof(uintptr_t),
      hipMemcpyHostToDevice,
      stream));

  auto in_dev = params_dev.data();
  auto out_dev = params_dev.data() + numInBatch;
  auto outCapacity_dev = (const uint32_t*)(params_dev.data() + 2 * numInBatch);

  auto inProvider = BatchProviderPointer((void**)in_dev);
  auto outProvider = BatchProviderPointer((void**)out_dev, outCapacity_dev);

  return floatDecompressDevice(
      res,
      updatedConfig,
      numInBatch,
      inProvider,
      outProvider,
      maxCapacity,
      outSuccess_dev,
      outSize_dev,
      stream);
}

FloatDecompressStatus floatDecompressSplitSize(
    StackDeviceMemory& res,
    const FloatDecompressConfig& config,
    uint32_t numInBatch,
    const void** in,
    void* out,
    const uint32_t* outSplitSizes,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    hipStream_t stream) {
  auto floatWordSize = getWordSizeFromFloatType(config.floatType);

  // Concatenate splitSize and splitSizePrefix together for a single h2d copy
  auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
  auto splitSize = splitSizeHost.data();
  auto splitSizePrefix = splitSizeHost.data() + numInBatch;
  uint32_t maxSplitSize = 0;

  bool is16ByteAligned = isPointerAligned(out, 16);

  for (uint32_t i = 0; i < numInBatch; ++i) {
    auto size = outSplitSizes[i];

    // If we only have one tensor in the batch, we only care if the start
    // pointer is 16 byte aligned.
    // Otherwise, all sizes except for the final one must ensure that all
    // splits give a 16 byte alignment.
    if ((i != numInBatch - 1) && (size % (16 / floatWordSize) != 0)) {
      is16ByteAligned = false;
    }

    splitSize[i] = size;
    if (i > 0) {
      splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
    }

    maxSplitSize = std::max(size, maxSplitSize);
  }

  auto sizes_dev = res.copyAlloc<uint32_t>(stream, splitSizeHost);

  // FIXME: combine with above for a single h2d copy
  auto in_dev = res.copyAlloc<void*>(stream, (void**)in, numInBatch);

  auto updatedConfig = config;
  updatedConfig.is16ByteAligned = is16ByteAligned;

  auto inProvider = BatchProviderPointer(in_dev.data());

  auto outProvider = BatchProviderSplitSize(
      out, sizes_dev.data(), sizes_dev.data() + numInBatch, floatWordSize);

  return floatDecompressDevice(
      res,
      updatedConfig,
      numInBatch,
      inProvider,
      outProvider,
      maxSplitSize,
      outSuccess_dev,
      outSize_dev,
      stream);
}

} // namespace hipans
