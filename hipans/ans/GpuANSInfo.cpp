#include "hipans/ans/BatchProvider.h"
#include "hipans/ans/GpuANSCodec.h"
#include "hipans/ans/GpuANSInfo.h"

namespace hipans {

void ansGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<void*>(stream, (void**)in, numInBatch);
  ansGetCompressedInfoDevice(
      res,
      (const void**)in_dev.data(),
      numInBatch,
      outSizes_dev,
      outChecksum_dev,
      stream);
}

void ansGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto inProvider = BatchProviderPointer((void**)in_dev);
  ansGetCompressedInfo(
      inProvider, numInBatch, outSizes_dev, outChecksum_dev, stream);
}

} // namespace hipans
