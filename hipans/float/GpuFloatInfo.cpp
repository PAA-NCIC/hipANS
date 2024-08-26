#include "hipans/ans/BatchProvider.h"
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/float/GpuFloatInfo.h"
#include "hipans/utils/DeviceUtils.h"
#include "hipans/utils/StackDeviceMemory.h"
#include "hipans/utils/StaticUtils.h"

namespace hipans {

void floatGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outChecksum_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<const void*>(stream, in, numInBatch);

  floatGetCompressedInfoDevice(
      res,
      in_dev.data(),
      numInBatch,
      outSizes_dev,
      outTypes_dev,
      outChecksum_dev,
      stream);

  HIP_TEST_ERROR();
}

void floatGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outChecksum_dev) {
    return;
  }

  auto inProvider = BatchProviderPointer((void**)in_dev);

  floatGetCompressedInfo(
      inProvider,
      numInBatch,
      outSizes_dev,
      outTypes_dev,
      outChecksum_dev,
      stream);
}

} // namespace hipans
