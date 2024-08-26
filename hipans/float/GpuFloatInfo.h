#pragma once

#include "hip/hip_runtime.h"
#include <assert.h>
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/float/GpuFloatUtils.h"
#include "hipans/utils/DeviceUtils.h"
#include "hipans/utils/StackDeviceMemory.h"
#include "hipans/utils/StaticUtils.h"

namespace hipans {

template <typename InProvider>
__global__ void floatGetCompressedInfoKernel(
    InProvider inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes,
    uint32_t* outTypes,
    uint32_t* outChecksum) {
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto header = (const GpuFloatHeader*)inProvider.getBatchStart(batch);
    header->checkMagicAndVersion();

    if (outSizes) {
      outSizes[batch] = header->size;
    }
    if (outTypes) {
      outTypes[batch] = uint32_t(header->getFloatType());
    }
    if (outChecksum) {
      assert(header->getUseChecksum());
      outChecksum[batch] = header->getChecksum();
    }
  }
}

template <typename InProvider>
void floatGetCompressedInfo(
    InProvider& inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outTypes_dev && !outTypes_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  floatGetCompressedInfoKernel<<<grid, block, 0, stream>>>(
      inProvider, numInBatch, outSizes_dev, outTypes_dev, outChecksum_dev);

  HIP_TEST_ERROR();
}

} // namespace hipans
