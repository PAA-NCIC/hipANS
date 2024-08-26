#pragma once

#include "hip/hip_runtime.h"
#include "hipans/ans/GpuANSUtils.h"
#include "hipans/utils/DeviceUtils.h"
#include "hipans/utils/StackDeviceMemory.h"
#include "hipans/utils/StaticUtils.h"

namespace hipans {

template <typename InProvider>
__global__ void ansGetCompressedInfoKernel(
    InProvider inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes,
    uint32_t* outChecksum) {
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    auto header = (const ANSCoalescedHeader*)inProvider.getBatchStart(batch);
    // Make sure it is valid
    header->checkMagicAndVersion();

    if (outSizes) {
      outSizes[batch] = header->getTotalUncompressedWords();
    }

    if (outChecksum) {
      assert(header->getUseChecksum());
      outChecksum[batch] = header->getChecksum();
    }
  }
}

template <typename InProvider>
void ansGetCompressedInfo(
    InProvider& inProvider,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outChecksum_dev,
    hipStream_t stream) {
  if (!outSizes_dev && !outChecksum_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  ansGetCompressedInfoKernel<<<grid, block, 0, stream>>>(
      inProvider, numInBatch, outSizes_dev, outChecksum_dev);

  HIP_TEST_ERROR();
}

} // namespace hipans
