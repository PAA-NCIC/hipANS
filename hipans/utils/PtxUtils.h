/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace hipans {

__device__ __forceinline__ unsigned int
getBitfield(uint8_t val, int pos, int len) {
  unsigned int ret;
  int num=(1<<len)-1;
  ret=(val>>(pos-1))&num;
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(uint16_t val, int pos, int len) {
  unsigned int ret;
  int num=(1<<len)-1;
  ret=(val>>(pos-1))&num;
  return ret;
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos, int len) {
  unsigned int ret;
  int num=(1<<len)-1;
  ret=(val>>(pos-1))&num;
  return ret;
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
  uint64_t ret;
  int num=(1<<len)-1;
  ret=(val>>(pos-1))&num;
  return ret;
}

__device__ __forceinline__ unsigned int
setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  unsigned int ret;
  unsigned int temp = ((1<<len)-1)&toInsert;
  temp = temp << pos;//400
  ret = ((1<<len)-1)<<pos;
  ret = ((~ret) & val) | temp;
  return ret;
}

__device__ __forceinline__ uint32_t rotateLeft(uint32_t v, uint32_t shift) {
  uint32_t out;
  uint32_t n = min(shift, 32);
  out = (v << n) | (v >> (32 - n));
  return out;
}

__device__ __forceinline__ uint32_t rotateRight(uint32_t v, uint32_t shift) {
  uint32_t out;
  uint32_t n = min(shift, 32);
  out = (v << (32 - n)) | (v >> n);
  return out;
}

__device__ __forceinline__ int getLaneId() {
  int laneId;
  laneId = threadIdx.x % kWarpSize;
  return laneId;
}

__device__ __forceinline__ uint64_t getLaneMaskLt() {
  uint64_t mask;
  int laneId;
  laneId = threadIdx.x % kWarpSize;
  mask = (1ULL << laneId) - 1;
  return mask;
}

__device__ __forceinline__ uint64_t getLaneMaskLe() {
  uint64_t mask;
  int laneId;
  laneId = threadIdx.x % kWarpSize;
  if(laneId == 63)
  mask = 0xffffffffffffffff;
  mask =(1ULL << (laneId+1)) - 1;
  return mask;
}

__device__ __forceinline__ uint64_t getLaneMaskGt() {
  uint64_t mask;
  int laneId;
  laneId = threadIdx.x % kWarpSize;
  if(laneId == 63)
  mask = 0xffffffffffffffff;
  else
  mask = (1ULL << (laneId+1)) - 1;
  mask = ~mask;
  return mask;
}

__device__ __forceinline__ uint64_t getLaneMaskGe() {
  uint64_t mask;
  int laneId;
  laneId = threadIdx.x % kWarpSize;
  mask = (1ULL << laneId) - 1;
  mask = ~mask;
  return mask;
}

template <typename T>
__device__ inline T warpReduceAllMin(T val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = min(val, __shfl_xor(val, mask, kWarpSize));
  }

  return val;
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllMax(T val) {
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor(val, mask, kWarpSize));
  }

  return val;
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor(val, mask, kWarpSize);
  }

  return val;
}

} // namespace hipans
