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
  return (val>>(pos-1))&((1<<len)-1);
}

__device__ __forceinline__ unsigned int
getBitfield(uint16_t val, int pos, int len) {
  return (val>>(pos-1))&((1<<len)-1);
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos, int len) {
  return (val>>(pos-1))&((1<<len)-1);
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
  return (val>>(pos-1))&((1<<len)-1);
}

__device__ __forceinline__ unsigned int
setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  return ((~(((1<<len)-1)<<pos)) & val) | ((((1<<len)-1)&toInsert) << pos);
}

__device__ __forceinline__ uint32_t rotateLeft(uint32_t v, uint32_t shift) {
  uint32_t n = min(shift, 32);
  return (v << n) | (v >> (32 - n));
}

__device__ __forceinline__ uint32_t rotateRight(uint32_t v, uint32_t shift) {
  uint32_t n = min(shift, 32);
  return (v << (32 - n)) | (v >> n);
}
__device__ __forceinline__ int getLaneId() {
  return threadIdx.x & 63;
}
__device__ __forceinline__ uint64_t getLaneMaskLt() {
  return (1ULL << getLaneId()) - 1;
}

__device__ __forceinline__ uint64_t getLaneMaskLe() {
  return (1ULL << (getLaneId() + 1)) - 1;
}

__device__ __forceinline__ uint64_t getLaneMaskGt() {
  return (~((1ULL << (getLaneId() + 1)) - 1)) & 0xFFFFFFFFFFFFFFFF;
}

__device__ __forceinline__ uint64_t getLaneMaskGe() {
  return ~((1ULL << getLaneId()) - 1);
}

template <typename T>
__device__ inline T warpReduceAllMin(T val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) 
    val = min(val, __shfl_xor(val, mask, kWarpSize));
  return val;
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllMax(T val) {
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor(val, mask, kWarpSize));
  return val;
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) 
    val += __shfl_xor(val, mask, kWarpSize);
  return val;
}

} // namespace hipans