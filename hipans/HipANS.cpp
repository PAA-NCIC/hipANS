#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <torch/types.h>
#include <vector>
#include "hipans/ans/GpuANSCodec.h"
#include "hipans/utils/StackDeviceMemory.h"
#include <iostream>

namespace hipans {

namespace {

// returns (totalSize, maxSize)
std::tuple<int64_t, int64_t> getTotalAndMaxSize(
    const std::vector<torch::Tensor>& tIns) {
  int64_t totalSize = 0;
  int64_t maxSize = 0;

  for (auto& t : tIns) {
    auto curSize = t.numel();
    // FIXME: due to int indexing, it's really total size
    TORCH_CHECK(
        curSize * t.element_size() <= std::numeric_limits<uint32_t>::max());

    totalSize += curSize;
    maxSize = std::max(maxSize, curSize);
  }

  TORCH_CHECK(maxSize <= std::numeric_limits<uint32_t>::max());

  return std::make_tuple(totalSize, maxSize);
}

} // namespace

//
// External API
//

constexpr int kDefaultPrecision = 10;

std::tuple<int64_t, int64_t> max_any_compressed_output_size(
    const std::vector<torch::Tensor>& ts) {
  auto sizes = getTotalAndMaxSize(ts);
  int64_t maxBytes = std::get<1>(sizes) * ts[0].element_size();

  return std::make_tuple(ts.size(), getMaxCompressedSize(maxBytes));
}

int64_t max_any_compressed_size(int64_t bytes) {
  return getMaxCompressedSize(bytes);
}

//////////////////////
//
// Compress
//
//////////////////////

std::tuple<torch::Tensor, torch::Tensor, int64_t> compress_data_res(
    StackDeviceMemory& res,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const c10::optional<torch::Tensor>& outCompressed,
    const c10::optional<torch::Tensor>& outCompressedSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  auto maxOutputSize = max_any_compressed_output_size(tIns);
  //
  // Validate input and validate / construct output
  //
  for (auto& t : tIns) {
    TORCH_CHECK(t.device().type() == at::kCUDA);
    TORCH_CHECK(t.is_contiguous());

    // device must be consistent
    TORCH_CHECK(t.get_device() == dev);

  }

  torch::Tensor comp;
  if (outCompressed) {
    TORCH_CHECK(outCompressed->dtype() == torch::kByte);
    TORCH_CHECK(outCompressed->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressed->is_contiguous());
    TORCH_CHECK(outCompressed->dim() == 2);
    TORCH_CHECK(outCompressed->size(0) >= tIns.size());
    TORCH_CHECK(outCompressed->size(1) >= std::get<1>(maxOutputSize));
    TORCH_CHECK(outCompressed->get_device() == dev);

    comp = *outCompressed;
  } else {
    comp = torch::empty(
        {(int64_t)tIns.size(), std::get<1>(maxOutputSize)},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Byte));
  }

  auto inPtrs = std::vector<const void*>(tIns.size());
  auto inSize = std::vector<uint32_t>(tIns.size());
  auto compPtrs = std::vector<void*>(tIns.size());

  for (size_t i = 0; i < tIns.size(); ++i) {
    auto& t = tIns[i];

    inPtrs[i] = t.data_ptr();
    inSize[i] = (t.numel() * t.element_size());
    compPtrs[i] = (uint8_t*)comp.data_ptr() + i * comp.size(1);
  }

  //
  // Validate / construct output sizes
  //
  torch::Tensor sizes;
  if (outCompressedSizes) {
    TORCH_CHECK(outCompressedSizes->dtype() == torch::kInt);
    TORCH_CHECK(outCompressedSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressedSizes->dim() == 1);
    TORCH_CHECK(outCompressedSizes->is_contiguous());
    TORCH_CHECK(outCompressedSizes->size(0) >= tIns.size());
    TORCH_CHECK(outCompressedSizes->get_device() == dev);

    sizes = *outCompressedSizes;
  } else {
    // FIXME: no uint32 in torch
    sizes = torch::empty(
        {(int64_t)tIns.size()},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Int));
  }
    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

  	hipStream_t stream;
	  HIP_VERIFY(hipStreamCreate(&stream));

    ansEncodeBatchPointer(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        inSize.data(),
        nullptr,
        compPtrs.data(),
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        stream);

  // how much temporary memory we actually used
  int64_t tempMemUsage = res.getMaxMemoryUsage();
  return std::make_tuple(std::move(comp), std::move(sizes), tempMemUsage);
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> compress_data(
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const c10::optional<torch::Tensor>& tempMem,
    const c10::optional<torch::Tensor>& outCompressed,
    const c10::optional<torch::Tensor>& outCompressedSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());

    // Should be on the same device as the first tensor passed
    TORCH_CHECK(tempMem->get_device() == tIns.front().get_device());
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  // The rest of the validation takes place here
  return compress_data_res(
      res, tIns, checksum, outCompressed, outCompressedSizes);
}

//////////////////////
//
// Decompress
//
//////////////////////

int64_t decompress_data_res(
    StackDeviceMemory& res,
    const std::vector<torch::Tensor>& tIns,
    const std::vector<torch::Tensor>& tOuts,
    bool checksum,
    const c10::optional<torch::Tensor>& outStatus,
    const c10::optional<torch::Tensor>& outSizes) {
  TORCH_CHECK(!tIns.empty());
  TORCH_CHECK(tIns.size() == tOuts.size());

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate input and output
  auto inPtrs = std::vector<const void*>(tIns.size());
  auto outPtrs = std::vector<void*>(tIns.size());
  auto outCapacity = std::vector<uint32_t>(tOuts.size());

  for (size_t i = 0; i < tIns.size(); ++i) {
    auto& tIn = tIns[i];
    auto& tOut = tOuts[i];

    TORCH_CHECK(tIn.device().type() == at::kCUDA);
    TORCH_CHECK(tIn.get_device() == dev);
    TORCH_CHECK(tIn.is_contiguous());

    TORCH_CHECK(tOut.device().type() == at::kCUDA);
    TORCH_CHECK(tOut.get_device() == dev);
    TORCH_CHECK(tOut.is_contiguous());

    TORCH_CHECK(tIn.dtype() == torch::kByte);

    inPtrs[i] = tIn.data_ptr();
    outPtrs[i] = tOut.data_ptr();

    auto outSize = (tOut.numel() * tOut.element_size());

    // FIXME: total range checking
    TORCH_CHECK(outSize <= std::numeric_limits<uint32_t>::max());
    outCapacity[i] = outSize;
  }

  // Validate outStatus, if passed
  if (outStatus) {
    TORCH_CHECK(outStatus->is_contiguous());
    TORCH_CHECK(outStatus->device().type() == at::kCUDA);
    TORCH_CHECK(outStatus->dtype() == torch::kByte);
    TORCH_CHECK(outStatus->numel() == tIns.size());
    TORCH_CHECK(outStatus->get_device() == dev);
  }

  // Validate outSizes, if passed
  if (outSizes) {
    TORCH_CHECK(outSizes->is_contiguous());
    TORCH_CHECK(outSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outSizes->dtype() == torch::kInt32);
    TORCH_CHECK(outSizes->numel() == tIns.size());
    TORCH_CHECK(outSizes->get_device() == dev);
  }

    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

  	hipStream_t stream;
	  HIP_VERIFY(hipStreamCreate(&stream));

    auto decStatus = ansDecodeBatchPointer(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        outPtrs.data(),
        outCapacity.data(),
        outStatus ? (uint8_t*)outStatus->data_ptr() : nullptr,
        // FIXME: int32_t versus uint32_t
        outSizes ? (uint32_t*)outSizes->data_ptr() : nullptr,
        stream);

    TORCH_CHECK(
        decStatus.error != ANSDecodeError::ChecksumMismatch,
        "ANSDecode: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");

  // how much temporary memory we actually used
  return res.getMaxMemoryUsage();
}

int64_t decompress_data(
    const std::vector<torch::Tensor>& tIns,
    const std::vector<torch::Tensor>& tOuts,
    bool checksum,
    const c10::optional<torch::Tensor>& tempMem,
    const c10::optional<torch::Tensor>& outStatus,
    const c10::optional<torch::Tensor>& outSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());
    TORCH_CHECK(tempMem->get_device() == tIns.front().get_device());
    // we don't care about data type, we just care about memory
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  // Rest of validation happens here
  return decompress_data_res(
      res, tIns, tOuts, checksum, outStatus, outSizes);
}

} // namespace hipans

TORCH_LIBRARY_FRAGMENT(hipans, m) {
  // compression size
  m.def("max_any_compressed_output_size(Tensor[] ts) -> (int, int)");
  m.def("max_any_compressed_size(int bytes) -> int");
  // data compress
  m.def(
      "compress_data(Tensor[] ts_in, bool checksum=False, Tensor? temp_mem=None, Tensor? out_compressed=None, Tensor? out_compressed_bytes=None) -> (Tensor, Tensor, int)");
  // data decompress
  m.def(
      "decompress_data(Tensor[] ts_in, Tensor[] ts_out, bool checksum=False, Tensor? temp_mem=None, Tensor? out_status=None, Tensor? out_decompressed_words=None) -> (int)");
}

TORCH_LIBRARY(hipans, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("hipans::max_any_compressed_output_size"),
      TORCH_FN(hipans::max_any_compressed_output_size));
  m.impl(
      TORCH_SELECTIVE_NAME("hipans::max_any_compressed_size"),
      TORCH_FN(hipans::max_any_compressed_size));
  m.impl(
      TORCH_SELECTIVE_NAME("hipans::compress_data"),
      TORCH_FN(hipans::compress_data));
  m.impl(
      TORCH_SELECTIVE_NAME("hipans::decompress_data"),
      TORCH_FN(hipans::decompress_data));
}
