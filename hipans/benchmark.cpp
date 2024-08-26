#include <iostream>
#include <fstream>
#include <vector>  
#include <random>  
#include <cstdint> 
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <torch/types.h>
#include <torch/torch.h> 
#include <chrono>
#include "hipans/ans/GpuANSCodec.h"
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/utils/StackDeviceMemory.h"
  
using namespace hipans;

std::tuple<size_t, size_t, double> calc_comp_ratio(
    const std::vector<torch::Tensor>& input_ts,
    const torch::Tensor& compressed_sizes_tensor) {
  
  size_t total_input_size = 0;
  size_t total_comp_size = 0;

  TORCH_CHECK(compressed_sizes_tensor.dim() == 1, "compressed_sizes_tensor must be a 1D tensor");
  TORCH_CHECK(compressed_sizes_tensor.size(0) == input_ts.size(), "The size of compressed_sizes_tensor does not match the number of input tensors");

  auto compressed_sizes = compressed_sizes_tensor.to(torch::kCPU).to(torch::kInt);

  for (size_t i = 0; i < input_ts.size(); ++i) {
    total_input_size += static_cast<size_t>(input_ts[i].numel()) * input_ts[i].element_size();

    total_comp_size += static_cast<size_t>(compressed_sizes[i].item<int64_t>());
  }

  double compression_ratio = static_cast<double>(total_comp_size) / total_input_size;

  return std::make_tuple(total_input_size, total_comp_size, compression_ratio);
}

FloatType getFloatTypeFromDtype(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half:
      return FloatType::kFloat16;
    case at::ScalarType::BFloat16:
      return FloatType::kBFloat16;
    case at::ScalarType::Float:
      return FloatType::kFloat32;
    default:
      TORCH_CHECK(
          t == at::ScalarType::Half || t == at::ScalarType::BFloat16 ||
          t == at::ScalarType::Float);
      return FloatType::kUndefined;
  }
}

at::ScalarType getDtypeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
      return at::ScalarType::Half;
    case FloatType::kBFloat16:
      return at::ScalarType::BFloat16;
    case FloatType::kFloat32:
      return at::ScalarType::Float;
    default:
      TORCH_CHECK(
          ft == FloatType::kFloat16 || ft == FloatType::kBFloat16 ||
          ft == FloatType::kFloat32);
      return at::ScalarType::Half;
  }
}

FloatType getFloatTypeFromTensor(const torch::Tensor& t) {
  return getFloatTypeFromDtype(t.dtype().toScalarType());
}

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

// Convert a compressed matrix into a list of tensors that are views into the
// compressed row pieces
std::vector<torch::Tensor> compressedMatrixToTensors(
    int numInBatch,
    torch::Tensor& matrix_dev,
    torch::Tensor& sizes_dev) {
	hipStream_t stream;
	HIP_VERIFY(hipStreamCreate(&stream));

  // We wish to return narrowed tensors with a view into the matrix
  auto sizes_host = std::vector<uint32_t>(numInBatch);

  HIP_VERIFY(hipMemcpyAsync(
      sizes_host.data(),
      sizes_dev.data_ptr(),
      sizeof(uint32_t) * numInBatch,
      hipMemcpyDeviceToHost,
      stream));

  auto out = std::vector<torch::Tensor>(numInBatch);

  auto matrix1d = matrix_dev.view({matrix_dev.numel()});
  auto cols = matrix_dev.size(1);

  for (int i = 0; i < numInBatch; ++i) {
    out[i] = matrix1d.narrow(0, i * cols, sizes_host[i]);//narrow：0：以行参照，1：以列参照
  }

  return out;
}

constexpr int kDefaultPrecision = 10;

std::tuple<int64_t, int64_t> max_float_compressed_output_size(
    const std::vector<torch::Tensor>& ts) {
  auto sizes = getTotalAndMaxSize(ts);

  auto maxCompSize = getMaxFloatCompressedSize(
      getFloatTypeFromTensor(ts[0]), std::get<1>(sizes));

  return std::make_tuple(ts.size(), maxCompSize);
}

// FIXME: can we pass a dtype somehow instead?
int64_t max_float_compressed_size(const torch::Tensor& dtype, int64_t size) {
  return getMaxFloatCompressedSize(getFloatTypeFromTensor(dtype), size);
}

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
    bool compressAsFloat,
    StackDeviceMemory& res,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const c10::optional<torch::Tensor>& outCompressed,
    const c10::optional<torch::Tensor>& outCompressedSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  auto maxOutputSize = compressAsFloat ? max_float_compressed_output_size(tIns)
                                       : max_any_compressed_output_size(tIns);
  //
  // Validate input and validate / construct output
  //
  for (auto& t : tIns) {
    TORCH_CHECK(t.device().type() == at::kCUDA);
    TORCH_CHECK(t.is_contiguous());

    // device must be consistent
    TORCH_CHECK(t.get_device() == dev);

    // must be all the same type unless we are compressing bytewise
    if (compressAsFloat) {
      TORCH_CHECK(t.dtype() == tIns[0].dtype());

      // must be a supported float type
      TORCH_CHECK(
          getFloatTypeFromDtype(t.dtype().toScalarType()) !=
          FloatType::kUndefined);
    }
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
    inSize[i] = compressAsFloat ? t.numel() : (t.numel() * t.element_size());
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
  if (compressAsFloat) {
    auto config = FloatCompressConfig(
        getFloatTypeFromTensor(tIns[0]),
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we'll figure this out later */,
        checksum);

  	hipStream_t stream;
	  HIP_VERIFY(hipStreamCreate(&stream));
    floatCompress(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        inSize.data(),
        compPtrs.data(),
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        stream);
  } else {
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
  }

  // how much temporary memory we actually used
  int64_t tempMemUsage = res.getMaxMemoryUsage();
  return std::make_tuple(std::move(comp), std::move(sizes), tempMemUsage);
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> compress_data(
    bool compressAsFloat,
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
      compressAsFloat, res, tIns, checksum, outCompressed, outCompressedSizes);
}

//////////////////////
//
// Decompress
//
//////////////////////

int64_t decompress_data_res(
    bool compressAsFloat,
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
    if (compressAsFloat) {
      TORCH_CHECK(
          tOut.dtype() == torch::kFloat16 || tOut.dtype() == torch::kBFloat16 ||
          tOut.dtype() == torch::kFloat32);
    }

    inPtrs[i] = tIn.data_ptr();
    outPtrs[i] = tOut.data_ptr();

    auto outSize =
        compressAsFloat ? tOut.numel() : (tOut.numel() * tOut.element_size());

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

  if (compressAsFloat) {
    auto config = FloatDecompressConfig(
        getFloatTypeFromTensor(tOuts[0]),
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we'll figure this out later */,
        checksum);

  	hipStream_t stream;
	  HIP_VERIFY(hipStreamCreate(&stream));

    auto decStatus = floatDecompress(
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
        decStatus.error != FloatDecompressError::ChecksumMismatch,
        "floatDecompress: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");
  } else {
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
  }

  // how much temporary memory we actually used
  return res.getMaxMemoryUsage();
}

int64_t decompress_data(
    bool compressAsFloat,
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
      compressAsFloat, res, tIns, tOuts, checksum, outStatus, outSizes);
}

std::tuple<double, double, size_t, size_t> get_float_comp_timings(const std::vector<torch::Tensor>& ts, int num_runs = 3) {  
    torch::Device dev = torch::kCUDA;
    torch::Tensor tempMem = torch::empty({384 * 1024 * 1024}, torch::kUInt8).to(dev);  

    double comp_time = 0.0;  
    double decomp_time = 0.0;  
    size_t total_size = 0;  
    size_t comp_size = 0;  

    for (int i = 0; i < num_runs + 1; ++i) {    
        auto [rows, cols] = max_float_compressed_output_size(ts);
  
        torch::Tensor comp = torch::empty({rows, cols}, torch::kUInt8).to(dev);  
        torch::Tensor sizes = torch::zeros({static_cast<long>(ts.size())}, torch::kInt).to(dev);  
        int64_t memUsed;

        auto start = std::chrono::high_resolution_clock::now();
        std::tie(comp, sizes, memUsed) = compress_data(true, ts, false, tempMem, comp, sizes); 
        auto end = std::chrono::high_resolution_clock::now();  

        if (i > 0) {  
            comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
        }  
 
        std::tie(total_size, comp_size, std::ignore) = calc_comp_ratio(ts, sizes);  
  
        std::vector<torch::Tensor> out_ts;  
        for (const auto& t : ts) {  
            out_ts.push_back(torch::empty_like(t).to(t.device()));  
        }  
  
        std::vector<torch::Tensor> comp_ts = comp.split(1, 0);
  
        torch::Tensor out_status = torch::empty({static_cast<long>(ts.size())}, torch::kUInt8).to(dev);  
        torch::Tensor out_sizes = torch::empty({static_cast<long>(ts.size())}, torch::kInt32).to(dev);  
  
        auto start1 = std::chrono::high_resolution_clock::now();  
        decompress_data(true, comp_ts, out_ts, false, tempMem, out_status, out_sizes);
        auto end1 = std::chrono::high_resolution_clock::now();

        if (i > 0) {  
            decomp_time += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e3;  
        }

        for (size_t j = 0; j < ts.size(); ++j) {
            assert(torch::equal(ts[j], out_ts[j]));
        }
    }   

    comp_time /= num_runs;  
    decomp_time /= num_runs;  
  
    return std::make_tuple(comp_time, decomp_time, total_size, comp_size);
}

std::tuple<double, double, size_t, size_t> get_any_comp_timings(const std::vector<torch::Tensor>& ts, int num_runs = 3) {  \
    torch::Device dev = torch::kCUDA;
    torch::Tensor tempMem = torch::empty({384 * 1024 * 1024}, torch::kUInt8).to(dev);
  
    double comp_time = 0.0;  
    double decomp_time = 0.0;  
    size_t total_size = 0;  
    size_t comp_size = 0;    
  
    for (int i = 0; i < num_runs + 1; ++i) {   
        auto [rows, cols] = max_any_compressed_output_size(ts);  
 
        torch::Tensor comp = torch::empty({rows, cols}, torch::dtype(torch::kUInt8).device(torch::kCUDA));  
        torch::Tensor sizes = torch::zeros({static_cast<long>(ts.size())}, torch::dtype(torch::kInt).device(torch::kCUDA));  
        int64_t memUsed;
         
        auto start = std::chrono::high_resolution_clock::now();
        std::tie(comp, sizes, memUsed) = compress_data(false, ts, true,  tempMem, comp, sizes);  
        auto end = std::chrono::high_resolution_clock::now();  

        if (i > 0) {  
            comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
        }  

        std::tie(total_size, comp_size, std::ignore) = calc_comp_ratio(ts, sizes);  
  
        std::vector<torch::Tensor> out_ts;  
        for (const auto& t : ts) {  
            out_ts.push_back(torch::empty_like(t).to(t.device()));  
        }  
  
        std::vector<torch::Tensor> comp_ts = comp.split(1, 0);
  
        torch::Tensor out_status = torch::empty({static_cast<long>(ts.size())}, torch::kUInt8).to(dev);  
        torch::Tensor out_sizes = torch::empty({static_cast<long>(ts.size())}, torch::kInt32).to(dev);  
  
        auto start1 = std::chrono::high_resolution_clock::now();    
        decompress_data(false, comp_ts, out_ts, true, tempMem, out_status, out_sizes);
        auto end1 = std::chrono::high_resolution_clock::now(); 

        if (i > 0) {  
            decomp_time += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e3;  
        }

        for (size_t j = 0; j < ts.size(); ++j) {  
            assert(torch::equal(ts[j], out_ts[j]));  
        }
    }   

    comp_time /= num_runs;  
    decomp_time /= num_runs;  
  
    return std::make_tuple(comp_time, decomp_time, total_size, comp_size);
}

int main(int argc, char* argv[]) {  
    if(argc == 1){ 
    torch::init_num_threads();  
    torch::Device device(torch::kCUDA); 

    std::vector<torch::ScalarType> dtypes = {torch::kBFloat16, torch::kFloat16, torch::kFloat32};  
    for (torch::ScalarType dt : dtypes) {   
        torch::TensorOptions options = torch::TensorOptions().dtype(dt).device(device);  
        std::vector<torch::Tensor> ts;
        ts.push_back(torch::randn({128 * 512 * 1024}, options));  
  
        double c, dc;  
        size_t total_size, comp_size;  
        std::tie(c, dc, total_size, comp_size) = get_float_comp_timings(ts);  
  
        double ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        double c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        double dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Float codec non-batched perf [128 * 512 * 1024] " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;  
   
        std::vector<torch::Tensor> ts_batch;
        for(int i = 0; i < 128; i++)
        ts_batch.push_back(torch::randn({512 * 1024}, options));
  
        std::tie(c, dc, total_size, comp_size) = get_float_comp_timings(ts_batch);  
  
        ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Float codec batched perf [128 , [512 * 1024]] " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    }  
    
    std::cout<<std::endl;

    for (torch::ScalarType dt : dtypes) {    
        torch::TensorOptions options = torch::TensorOptions().dtype(dt).device(device);  
        std::vector<torch::Tensor> ts;
        ts.push_back(torch::randn({128 * 512 * 1024}, options));

        double c, dc;  
        size_t total_size, comp_size;  
        std::tie(c, dc, total_size, comp_size) = get_any_comp_timings(ts);  
  
        double ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        double c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        double dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Raw ANS byte-wise batched perf [128 * 512 * 1024] " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;  
           
        std::vector<torch::Tensor> ts_batch;
        for(int i = 0; i < 128; i++)
        ts_batch.push_back(torch::randn({512 * 1024}, options)); // 使用randn代替normal，因为LibTorch没有normal函数  
  
        std::tie(c, dc, total_size, comp_size) = get_any_comp_timings(ts_batch);  
  
        ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Raw ANS byte-wise batched perf [128 , [512 * 1024]] " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    }  
    }
    else if(argc > 1){
        std::string file_path = argv[1];
        std::ifstream data_file(file_path, std::ios::binary);
        data_file.seekg(0, std::ios::end);
        size_t size = data_file.tellg();
        data_file.seekg(0, std::ios::beg);

        std::vector<float> data(size / sizeof(float));
        data_file.read(reinterpret_cast<char*>(data.data()), size);
        data_file.close();
        torch::Tensor tensor = torch::from_blob(data.data(), {static_cast<int64_t>(data.size())});
        auto dt = torch::kFloat32;
        tensor = tensor.toType(dt);
        tensor = tensor.to(torch::kCUDA);
        std::vector<torch::Tensor> ts;
        ts.push_back(tensor);

        double c, dc;  
        size_t total_size, comp_size;  
        std::tie(c, dc, total_size, comp_size) = get_float_comp_timings(ts);  
 
        double ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        double c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        double dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Float codec img perf " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
        
        std::tie(c, dc, total_size, comp_size) = get_any_comp_timings(ts);  
  
        ratio = static_cast<double>(comp_size) / static_cast<double>(total_size);  
        c_bw = ( 1.0 * total_size / 1e9 ) / ( c * 1e-3 );  
        dc_bw = ( 1.0 * comp_size / 1e9 ) / ( dc * 1e-3 );  
  
        std::cout << "Raw ANS byte-wise img perf " << torch::toString(dt) << std::endl;  
        std::cout << "comp   time " << std::fixed << std::setprecision(3) << c << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s, compression "   
                  << total_size << " -> " << comp_size << " bytes ("   
                  << std::fixed << std::setprecision(4) << ratio << "x)" << std::endl;  
        std::cout << "decomp time " << std::fixed << std::setprecision(3) << dc << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
        
    }
    return 0;

}
