#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "hipans/ans/GpuANSCodec.h"
#include "hipans/ans/GpuANSUtils.h"
#include "hipans/utils/StackDeviceMemory.h"

using namespace hipans;
std::vector<GpuMemoryReservation<uint8_t>> buffersToDevice(
    StackDeviceMemory& res,
    const std::vector<uint32_t>& sizes,
    hipStream_t stream) {
  auto out = std::vector<GpuMemoryReservation<uint8_t>>();

  for (auto& s : sizes) {
    out.emplace_back(res.alloc<uint8_t>(stream, s, AllocType::Permanent));
  }

  return out;
}

void compressFileWithANS(StackDeviceMemory& res,
		const std::string& inputFilePath,
		const std::string& tempFilePath,
		std::vector<uint32_t>& batchSizes,
		std::vector<void*>& encPtrs,
		std::vector<uint32_t>& compressedSize,
		int precision,
		uint32_t& outBatchStride,
		hipStream_t stream
		) {
        
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    if (inputFile.fail()) {
        throw std::runtime_error("Cannot open input file");
    }
    std::streamsize fileSize = inputFile.tellg();
    std::vector<uint8_t> fileData(fileSize);

    inputFile.seekg(0, std::ios::beg);
    inputFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);
    
    inputFile.close();
    
    auto devData = res.alloc<uint8_t>(stream, static_cast<uint64_t>(fileSize));
    hipMemcpy(devData.data(), fileData.data(), static_cast<size_t>(fileSize), hipMemcpyHostToDevice);
    
    auto num = fileSize;
    int bsize;
    if(fileSize % 536870912 == 0){
       bsize = fileSize / 536870912;
    batchSizes.resize(bsize);
    for(int i=0;i<batchSizes.size();i++)
        batchSizes[i] = 536870912;
    }
    else {
    bsize = fileSize / 536870912 + 1;
    batchSizes.resize(bsize);
    for(int i=0;i<batchSizes.size()-1;i++){
        batchSizes[i] = 536870912;
        num -= 536870912;
    }
    batchSizes[bsize - 1] = num;
    }
 
    auto numInBatch = batchSizes.size();
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
      maxSize = std::max(maxSize, v);
    }
    outBatchStride = getMaxCompressedSize(maxSize);
   
    auto outCompressedSizeDev = res.alloc<uint32_t>(stream, numInBatch);

    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < inPtrs.size(); ++i) {
        inPtrs[i] = (uint8_t*)devData.data() + curOffset;
        curOffset += batchSizes[i];
      }
    }
    
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);
 
    encPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
       encPtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
    }

    double comp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();   
    ansEncodeBatchPointer(
        res,
        hipans::ANSCodecConfig(precision, true),
        numInBatch, 
        inPtrs.data(),
        batchSizes.data(),
        nullptr,
        encPtrs.data(),
        outCompressedSizeDev.data(),
        stream);
    auto end = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    uint64_t total_size = fileSize;
    double c_bw = ( 1.0 * total_size / 1e9 ) / ( comp_time * 1e-3 );  

    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;

    compressedSize = outCompressedSizeDev.copyToHost(stream);
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    if (!outputFile) {
        throw std::runtime_error("Cannot open output file");
    }
    for(int i=0;i<numInBatch;i++){
        std::vector<uint8_t> compressedData(compressedSize[i]);
        hipMemcpy(compressedData.data(), encPtrs.data()[i], compressedSize[i]*sizeof(uint8_t), hipMemcpyDeviceToHost);
        outputFile.write(reinterpret_cast<const char*>(compressedData.data()), compressedSize[i]*sizeof(uint8_t));
    }
    outputFile.close();
}

void decompressFileWithANS(StackDeviceMemory& res,
		const std::string& tempFilePath, 
		const std::string& outputFilePath,                 
		std::vector<uint32_t>& batchSizes,
        std::vector<void*>& encPtrs,
        std::vector<uint32_t>& compressedSize,                
		int precision,
		uint32_t& outBatchStride,
		hipStream_t stream) {
    auto numInBatch = batchSizes.size();
    
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

    auto filePtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < batchSizes.size(); ++i) {
      filePtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
    }
    std::ifstream inFile(tempFilePath, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Cannot open input file");
    }
    for(int i=0;i<batchSizes.size();i++){
        std::vector<uint8_t> fileCompressedData(compressedSize[i]);
        inFile.read(reinterpret_cast<char*>(fileCompressedData.data()), compressedSize[i]);
        hipMemcpy(filePtrs[i],fileCompressedData.data(),compressedSize[i]*sizeof(uint8_t),hipMemcpyHostToDevice);
    }
    inFile.close();
   
    auto dec_dev = buffersToDevice(res, batchSizes, stream);
    auto decPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < batchSizes.size(); ++i) {
        decPtrs[i] = dec_dev[i].data();
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    double decomp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    ansDecodeBatchPointer(
        res,
        ANSCodecConfig(precision, true),
        numInBatch,
        (const void**)filePtrs.data(), 
        decPtrs.data(),
        batchSizes.data(),
        outSuccess_dev.data(),
        outSize_dev.data(),
        stream);
    auto end = std::chrono::high_resolution_clock::now();  
    decomp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    uint64_t comp_size = 0;
    for(auto v: compressedSize)
        comp_size += v;
    double dc_bw = ( 1.0 * comp_size / 1e9 ) / ( decomp_time * 1e-3 );

    std::cout << "decomp time " << std::fixed << std::setprecision(3) << decomp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    
    std::ofstream outFile(outputFilePath, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Cannot open output file");
    }
    for(int i=0;i<batchSizes.size();i++){
        std::vector<uint8_t> unCompressData(batchSizes[i]);
        hipMemcpy(unCompressData.data(),decPtrs[i],batchSizes[i]*sizeof(uint8_t),hipMemcpyDeviceToHost);
        outFile.write(reinterpret_cast<const char*>(unCompressData.data()), batchSizes[i]*sizeof(uint8_t));
    }
    outFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.img> <temp.ans> <output.img>" << std::endl;
        return 1;
    }
    auto res = makeStackMemory();
    hipStream_t stream;   
    HIP_VERIFY(hipStreamCreate(&stream));
    std::vector<uint32_t> batchSizes;
    std::vector<void*> encPtrs;
    std::vector<uint32_t> compressedSize;
    int precision = 10; // 根据需要设置精度
    uint32_t outBatchStride;
    try {
        compressFileWithANS(res,argv[1], argv[2],batchSizes,encPtrs,compressedSize,precision,outBatchStride, stream);
	    decompressFileWithANS(res,argv[2],argv[3],batchSizes,encPtrs,compressedSize,precision,outBatchStride, stream);
        std::cout << "Compression completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
