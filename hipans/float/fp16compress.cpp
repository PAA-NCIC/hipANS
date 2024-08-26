#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <chrono>
#include <iomanip>
#include "hipans/float/GpuFloatCodec.h"
#include "hipans/float/GpuFloatUtils.h"
#include "hipans/utils/StackDeviceMemory.h"

using namespace hipans;

void compressFloat16File(StackDeviceMemory& res, const std::string& inputFilePath, const std::string& tempFilePath, std::vector<uint32_t>& batchSizes, size_t& numElements, int probBits, uint32_t& maxCompressedSize, int& totalSize, std::vector<uint32_t>& compressedSize, int& numInBatch,hipStream_t stream) {
    auto compConfig = FloatCompressConfig(FloatType::kFloat16, ANSCodecConfig(probBits), false, true);
    std::ifstream inputFile(inputFilePath, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << inputFilePath << std::endl;
        return;
    }
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    numElements = fileSize / sizeof(uint16_t);
    std::vector<FloatTypeInfo<FloatType::kFloat16>::WordT> orig(numElements);
    inputFile.read(reinterpret_cast<char*>(orig.data()), fileSize);
    if (!inputFile) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    inputFile.close();

    auto orig_dev = res.copyAlloc(stream, orig);
    batchSizes = {static_cast<uint32_t>(numElements)};
    numInBatch = batchSizes.size();

    totalSize = 0;
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
      totalSize += v;
      maxSize = std::max(maxSize, v);
    }
    
    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < inPtrs.size(); ++i) {
        inPtrs[i] = (FloatTypeInfo<FloatType::kFloat16>::WordT*)orig_dev.data() + curOffset;
        curOffset += batchSizes[i]*sizeof(FloatTypeInfo<FloatType::kFloat16>::WordT);
      }
    }

    maxCompressedSize = getMaxFloatCompressedSize(FloatType::kFloat16, maxSize);
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

    auto encPtrs = std::vector<void*>(batchSizes.size());
    {
      for (int i = 0; i < inPtrs.size(); ++i) {
        encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
      }
    }

    auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);
    double comp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    floatCompress(
        res,
        compConfig,
        numInBatch,
        inPtrs.data(),
        batchSizes.data(),
        encPtrs.data(),
        outBatchSize_dev.data(),
        stream);
    auto end = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    uint64_t total_size = fileSize;
    double c_bw = ( 1.0 * total_size / 1e9 ) / ( comp_time * 1e-3 );  

    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;
    hipStreamSynchronize(stream);

    compressedSize = outBatchSize_dev.copyToHost(stream);
    std::vector<uint8_t> compressedHost(compressedSize[0]);
    hipMemcpy(compressedHost.data(), enc_dev.data(), compressedSize[0], hipMemcpyDeviceToHost);
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    ASSERT_TRUE(outputFile.is_open()) << "Cannot open output file";
    outputFile.write(reinterpret_cast<const char*>(compressedHost.data()), compressedSize[0]);
    outputFile.close();
}

void decompressFloat16File(StackDeviceMemory& res, const std::string& tempFilePath, const std::string& outputFilePath, std::vector<uint32_t>& batchSizes, size_t& numElements, int probBits, uint32_t& maxCompressedSize, int& totalSize, std::vector<uint32_t>& compressedSize, int& numInBatch, hipStream_t stream) {
    std::ifstream inputFile(tempFilePath, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << tempFilePath << std::endl;
        return;
    }
    std::vector<uint8_t> compressedData(compressedSize[0]);
    inputFile.read(reinterpret_cast<char*>(compressedData.data()), compressedSize[0]);
    inputFile.close();

    auto compressedData_dev = res.alloc<uint8_t>(stream, compressedData.size());
    hipMemcpy(compressedData_dev.data(), compressedData.data(), compressedData.size(), hipMemcpyHostToDevice);
    auto encPtrs = std::vector<const void*>(batchSizes.size());
    for (int i = 0; i < encPtrs.size(); ++i) {
        encPtrs[i] = (uint8_t*)compressedData_dev.data() + i * maxCompressedSize;
    }

    // Decode data
    auto dec_dev = res.alloc<FloatTypeInfo<FloatType::kFloat16>::WordT>(stream, totalSize);
    auto decPtrs = std::vector<void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < encPtrs.size(); ++i) {
        decPtrs[i] = (FloatTypeInfo<FloatType::kFloat16>::WordT*)dec_dev.data() + curOffset;
        curOffset += batchSizes[i]*sizeof(FloatTypeInfo<FloatType::kFloat16>::WordT);
      }
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);
    
    double decomp_time = 0.0;
    auto decompConfig =
        FloatDecompressConfig(FloatType::kFloat16, ANSCodecConfig(probBits), false, true);
    auto start = std::chrono::high_resolution_clock::now();
    floatDecompress(
        res,
        decompConfig,
        1,
        (const void**)encPtrs.data(),
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
    std::vector<FloatTypeInfo<FloatType::kFloat16>::WordT> decompressedHost(batchSizes[0]);
    
    hipMemcpy(decompressedHost.data(), dec_dev.data(), batchSizes[0]*sizeof(uint16_t), hipMemcpyDeviceToHost);
    std::ofstream outputFile(outputFilePath, std::ios::binary);
    ASSERT_TRUE(outputFile.is_open()) << "Cannot open output file";
    outputFile.write(reinterpret_cast<const char*>(decompressedHost.data()), decompressedHost.size()*sizeof(uint16_t));
    outputFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.img> <temp.ans> <output.img>" << std::endl;
        return 1;
    }
    auto res = makeStackMemory();
    size_t numElements;
    hipStream_t stream;
    HIP_VERIFY(hipStreamCreate(&stream));
    int probBits = 10; // 根据需要设置精度
    std::vector<uint32_t> batchSizes;
    uint32_t maxCompressedSize;
    std::vector<uint32_t> compressedSize;
    int numInBatch;
    int totalSize;
    try{
    compressFloat16File(res, argv[1], argv[2], batchSizes, numElements, probBits, maxCompressedSize, totalSize, compressedSize, numInBatch,stream);
    decompressFloat16File(res, argv[2], argv[3], batchSizes, numElements, probBits, maxCompressedSize, totalSize, compressedSize, numInBatch,stream);
    hipStreamDestroy(stream);
    std::cout << "Compression completed successfully." << std::endl;
    }catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
