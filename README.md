# hipANS: Portable Lossless Compression of Numerical Data for GPU Architectures

Developer: Jinwu Yang, Yida Gu, Dingwen Tao @ Institute of Computing Technology, Chinese Academy of Sciences

hipANS is a GPU library implemented in the HIP programming language for fast, specialized lossless data compression, designed for ML and HPC applications. It also includes the **first** publicly available generalized [asymmetric numeral system (ANS)](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) compressor and decompressor, compatible with various GPU architectures, including AMD GPUs. It is a GPU analogue to Yann Collet's [FSE (Finite State Entropy)](https://github.com/Cyan4973/FiniteStateEntropy) ANS library. It has been primarily ported and optimized from Facebook's [dietGPU](https://github.com/facebookresearch/dietgpu) library.

It currently consists of two parts:

- **ANS entropy codec**: a generalized byte-oriented range-based ANS (rANS) entropy encoder and decoder, that operates at throughputs around 50-150 GB/s for reasonable data sizes on an MI100/MI210 GPU.
- **Floating point codec**: an extension to the above to handle fast lossless compression and decompression of unstructured floating point data, for use in ML and HPC applications, especially in communicating over local interconnects (PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand). This operates at around 50-300 GB/s for reasonable data sizes on an MI100/MI210 GPU.

Both APIs are available in both C++ (raw device pointers) and Python/PyTorch (PyTorch tensor) API forms.

## Building

Clone this repo using

```shell
git clone https://github.com/hipdac-lab/hipANS
```

Do the standard CMake thing:

```shell
cd hipANS; mkdir build; cd build;
sudo cmake .. -G Ninja
sudo cmake --build . --target all
```

If you get complaints about `/hipANS/third_party/glog... does not contain a CMakeLists.txt file.` then you didn't pull the submodules; run

```shell
git submodule sync
git submodule update --init --recursive --jobs 0
```
and try again.

## Library rationale

As on-device global memory / HBM bandwidth continues to improve at a faster rate than CPU/GPU interconnect or server-to-server networking bandwidth, spending GPU compute and gmem bandwidth to save on data sent over interconnects is becoming more advantageous. HipANS aims to target this gap.

One can imagine a Pareto-optimal tradeoff curve between realizable compression ratios versus speed. On one end of the curve exists algorithms for supporting arbitrary data using dictionary/LZ type compression like some of the techniques in [Nvidia's nvCOMP](https://github.com/NVIDIA/nvcomp) at potentially high compression rates. At another end of the curve, one can imagine use completely on-device as something like a 1990s-style virtual RAM extender, where achievable compression is only 0.6x-0.9x or so, but compression can operate at around 1/4x to 1/2x the peak global memory bandwidth of the GPU. We emphasize the latter, where speed rather than compression ratio is important, where we can compress data that is even sent between GPUs in a single server over NVLink or PCIe. The savings may be low, but the effective network speed can be increased by 10-30%. For large-scale neural network training on hundreds of GPUs, this could translate into an additional 5-10% end-to-end performance increase.

The initial focus of this library will be in HPC/ML distributed collective communications libraries, for primitives such as all-to-all, all-gather, reduce-scatter and all-reduce. Right now no off the shelf integration is provided (in progress), but the basics of the C++ API are available for use, as are Python-level PyTorch tensor-based APIs.

## ANS codec

The rANS codec operates on 8 bit bytes. It can compress arbitrary data, but using statistics gathered on a bytewise basis, so data highly structured or redundant at a level above byte level will typically not compress well. This codec however is meant to be applicable for any number of lossless compression applications, including usage as an entropy coder for LZ or RLE type matches for a fully-formed compression system. Symbol probability precisions supported are 9, 10 and 11 bits (i.e., symbol occurances are quantized to the nearest 1/512, 1/1024 or 1/2048).

## Float codec

The floating point compressor at the moment uses the rANS codec to handle compression of floating point exponents, as typically in ML/HPC data a very limited exponent dynamic range is used and is highly compressible. Floating point sign and significand values tend to be less compressible / fairly high entropy in practice, though sparse data or presence of functions like ReLU in neural networks can result in a lot of outright zero values which are very compressible. A future extension to the library will allow for specialized compression of sparse or semi-sparse data, specializing compression of zeros. At the moment only float16 (IEEE 754 binary16) and bfloat16 (fields of the most significant 16 bits of a IEEE 754 binary32 word) are supported, with float32 (IEEE 754 binary32) support coming shortly.

## API design

The design fundamentals, originally established for the CUDA-based DietGPU library targeting CC 3.5+ (Kepler class) GPUs or later, have been effectively adapted for compatibility with the ROCm platform, specifically focusing on MI100/MI210 GPUs. This strategic porting, now named hipANS, ensures that the library leverages the advanced capabilities of AMD GPUs while retaining the efficiency and performance optimized for V100/A100 GPUs.

The library's APIs have been expertly ported to maintain their batch-oriented approach, available in both C++ and Python/PyTorch forms. This ensures that operations with raw pointers in C++ and tensor operations in PyTorch are both supported, facilitating a seamless transition for developers across platforms. The batch processing capability allows for the compression and decompression of multiple independent data arrays, with the stipulation that when using the floating-point compressor, all arrays in a batch must share the same data type. The ANS compression symbol probabilities are calculated on a per-array basis within the batch, ensuring that each compressed tensor produced is independently decompressible, with ANS statistics customized for each array. 

Despite the batch-oriented design, the library is also optimized for scenarios where a single large batch size is provided, offering commendable performance. It is worth noting that for substantial data sizes, a batch size (bs) of 1 may currently outperform larger batch sizes due to work imbalance issues. The library handles arrays of varying sizes, treating all input data as unstructured one-dimensional arrays, thereby abstracting away the need for PyTorch API to focus on data dimensionality.

The compression process operates on 4 KiB segments of the input data, allocated to individual warps. It is generally recommended to have at least 512 KiB of data to use hipANS effectively, given the overheads associated with compression. For optimal performance, the total data size should be significant enough to match the number of concurrently running warps capable of fully utilizing the GPU's streaming multiprocessors (SMs).

All computations within the library are device-side, with a design emphasis on minimizing memory allocations and deallocations, as well as reducing unnecessary device-to-host and host-to-device interactions and synchronizations. With properly sized inputs and outputs and sufficient pre-allocated temporary memory, compression and decompression can proceed asynchronously on the GPU without the need for CPU intervention.

However, during the compression process, only the GPU is aware of the actual final compressed size. Therefore, it is essential for applications to transfer the output size buffer, detailing the final compressed sizes for each job in the batch, back to the host. This facilitates the management of compressed data, whether for relocation in local memory or transmission over a network. Since the final output size is unpredictable, the library includes a function to estimate the maximum possible compressed output size, aiding in the allocation of appropriately sized memory regions for output. For applications seeking to realize compression savings beyond networking, additional memory allocation and data transfer to a precisely sized new buffer will be necessary.

This transition of the DietGPU library from the CUDA platform to the ROCm platform, resulting in hipANS, marks a significant advancement in providing a versatile and high-performance compression solution for a broader range of HPC applications.

## References

Prior GPU-based ANS implementations [to my knowledge](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations) include:

- [GST: GPU-decodable Supercompressed Textures](https://gamma.cs.unc.edu/GST/) (not a generalized ANS codec; meant as part of a texture compression scheme)
- Weissenberger and Schmidt, [Massively Parallel ANS Decoding on GPUs](https://dl.acm.org/doi/10.1145/3337821.3337888) (a decoder only)

Related GPU entropy coder works include:

- Yamamoto et al., [Huffman Coding with Gap Arrays for GPU Acceleration](https://dl.acm.org/doi/10.1145/3404397.3404429)

Related lossless floating point compression works include:

- Lindstrom and Isenburg, [Fast and Efficient Compression of Floating-Point Data](https://computing.llnl.gov/projects/fpzip) (CPU-based)
- Various GPU works from Martin Burtscher's group at Texas State such as Yang et al., [MPC: A Massively Parallel Compression Algorithm for Scientific Data](https://www.semanticscholar.org/paper/MPC%3A-A-Massively-Parallel-Compression-Algorithm-for-Yang-Mukka/1ab6910c90ad714e29954ccd69d569eb2003eb20)

These works are sometimes oriented at compressing HPC-type data (e.g., 2d/3d/Nd grid data) where there may be local/dimensional correlations that can be exploited.

- [nvCOMP](https://github.com/NVIDIA/nvcomp), Nvidia's GPU lossless compression library.
- [dietGPU](https://github.com/facebookresearch/dietgpu), DietGPU: GPU-based lossless compression for numerical data.

## License

hipANS is licensed with the MIT license, available in the LICENSE file at the top level.
