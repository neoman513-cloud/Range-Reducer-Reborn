// author: https://t.me/biernus
// Modified: Single random base + thread offset approach
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <chrono>
#pragma once

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert hex string to bytes
__device__ __host__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

// Convert hex string to BigInt - optimized
__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    uint64_t a1, a2, b1, b2;
    uint32_t c1, c2;
    
    memcpy(&a1, hash1, 8);
    memcpy(&a2, hash1 + 8, 8);
    memcpy(&c1, hash1 + 16, 4);

    memcpy(&b1, hash2, 8);
    memcpy(&b2, hash2 + 8, 8);
    memcpy(&c2, hash2 + 16, 4);

    return (a1 == b1) && (a2 == b2) && (c1 == c2);
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

// Device function to generate random BigInt in range [min, max]
__device__ void generate_random_in_range(BigInt* result, curandStateMRG32k3a_t* state, 
                                         const BigInt* min_val, const BigInt* max_val) {
    // Calculate range = max - min
    BigInt range;
    bool borrow = false;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_val->data[i] - (uint64_t)min_val->data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    // Generate random value in [0, range]
    BigInt random;
    for (int w = 0; w < BIGINT_WORDS; w += 4) {
        if (w + 0 < BIGINT_WORDS) random.data[w + 0] = curand(state);
        if (w + 1 < BIGINT_WORDS) random.data[w + 1] = curand(state);
        if (w + 2 < BIGINT_WORDS) random.data[w + 2] = curand(state);
        if (w + 3 < BIGINT_WORDS) random.data[w + 3] = curand(state);
    }
    
    // Reduce random to range
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    if (highest_word >= 0) {
        uint32_t mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        
        random.data[highest_word] &= mask;
        
        for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
            random.data[i] = 0;
        }
        
        bool greater = false;
        for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
            if (random.data[i] > range.data[i]) {
                greater = true;
                break;
            } else if (random.data[i] < range.data[i]) {
                break;
            }
        }
        
        if (greater) {
            for (int i = 0; i < BIGINT_WORDS; ++i) {
                random.data[i] = random.data[i] % (range.data[i] + 1);
            }
        }
    }
    
    // Add min: result = random + min
    bool carry = false;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)random.data[i] + (uint64_t)min_val->data[i] + (carry ? 1 : 0);
        result->data[i] = (uint32_t)sum;
        carry = (sum > 0xFFFFFFFFULL);
    }
}

__device__ __host__ void clear_last_6_hex(BigInt* num) {
    // Last 5 hex digits = 20 bits
    // Clear the least significant 20 bits
    num->data[0] &= 0xFF000000;  // Keep upper 12 bits, clear lower 20 bits
}

// Global device constants for min/max as BigInt
__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;
__constant__ BigInt d_base_key;  // Shared base key for all threads

__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};


__global__ void start(const uint8_t* target)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate this thread's starting private key: base_key + (tid * BATCH_SIZE)
    BigInt priv_base;
    BigInt offset;
    init_bigint(&offset, tid * BATCH_SIZE);
    
    // priv_base = (base_key + tid * BATCH_SIZE) mod n
    ptx_u256Add(&priv_base, &d_base_key, &offset);
    
    // Reduce modulo n if needed
    if (compare_bigint(&priv_base, &const_n) >= 0) {
        ptx_u256Sub(&priv_base, &priv_base, &const_n);
    }
    
    // Early exit: Check if starting key is within [min, max] range
    if (compare_bigint(&priv_base, &d_min_bigint) < 0 || 
        compare_bigint(&priv_base, &d_max_bigint) > 0) {
        return; // Skip if out of range
    }
    
    // Early exit: Check if already found by another thread
    if (g_found) return;
    
    // Array to hold batch of points
    ECPointJac result_jac_batch[BATCH_SIZE];
    uint8_t hash160_batch[BATCH_SIZE][20];
    
    // --- Compute base point: P = priv_base * G ---
    scalar_multiply_multi_base_jac(&result_jac_batch[0], &priv_base);
    
    // --- Generate sequential points: P+G, P+2G, P+3G, ... P+(BATCH_SIZE-1)G ---
    #pragma unroll
    for (int i = 1; i < BATCH_SIZE; ++i) {
        // result_jac_batch[i] = result_jac_batch[i-1] + G
        add_G_to_point_jac(&result_jac_batch[i], &result_jac_batch[i-1]);
    }
    
    // --- Convert the entire batch to hash160s with ONE inverse ---
    jacobian_batch_to_hash160(result_jac_batch, hash160_batch);
    
    // Debug print for first thread only (removed from hot path)
    if (tid == 0) {
        char hash160_str[41];
        char hex_key[65];
        bigint_to_hex(&priv_base, hex_key);
        hash160_to_hex(hash160_batch[0], hash160_str);
        printf("Base key: %s -> %s | scanned completely last 6 hex\n", hex_key, hash160_str);
    }
    
    // --- Optimized batch checking with early exit ---
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
        // Check if another thread already found it
        if (g_found) return;
        
        if (compare_hash160_fast(hash160_batch[i], target)) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                // Calculate the actual private key: priv_base + i
                BigInt priv_found;
                BigInt key_offset;
                init_bigint(&key_offset, i);
                
                // priv_found = (priv_base + i) mod n
                ptx_u256Add(&priv_found, &priv_base, &key_offset);
                
                // Reduce modulo n if needed
                if (compare_bigint(&priv_found, &const_n) >= 0) {
                    ptx_u256Sub(&priv_found, &priv_found, &const_n);
                }
                
                char found_hex[65];
                bigint_to_hex(&priv_found, found_hex);
                hash160_to_hex(hash160_batch[i], g_found_hash160);
                memcpy(g_found_hex, found_hex, 65);
                return;
            }
        }
    }
}

bool run_with_quantum_data(const char* min, const char* max, const char* target, int blocks, int threads, int device_id) {
    uint8_t shared_target[20];
    hex_string_to_bytes(target, shared_target, 20);
    uint8_t *d_target;
    cudaMalloc(&d_target, 20);
    cudaMemcpy(d_target, shared_target, 20, cudaMemcpyHostToDevice);
    
    // Convert min and max hex strings to BigInt and copy to device
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    int total_threads = blocks * threads;
    int found_flag;
    
    // Calculate keys processed per kernel launch
    uint64_t keys_per_kernel = (uint64_t)total_threads * BATCH_SIZE;
    
    printf("Searching in range:\n");
    printf("Min: %s\n", min);
    printf("Max: %s\n", max);
    printf("Target: %s\n", target);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total threads: %d\n", total_threads);
    printf("Keys per kernel: %llu (each thread checks %d sequential keys)\n\n", 
           (unsigned long long)keys_per_kernel, BATCH_SIZE);
    
    // Performance tracking variables
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
	
    uint64_t seed;
	BCryptGenRandom(NULL, (PUCHAR)&seed, sizeof(seed), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    // Pre-calculate range once (outside loop)
    BigInt range;
    bool borrow = false;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_bigint.data[i] - (uint64_t)min_bigint.data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    // Find highest non-zero word and create mask (outside loop)
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    uint32_t mask = 0xFFFFFFFF;
    if (highest_word >= 0) {
        mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
    }
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while(true) {
        // Generate ONE random base key for all threads on HOST
        BigInt base_key;
        BigInt random_val;
        
        // Generate random value in [0, range]
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            random_val.data[i] = dis(gen);
        }
        
        // Apply mask to fit within range
        if (highest_word >= 0) {
            random_val.data[highest_word] &= mask;
            
            for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
                random_val.data[i] = 0;
            }
        }
        
        // base_key = random_val + min
        bool carry = false;
        for (int i = 0; i < BIGINT_WORDS; ++i) {
            uint64_t sum = (uint64_t)random_val.data[i] + (uint64_t)min_bigint.data[i] + (carry ? 1 : 0);
            base_key.data[i] = (uint32_t)sum;
            carry = (sum > 0xFFFFFFFFULL);
        }
        
        clear_last_6_hex(&base_key);
        
        // Copy base key to device constant memory asynchronously
        cudaMemcpyToSymbolAsync(d_base_key, &base_key, sizeof(BigInt), 0, cudaMemcpyHostToDevice, stream);
        
        // Launch kernel with stream
        start<<<blocks, threads, 0, stream>>>(d_target);
        
        // Asynchronous check - only sync when needed
        cudaError_t err = cudaStreamQuery(stream);
        if (err == cudaSuccess) {
            // Update counters
            total_keys_checked += keys_per_kernel;
            
            // Check if found
            cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
            if (found_flag) {
                cudaStreamSynchronize(stream);
                printf("\n\n");
                
                char found_hex[65], found_hash160[41];
                cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
                cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
                
                double total_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - start_time
                ).count();
                
                printf("FOUND!\n");
                printf("Private Key: %s\n", found_hex);
                printf("Hash160: %s\n", found_hash160);
                printf("Total time: %.2f seconds\n", total_time);
                printf("Total keys checked: %llu (%.2f million)\n", 
                       (unsigned long long)total_keys_checked,
                       total_keys_checked / 1000000.0);
                printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
                
                std::ofstream outfile("result.txt", std::ios::app);
                if (outfile.is_open()) {
                    std::time_t now = std::time(nullptr);
                    char timestamp[100];
                    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                    outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                    outfile << "Total keys checked: " << total_keys_checked << std::endl;
                    outfile << "Time taken: " << total_time << " seconds" << std::endl;
                    outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                    outfile << std::endl;
                    outfile.close();
                    std::cout << "Result appended to result.txt" << std::endl;
                }
                
                cudaStreamDestroy(stream);
                cudaFree(d_target);
                return true;
            }
        } else if (err != cudaErrorNotReady) {
            // Handle actual errors
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            cudaStreamSynchronize(stream);
        }
    }
    
    cudaStreamDestroy(stream);
    cudaFree(d_target);
    return false;
}
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target> [device_id]" << std::endl;
        return 1;
    }
    int blocks = 1024;
    int threads = 128;
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    // Set GPU device
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Validate input lengths match
    if (strlen(argv[1]) != strlen(argv[2])) {
        std::cerr << "Error: min and max must have the same length" << std::endl;
        return 1;
    }
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_quantum_data(argv[1], argv[2], argv[3], blocks, threads, device_id);
    
    return 0;
}