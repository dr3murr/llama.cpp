#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Private Qwen3.8 SM60 development path. The device argument is the logical
// CUDA backend index (the same index accepted by ggml_backend_cuda_init).
// Registration owns persistent device copies until explicitly unregistered.
GGML_BACKEND_API bool ggml_backend_cuda_qwen38_pascal_ffn_register(
        int          device,
        int          layer,
        const char * gate_up_path,
        const char * down_path,
        char       * error,
        size_t       error_size);
GGML_BACKEND_API void ggml_backend_cuda_qwen38_pascal_ffn_unregister(int device, int layer);
GGML_BACKEND_API bool ggml_backend_cuda_qwen38_pascal_ffn_is_registered(int device, int layer);

GGML_BACKEND_API bool ggml_backend_cuda_qwen38_pascal_gdn_in_register(
        int          device,
        int          layer,
        const char * qkv_path,
        const char * z_path,
        char       * error,
        size_t       error_size);
GGML_BACKEND_API void ggml_backend_cuda_qwen38_pascal_gdn_in_unregister(int device, int layer);
GGML_BACKEND_API bool ggml_backend_cuda_qwen38_pascal_gdn_in_is_registered(int device, int layer);

// Releases the private Qwen3.8 Pascal resident-FP16 prefill cache.  The
// hybrid server calls this after model execution has stopped and before the
// CUDA backend is shut down so a later process/model cannot inherit entries
// keyed by stale tensor addresses.
GGML_BACKEND_API void ggml_backend_cuda_qwen38_pascal_resident_fp16_clear(void);

// Releases exact packed-weight copies owned by the experimental Qwen3.8
// layer-major prefill executor after one staged layer completes.
GGML_BACKEND_API bool ggml_backend_cuda_qwen38_layer_major_stage_tensor(
        const struct ggml_tensor * src,
        struct ggml_tensor       * proxy,
        int                        device);
GGML_BACKEND_API void ggml_backend_cuda_qwen38_layer_major_stage_clear(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
