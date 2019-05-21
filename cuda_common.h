#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "cuda_runtime.h"

#ifdef DEBUG
#define CUDA_MALLOC_DEVICE cudaMallocManaged
#else
#define CUDA_MALLOC_DEVICE cudaMalloc
#endif


#define CUDA_CHECK(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// https://github.com/rapidsai/cuml/blob/branch-0.8/cpp/src_prims/linalg/cublas_wrappers.h

#define CUBLAS_CHECK(call)                                                     \
  {                                                                            \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                             \
      fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUBLAS_STATUS_NOT_INITIALIZED:                                    \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_NOT_INITIALIZED");            \
          exit(1);                                                             \
        case CUBLAS_STATUS_ALLOC_FAILED:                                       \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_ALLOC_FAILED");               \
          exit(1);                                                             \
        case CUBLAS_STATUS_INVALID_VALUE:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_INVALID_VALUE");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_ARCH_MISMATCH:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_ARCH_MISMATCH");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_MAPPING_ERROR:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_MAPPING_ERROR");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_EXECUTION_FAILED:                                   \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_EXECUTION_FAILED");           \
          exit(1);                                                             \
        case CUBLAS_STATUS_INTERNAL_ERROR:                                     \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_INTERNAL_ERROR");             \
      }                                                                        \
      exit(1);                                                                 \
      exit(1);                                                                 \
    }                                                                          \
  }

// https://github.com/rapidsai/cuml/blob/branch-0.8/cpp/src_prims/linalg/cusparse_wrappers.h

#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                        \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      fprintf(stderr, "Got CUSPARSE error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUSPARSE_STATUS_NOT_INITIALIZED:                                  \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_NOT_INITIALIZED");            \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ALLOC_FAILED:                                     \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ALLOC_FAILED");               \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INVALID_VALUE:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INVALID_VALUE");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ARCH_MISMATCH:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ARCH_MISMATCH");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_MAPPING_ERROR:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_MAPPING_ERROR");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_EXECUTION_FAILED:                                 \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_EXECUTION_FAILED");           \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INTERNAL_ERROR:                                   \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INTERNAL_ERROR");             \
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                                   \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");             \
      }                                                                        \
      exit(1);                                                                 \
      exit(1);                                                                 \
    }                                                                          \
  }

#endif // CUDA_COMMON_H
