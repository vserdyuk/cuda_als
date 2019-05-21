#include <memory>


#include "als_model.h"
#include "cuda_runtime.h"
#include "cuda_common.h"

/*
auto cuda_malloc_device = [](size_t size) {
	void *ptr;
	CUDA_CHECK(CUDA_MALLOC_DEVICE(&ptr, size));
	return ptr;
};

auto cuda_deleter_device = [](void *ptr) {
	CUDA_CHECK(cudaFree(ptr));
};
*/

als_model::als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f, float lambda, int iters):
		train_ratings(train_ratings), test_ratings(test_ratings), f(f), lambda(lambda), iters(iters) {
	m = train_ratings.row_cnt;
	n = train_ratings.col_cnt;

	CUDA_CHECK(cudaMallocHost((void **)&h_VT, n * f * sizeof(h_VT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VT, n * f * sizeof(d_VT[0])));

	CUDA_CHECK(cudaMallocHost((void **)&h_U, m * f * sizeof(h_U[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_U, m * f * sizeof(d_U[0])));

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_vvts, f * f * m * sizeof(d_vvts[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_uuts, f * f * n * sizeof(d_uuts[0])));

	// float *d_RVT;	// device ratings multiplied by transposed global item factor matrix, f x m (ythetaT)

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_RVT, f * m * sizeof(d_RVT[0])));

	// float *d_RTUT;	// device transposed ratings multiplied by transposed global user factor matrix (yTXT)

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_RTUT, f * n * sizeof(d_RTUT[0])));

	CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
	CUBLAS_CHECK(cublasCreate_v2(&cublas_handle));
}

als_model::~als_model() {
	CUDA_CHECK(cudaFreeHost(h_VT));
	CUDA_CHECK(cudaFree(d_VT));

	CUDA_CHECK(cudaFreeHost(h_U));
	CUDA_CHECK(cudaFree(d_U));

	CUDA_CHECK(cudaFree(d_vvts));
	CUDA_CHECK(cudaFree(d_uuts));

	CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void als_model::train() {
	//std::unique_ptr<float, decltype(cuda_deleter_device)> thetaT((float*)cuda_malloc_device(f * n * sizeof(float)), cuda_deleter_device);
	//float *a = thetaT.get();

	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < n * f; k++)
		h_VT[k] = 0.2*((float)rand() / (float)RAND_MAX);
	for (int k = 0; k < m * f; k++)
		h_U[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);

	CUDA_CHECK(cudaMemcpy(d_VT, h_VT, n * f * sizeof(h_VT[0]), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_U, h_U, m * f * sizeof(h_U[0]), cudaMemcpyHostToDevice));

	float *d_RV;

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_RV, m * f * sizeof(d_RV[0])));

	const float alpha = 1.0f;
	const float beta = 0.0f;

	// R * VT.T = RV

	CUSPARSE_CHECK(cusparseScsrmm2(
			cusparse_handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_TRANSPOSE,
			m,
			f,
			n,
			train_ratings.val_cnt,
			&alpha,
			train_ratings.cusparse_descr,
			train_ratings.d_csr_coo_vals,
			train_ratings.d_csr_row_ptrs,
			train_ratings.d_csr_coo_col_idxs,
			d_VT,
			f,
			&beta,
			d_RV,
			m
	));

	// RV.T = RVT

	CUBLAS_CHECK(cublasSgeam(
			cublas_handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			f,
			m,
			&alpha,
			d_RV,
			m,
			&beta,
			d_RVT,
			f,
			d_RVT,
			f
	));


	for(size_t i = 0; i < iters; ++i) {
		// TODO
	}

	int i = 0;
}

