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

__global__
void calculate_vvts(float *vvts, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT) {
	// start = current row pointer
	// end = next row pointer
	// nr_of_cols = end - start
	// thread_col = threadIdx.x
	// for i = 0 to f (from up to down)
	//     for j = 0 to nr_of_cols
	//         curr_col_in_global = csr_col_idxs[start + j]
	//         tmp += VT[curr_col_in_global * f + i] * VT[curr_col_in_global * f + i]
	//     vvts[blockIdx.x * f * f + i] = tmp
	//

	// left matrix is vt (i.e. columns from VT for items rated by current user)
	// top matrix is v (i.e. vt transposed)

	/* start = current row pointer
	 * end = next row pointer
	 * nr_of_items = end - start
	 * top_col = out_col = threadIdx.x
	 * for(left_row = out_row = 0; left_row < f; ++left_row ++ out_row)
	 *     temp = 0
	 *     for(curr_item_nr = 0; curr_item_nr < nr_of_items; ++curr_item)
	 *         left_col = top_row = curr_item_nr
	 *         // VT is column-major:
	 *         //     for left matrix, move to next row <=> add 1 to idx
	 *         //     for left matrix, move to next col <=> add f to idx
	 *         //     for top matrix, move to next row <=> add f to idx
	 *         //     for top matrix, move to next col <=> add 1 to idx
	 *
	 *         left_row_offset = left_row
	 *         left_col_offset = csr_col_idxs[start + left_col] * f
	 *
	 *         top_row_offset = csr_col_idxs[start + top_row] * f
	 *         top_col_offset = top_row
	 *
	 *         temp += VT[left_row_offset + left_col_offset] * VT[top_row_offset + top_col_offset]
	 *     vvts[blockIdx.x * f * f + left_row + ] = temp
	 */

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x;

		int start = csr_row_ptrs[user_idx];
		int end = csr_row_ptrs[user_idx + 1];

		int items_cnt = end - start;

		int top_col = threadIdx.x;
		int out_col = top_col;

		int left_row = 0;
		int out_row = left_row;

		while(left_row < f) {
			float out = 0;

			for(int item_nr = 0; item_nr < items_cnt; ++ item_nr) {
				int left_col = item_nr;
				int top_row = left_col;

				// VT is column-major:
				//     for left matrix, move to next row <=> add 1 to idx
				//     for left matrix, move to next col <=> add f to idx
				//     for top matrix, move to next row <=> add f to idx
				//     for top matrix, move to next col <=> add 1 to idx

				int left_row_offset = left_row;
				int left_col_offset = csr_col_idxs[start + left_col] * f;

				int top_row_offset = csr_col_idxs[start + top_row] * f;
				int top_col_offset = top_col;

				out += VT[left_row_offset + left_col_offset] * VT[top_row_offset + top_col_offset];
			}

			// regularization
			if(left_row == top_col) {
				out += items_cnt * lambda;
			}

			vvts[user_idx * f * f + out_row + out_col * f] = out;

			++left_row;
			++out_row;
		}
	}
}

als_model::als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f, float lambda, int iters):
		train_ratings(train_ratings), test_ratings(test_ratings), f(f), lambda(lambda), iters(iters) {
	m = train_ratings.row_cnt;
	n = train_ratings.col_cnt;

	CUDA_CHECK(cudaMallocHost((void **)&h_VT, n * f * sizeof(h_VT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VT, n * f * sizeof(d_VT[0])));

	CUDA_CHECK(cudaMallocHost((void **)&h_U, m * f * sizeof(h_U[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_U, m * f * sizeof(d_U[0])));

	// на первом этапе без X_BATCH размер f * f * m, затем будет f * f * batch_size
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_vvts, f * f * m * sizeof(d_vvts[0])));

	// на первом этапе без THETA_BATCH размер f * f * n, затем будет f * f * batch_size
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_uuts, f * f * n * sizeof(d_uuts[0])));

	// float *d_RVT;	// device ratings multiplied by transposed global item factor matrix, f x m (ythetaT)

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_RVT, f * m * sizeof(d_RVT[0])));

	// float *d_RTUT;	// device transposed ratings multiplied by transposed global user factor matrix f x n (yTXT)

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
	for (size_t k = 0; k < n * f; k++)
		h_VT[k] = 0.2*((float)rand() / (float)RAND_MAX);
	for (size_t k = 0; k < m * f; k++)
		h_U[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);

	CUDA_CHECK(cudaMemcpy(d_VT, h_VT, n * f * sizeof(h_VT[0]), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_U, h_U, m * f * sizeof(h_U[0]), cudaMemcpyHostToDevice));

	for(size_t it = 0; it < iters; ++it) {
		// ---------- update U ----------

		float *d_RV;

		// TODO: single array of max(m, n) * f allocated in model constructor

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

		CUDA_CHECK(cudaFree(d_RV));

		// void calculate_vvts(float *vvts, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT) {
		calculate_vvts<<<m, f>>>(
				d_vvts,
				train_ratings.d_csr_row_ptrs,
				train_ratings.d_csr_coo_col_idxs,
				lambda,
				m,
				f,
				d_VT
		);

#ifdef DEBUG
		CUDA_CHECK(cudaDeviceSynchronize());
#endif
		// TODO: single array of max(m, n) allocated in model constructor

		// host array of pointers to each device vvt
		float **h_d_vvts_ptrs;

		CUDA_CHECK(cudaMallocHost((void **)&h_d_vvts_ptrs, m * sizeof(h_d_vvts_ptrs[0])));

		for(size_t i = 0; i < m; ++i) {
			h_d_vvts_ptrs[i] = &d_vvts[i * f * f];
		}

		// device array of pointers to each device vvt
		float **d_d_vvts_ptrs;

		CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_vvts_ptrs, m * sizeof(d_d_vvts_ptrs[0])));
		CUDA_CHECK(cudaMemcpy(d_d_vvts_ptrs, h_d_vvts_ptrs, m * sizeof(h_d_vvts_ptrs[0]), cudaMemcpyHostToDevice));

		// required by cublasSgetrfBatched but not used for now
		int *d_getrf_infos;

		CUDA_CHECK(CUDA_MALLOC_DEVICE(&d_getrf_infos, m * sizeof(d_getrf_infos[0])));

		// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
		CUBLAS_CHECK(cublasSgetrfBatched(cublas_handle, f, d_d_vvts_ptrs, f, NULL, d_getrf_infos, m));

#ifdef DEBUG
		CUDA_CHECK(cudaDeviceSynchronize());
#endif

		int getrs_info;
		
		// host array of pointers to each device RVT column
		float **h_d_RVT_ptrs;

		CUDA_CHECK(cudaMallocHost((void **)&h_d_RVT_ptrs, m * sizeof(h_d_RVT_ptrs[0])));

		for(size_t i = 0; i < m; ++i) {
			h_d_RVT_ptrs[i] = &d_RVT[i * f];
		}

		// device array of pointers to each device RVT column
		float **d_d_RVT_ptrs;

		CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_RVT_ptrs, m * sizeof(d_d_vvts_ptrs[0])));
		CUDA_CHECK(cudaMemcpy(d_d_RVT_ptrs, h_d_RVT_ptrs, m * sizeof(h_d_RVT_ptrs[0]), cudaMemcpyHostToDevice));

		// d_RVT gets overwritten by result
		// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
		CUBLAS_CHECK(cublasSgetrsBatched(
				cublas_handle,
				CUBLAS_OP_N,
				f,
				1,
				(const float * const *)d_d_vvts_ptrs,
				f,
				nullptr,
				(float * const *)d_d_RVT_ptrs,
				f,
				&getrs_info,
				m
		));

#ifdef DEBUG
		CUDA_CHECK(cudaDeviceSynchronize());
#endif

		// write result
		CUDA_CHECK(cudaMemcpy(d_U, d_RVT, m * f, cudaMemcpyDeviceToDevice));

		CUDA_CHECK(cudaFree(h_d_vvts_ptrs));
		CUDA_CHECK(cudaFree(d_d_vvts_ptrs));
		CUDA_CHECK(cudaFree(d_getrf_infos));
		CUDA_CHECK(cudaFree(h_d_RVT_ptrs));
		CUDA_CHECK(cudaFree(d_d_RVT_ptrs));

		// ---------- update V ----------

		// TODO

		int stop = 0;
	}

}

