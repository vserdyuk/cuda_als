#include <memory>


#include "als_model.h"
#include "cuda_runtime.h"
#include "cuda_common.h"
#include "logger.h"

__global__
void calculate_vtvs(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT) {
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
	 *     vtvs[blockIdx.x * f * f + left_row + ] = temp
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

			vtvs[user_idx * f * f + out_row + out_col * f] = out;

			++left_row;
			++out_row;
		}
	}
}

__global__
void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x;

		int start = csr_row_ptrs[user_idx];
		int end = csr_row_ptrs[user_idx + 1];

		int items_cnt = end - start;

		int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;


		int top_col = threadIdx.x;
		int out_col = top_col;

		for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
			int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

			// Each thread reads smem_col_cnt columns - all threads are busy
			for(int smem_col = 0; smem_col < smem_items; ++smem_col) {
				smem[f * smem_col + threadIdx.x] = VT[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x];
			}

			__syncthreads();

			// actual work

			int left_row = 0;
			int out_row = left_row;

			while(left_row < f) {
				float out = 0;

				for(int item_nr = 0; item_nr < smem_items; ++item_nr) {
					// VT is column-major:
					//     for left matrix, move to next row <=> add 1 to idx
					//     for left matrix, move to next col <=> add f to idx
					//     for top matrix, move to next row <=> add f to idx
					//     for top matrix, move to next col <=> add 1 to idx

					int left_col = item_nr;
					int top_row = left_col;

					int left_row_offset = left_row;
					int left_col_offset = left_col * f;

					int top_row_offset = top_row * f;
					int top_col_offset = top_col;

					out += smem[left_row_offset + left_col_offset] * smem[top_row_offset + top_col_offset];
				}

				vtvs[user_idx * f * f + out_row + out_col * f] += out;

				++left_row;
				++out_row;
			}
		}

		// regularization
		vtvs[user_idx * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

__global__
void calculate_vtvs_smem_row_major_no_calc(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x;

		int start = csr_row_ptrs[user_idx];
		int end = csr_row_ptrs[user_idx + 1];

		int items_cnt = end - start;

		int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;


		int top_col = threadIdx.x;
		int out_col = top_col;

		for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
			int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

			// Each thread reads smem_col_cnt columns - all threads are busy
			for(int smem_col = 0; smem_col < smem_items; ++smem_col) {
				smem[f * smem_col + threadIdx.x] = VT[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x];
			}

			__syncthreads();

			// no actual work - just measuring smem loading time
		}
	}
}

__global__
void calculate_vtvs_smem_col_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x;

		int start = csr_row_ptrs[user_idx];
		int end = csr_row_ptrs[user_idx + 1];

		int items_cnt = end - start;

		int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;


		int top_col = threadIdx.x;
		int out_col = top_col;

		for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
			int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

			// First smem_items threads read one column each, other threads are busy
			// Should be enough threads to read smem_items columns
			if(threadIdx.x < smem_items) {
				int global_col = csr_col_idxs[start + smem_iter * smem_col_cnt + threadIdx.x];
				for(int row = 0; row < f; ++row) {
					smem[f * threadIdx.x + row] = VT[f * global_col + row];
				}
			}

			__syncthreads();

			// actual work

			int left_row = 0;
			int out_row = left_row;

			while(left_row < f) {
				float out = 0;

				for(int item_nr = 0; item_nr < smem_items; ++item_nr) {
					// VT is column-major:
					//     for left matrix, move to next row <=> add 1 to idx
					//     for left matrix, move to next col <=> add f to idx
					//     for top matrix, move to next row <=> add f to idx
					//     for top matrix, move to next col <=> add 1 to idx

					int left_col = item_nr;
					int top_row = left_col;

					int left_row_offset = left_row;
					int left_col_offset = left_col * f;

					int top_row_offset = top_row * f;
					int top_col_offset = top_col;

					out += smem[left_row_offset + left_col_offset] * smem[top_row_offset + top_col_offset];
				}

				vtvs[user_idx * f * f + out_row + out_col * f] += out;

				++left_row;
				++out_row;
			}
		}

		// regularization
		vtvs[user_idx * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

__global__
void calculate_vtvs_smem_col_major_two_threads(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x;

		int start = csr_row_ptrs[user_idx];
		int end = csr_row_ptrs[user_idx + 1];

		int items_cnt = end - start;

		int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;


		int top_col = threadIdx.x;
		int out_col = top_col;

		for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {

			int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

			// First smem_items * 2 threads read one half column each, other threads are busy.
			// Should be enough threads to read smem_items columns
			if(threadIdx.x < smem_items * 2) {
				int global_col = csr_col_idxs[start + smem_iter * smem_col_cnt + threadIdx.x];
				int first_row = f / 2 * threadIdx.x % 2;
				for(int row = first_row; row < f / 2; ++row) {
					smem[f * threadIdx.x / 2 + row] = VT[f * global_col + row];
				}
			}

			__syncthreads();

			// actual work

			int left_row = 0;
			int out_row = left_row;

			while(left_row < f) {
				float out = 0;

				for(int item_nr = 0; item_nr < smem_items; ++item_nr) {
					// VT is column-major:
					//     for left matrix, move to next row <=> add 1 to idx
					//     for left matrix, move to next col <=> add f to idx
					//     for top matrix, move to next row <=> add f to idx
					//     for top matrix, move to next col <=> add 1 to idx

					int left_col = item_nr;
					int top_row = left_col;

					int left_row_offset = left_row;
					int left_col_offset = left_col * f;

					int top_row_offset = top_row * f;
					int top_col_offset = top_col;

					out += smem[left_row_offset + left_col_offset] * smem[top_row_offset + top_col_offset];
				}

				vtvs[user_idx * f * f + out_row + out_col * f] += out;	// how bad is += for performance?

				++left_row;
				++out_row;
			}
		}

		// regularization
		vtvs[user_idx * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

std::string als_model::to_string(CALCULATE_VVTS_TYPE calculate_vvts_type) {
	switch(calculate_vvts_type) {
		case CALCULATE_VVTS_TYPE::SIMPLE: return "SIMPLE";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR: return "SMEM_ROW_MAJOR";
		case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR: return "SMEM_COL_MAJOR";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC: return "SMEM_ROW_MAJOR_NO_CALC";
		case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS: return "SMEM_COL_MAJOR_TWO_THREADS";
		default: return "UNKNOWN";
	}
}

als_model::als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f, float lambda, int iters, CALCULATE_VVTS_TYPE calculate_vvts_type, int smem_col_cnt):
		train_ratings(train_ratings), test_ratings(test_ratings), f(f), lambda(lambda), iters(iters), calculate_vvts_type(calculate_vvts_type),
		smem_col_cnt(smem_col_cnt) {
	m = train_ratings.row_cnt;
	n = train_ratings.col_cnt;

	CUDA_CHECK(cudaMallocHost((void **)&h_VT, n * f * sizeof(h_VT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VT, n * f * sizeof(d_VT[0])));

	CUDA_CHECK(cudaMallocHost((void **)&h_UT, m * f * sizeof(h_UT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_UT, m * f * sizeof(d_UT[0])));

	// на первом этапе без X_BATCH размер f * f * m, затем будет f * f * batch_size
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_vtvs, f * f * m * sizeof(d_vtvs[0])));

	// на первом этапе без THETA_BATCH размер f * f * n, затем будет f * f * batch_size
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_utus, f * f * n * sizeof(d_utus[0])));

	// float *d_VTRT;	// device transposed global item factor matrix multiplied by transposed ratings, f x m (confusing name ythetaT, IMHO thetaTyT is clearer)

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VTRT, f * m * sizeof(d_VTRT[0])));

	// float *d_UTR;	// device transposed global user factor matrix multiplied by ratings, f x n (confusing name yTXT, IMHO XTy is clearer)

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_UTR, f * n * sizeof(d_UTR[0])));

	CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
	CUBLAS_CHECK(cublasCreate_v2(&cublas_handle));
}

als_model::~als_model() {
	CUDA_CHECK(cudaFreeHost(h_VT));
	CUDA_CHECK(cudaFree(d_VT));

	CUDA_CHECK(cudaFreeHost(h_UT));
	CUDA_CHECK(cudaFree(d_UT));

	CUDA_CHECK(cudaFree(d_vtvs));
	CUDA_CHECK(cudaFree(d_utus));

	CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void als_model::train() {

#ifdef USE_LOGGER
	g_logger.log("als model training started", true);
#endif

	unsigned int seed = 0;
	srand (seed);
	for (size_t k = 0; k < n * f; k++)
		h_VT[k] = 0.2*((float)rand() / (float)RAND_MAX);
	for (size_t k = 0; k < m * f; k++)
		h_UT[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);

	CUDA_CHECK(cudaMemcpy(d_VT, h_VT, n * f * sizeof(h_VT[0]), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_UT, h_UT, m * f * sizeof(h_UT[0]), cudaMemcpyHostToDevice));

#ifdef USE_LOGGER
	g_logger.log("factors initialization done", true);
#endif

	for(size_t it = 0; it < iters; ++it) {

#ifdef USE_LOGGER
		g_logger.als_iter = it + 1;
#endif

		// ---------- update U ----------
		{

#ifdef USE_LOGGER
			g_logger.log("update U started", true);
#endif

			// device ratings multiplied by global item factor matrix, m x f (ytheta)
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

			// (RV).T = VTRT

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
					d_VTRT,
					f,
					d_VTRT,
					f
			));

			CUDA_CHECK(cudaFree(d_RV));

#ifdef USE_LOGGER
			g_logger.log("VTRT via cuSPARSE and cuBLAS done", true);
#endif

#ifdef USE_LOGGER
			g_logger.log("vtvs calculation started type=" + to_string(calculate_vvts_type), true);
#endif

			int smem_size = 0;

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
				smem_size = smem_col_cnt * f * sizeof(d_VT[0]);
				g_logger.log("vtvs smem_col_cnt=" + std::to_string(smem_col_cnt) + " smem_size=" + std::to_string(smem_size), true);
			}

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_row_major<<<m, f, smem_size>>>(d_vtvs, train_ratings.d_csr_row_ptrs,
						train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_row_major_no_calc<<<m, f, smem_size>>>(d_vtvs, train_ratings.d_csr_row_ptrs,
						train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
				if(f < smem_col_cnt) {
					throw std::runtime_error("SMEM_COL_MAJOR: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt("
							+ std::to_string(smem_col_cnt) + ")"
					);
				}
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_col_major<<<m, f, smem_size>>>(d_vtvs, train_ratings.d_csr_row_ptrs,
						train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
				if(f < smem_col_cnt * 2) {
					throw std::runtime_error("SMEM_COL_MAJOR_TWO_THREADS: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt * 2 ("
							+ std::to_string(smem_col_cnt) + " * 2 = " + std::to_string(smem_col_cnt * 2) + ")"
					);
				}
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_col_major_two_threads<<<m, f, smem_size>>>(d_vtvs, train_ratings.d_csr_row_ptrs,
						train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SIMPLE:
			default:
				// void calculate_vtvs(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT) {
				calculate_vtvs<<<m, f>>>(d_vtvs, train_ratings.d_csr_row_ptrs,
						train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT
				);
			}
			CUDA_CHECK(cudaPeekAtLastError());

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("vtvs calculation done type=" + to_string(calculate_vvts_type), true);
#endif

			// TODO: single array of max(m, n) allocated in model constructor

			// host array of pointers to each device vtv
			float **h_d_vtvs_ptrs;

			CUDA_CHECK(cudaMallocHost((void **)&h_d_vtvs_ptrs, m * sizeof(h_d_vtvs_ptrs[0])));

			for(size_t i = 0; i < m; ++i) {
				h_d_vtvs_ptrs[i] = &d_vtvs[i * f * f];
			}

			// device array of pointers to each device vtv
			float **d_d_vtvs_ptrs;

			CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_vtvs_ptrs, m * sizeof(d_d_vtvs_ptrs[0])));
			CUDA_CHECK(cudaMemcpy(d_d_vtvs_ptrs, h_d_vtvs_ptrs, m * sizeof(h_d_vtvs_ptrs[0]), cudaMemcpyHostToDevice));

			// required by cublasSgetrfBatched but not used for now
			int *d_getrf_infos;

			CUDA_CHECK(CUDA_MALLOC_DEVICE(&d_getrf_infos, m * sizeof(d_getrf_infos[0])));

			// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
			CUBLAS_CHECK(cublasSgetrfBatched(cublas_handle, f, d_d_vtvs_ptrs, f, NULL, d_getrf_infos, m));

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("vtvs batched LU factorization done", true);
#endif

			int getrs_info;

			// host array of pointers to each device VTRT column
			float **h_d_VTRT_ptrs;

			CUDA_CHECK(cudaMallocHost((void **)&h_d_VTRT_ptrs, m * sizeof(h_d_VTRT_ptrs[0])));

			for(size_t i = 0; i < m; ++i) {
				h_d_VTRT_ptrs[i] = &d_VTRT[i * f];
			}

			// device array of pointers to each device VTRT column
			float **d_d_VTRT_ptrs;

			CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_VTRT_ptrs, m * sizeof(d_d_VTRT_ptrs[0])));
			CUDA_CHECK(cudaMemcpy(d_d_VTRT_ptrs, h_d_VTRT_ptrs, m * sizeof(h_d_VTRT_ptrs[0]), cudaMemcpyHostToDevice));

			// d_VTRT gets overwritten by result (VT)
			// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
			CUBLAS_CHECK(cublasSgetrsBatched(
					cublas_handle,
					CUBLAS_OP_N,
					f,
					1,
					(const float * const *)d_d_vtvs_ptrs,
					f,
					nullptr,
					(float * const *)d_d_VTRT_ptrs,
					f,
					&getrs_info,
					m
			));

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("U batched solve done", true);
#endif

			// write result
			CUDA_CHECK(cudaMemcpy(d_UT, d_VTRT, m * f * sizeof(d_VTRT[0]), cudaMemcpyDeviceToDevice));

			CUDA_CHECK(cudaFreeHost(h_d_vtvs_ptrs));
			CUDA_CHECK(cudaFree(d_d_vtvs_ptrs));
			CUDA_CHECK(cudaFree(d_getrf_infos));
			CUDA_CHECK(cudaFreeHost(h_d_VTRT_ptrs));
			CUDA_CHECK(cudaFree(d_d_VTRT_ptrs));

#ifdef USE_LOGGER
			g_logger.log("update U done", true);
#endif

		}	// update U block
		// ---------- update V ----------
		{

#ifdef USE_LOGGER
			g_logger.log("update V started", true);
#endif

			// device transposed ratings multiplied by global user factor matrix, m x f (yTX)
			float *d_RTU;

			// TODO: single array of max(m, n) * f allocated in model constructor

			CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_RTU, n * f * sizeof(d_RTU[0])));

			const float alpha = 1.0f;
			const float beta = 0.0f;

			// RT * UT.T = RTU

			// https://docs.nvidia.com/cuda/cusparse/index.html#csc-format
			// Note: The matrix A in CSR format has exactly the same memory layout as its transpose in CSC format (and vice versa).
			CUSPARSE_CHECK(cusparseScsrmm2(
					cusparse_handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					CUSPARSE_OPERATION_TRANSPOSE,
					n,
					f,
					m,
					train_ratings.val_cnt,
					&alpha,
					train_ratings.cusparse_descr,
					train_ratings.d_csc_vals,
					train_ratings.d_csc_col_ptrs,
					train_ratings.d_csc_row_idxs,
					d_UT,
					f,
					&beta,
					d_RTU,
					n
			));

			// (RTU).T = UTR

			CUBLAS_CHECK(cublasSgeam(
					cublas_handle,
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					f,
					n,
					&alpha,
					d_RTU,
					n,
					&beta,
					d_UTR,
					f,
					d_UTR,
					f
			));

			CUDA_CHECK(cudaFree(d_RTU));

#ifdef USE_LOGGER
			g_logger.log("d_UTR via cuSPARSE and cuBLAS done", true);
#endif

#ifdef USE_LOGGER
			g_logger.log("utus calculation started type=" + to_string(calculate_vvts_type), true);
#endif

			// Function is named calculate_vtvs but here we actually calculate utus.
			// Naming is kept for U update for easier debugging

			int smem_size = 0;

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
				smem_size = smem_col_cnt * f * sizeof(d_UT[0]);
				g_logger.log("utus smem_col_cnt=" + std::to_string(smem_col_cnt) + " smem_size=" + std::to_string(smem_size), true);
			}

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_row_major<<<n, f, smem_size>>>(d_utus, train_ratings.d_csc_col_ptrs,
						train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_row_major_no_calc<<<n, f, smem_size>>>(d_utus, train_ratings.d_csc_col_ptrs,
						train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
				if(f < smem_col_cnt) {
					throw std::runtime_error("SMEM_COL_MAJOR: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt("
							+ std::to_string(smem_col_cnt) + ")"
					);
				}
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_col_major<<<n, f, smem_size>>>(d_utus, train_ratings.d_csc_col_ptrs,
						train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
				if(f < smem_col_cnt * 2) {
					throw std::runtime_error("SMEM_COL_MAJOR_TWO_THREADS: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt * 2 ("
							+ std::to_string(smem_col_cnt) + " * 2 = " + std::to_string(smem_col_cnt * 2) + ")"
					);
				}
				// void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt) {
				calculate_vtvs_smem_col_major_two_threads<<<n, f, smem_size>>>(d_utus, train_ratings.d_csc_col_ptrs,
						train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt
				);
				break;
			case CALCULATE_VVTS_TYPE::SIMPLE:
			default:
				// void calculate_vtvs(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT) {
				calculate_vtvs<<<n, f>>>(d_utus, train_ratings.d_csc_col_ptrs,
						train_ratings.d_csc_row_idxs, lambda, n, f, d_UT
				);
			}

			CUDA_CHECK(cudaPeekAtLastError());

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("utus calculation done", true);
#endif

			// TODO: single array of max(m, n) allocated in model constructor

			// host array of pointers to each device utu
			float **h_d_utus_ptrs;

			CUDA_CHECK(cudaMallocHost((void **)&h_d_utus_ptrs, n * sizeof(h_d_utus_ptrs[0])));

			for(size_t i = 0; i < n; ++i) {
				h_d_utus_ptrs[i] = &d_utus[i * f * f];
			}

			// device array of pointers to each device utu
			float **d_d_utus_ptrs;

			CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_utus_ptrs, n * sizeof(d_d_utus_ptrs[0])));
			CUDA_CHECK(cudaMemcpy(d_d_utus_ptrs, h_d_utus_ptrs, n * sizeof(h_d_utus_ptrs[0]), cudaMemcpyHostToDevice));

			// required by cublasSgetrfBatched but not used for now
			int *d_getrf_infos;

			CUDA_CHECK(CUDA_MALLOC_DEVICE(&d_getrf_infos, n * sizeof(d_getrf_infos[0])));

			// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
			CUBLAS_CHECK(cublasSgetrfBatched(cublas_handle, f, d_d_utus_ptrs, f, NULL, d_getrf_infos, n));

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("utus batched LU factorization done type=" + to_string(calculate_vvts_type), true);
#endif

			int getrs_info;

			// host array of pointers to each device UTR column
			float **h_d_UTR_ptrs;

			CUDA_CHECK(cudaMallocHost((void **)&h_d_UTR_ptrs, n * sizeof(h_d_UTR_ptrs[0])));

			for(size_t i = 0; i < n; ++i) {
				h_d_UTR_ptrs[i] = &d_UTR[i * f];
			}

			// device array of pointers to each device UTR column
			float **d_d_UTR_ptrs;

			CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_UTR_ptrs, n * sizeof(d_d_utus_ptrs[0])));
			CUDA_CHECK(cudaMemcpy(d_d_UTR_ptrs, h_d_UTR_ptrs, n * sizeof(h_d_UTR_ptrs[0]), cudaMemcpyHostToDevice));

			// d_UTR gets overwritten by result (UT)
			// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
			CUBLAS_CHECK(cublasSgetrsBatched(
					cublas_handle,
					CUBLAS_OP_N,
					f,
					1,
					(const float * const *)d_d_utus_ptrs,
					f,
					nullptr,
					(float * const *)d_d_UTR_ptrs,
					f,
					&getrs_info,
					n
			));

#if defined (DEBUG) || defined(USE_LOGGER)
			CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
			g_logger.log("V batched solve done", true);
#endif

			// write result
			CUDA_CHECK(cudaMemcpy(d_VT, d_UTR, n * f * sizeof(d_UTR[0]), cudaMemcpyDeviceToDevice));

			CUDA_CHECK(cudaFreeHost(h_d_utus_ptrs));
			CUDA_CHECK(cudaFree(d_d_utus_ptrs));
			CUDA_CHECK(cudaFree(d_getrf_infos));
			CUDA_CHECK(cudaFreeHost(h_d_UTR_ptrs));
			CUDA_CHECK(cudaFree(d_d_UTR_ptrs));

#ifdef USE_LOGGER
			g_logger.log("update V done", true);
#endif

		}	// update V block
	}	// iters loop

#ifdef USE_LOGGER
	g_logger.als_iter = 0;
#endif

	// final result from device to host

	CUDA_CHECK(cudaMemcpy(h_VT, d_VT, m * f, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_UT, d_UT, n * f, cudaMemcpyDeviceToHost));

#ifdef USE_LOGGER
	g_logger.log("als model training done", true);
#endif
}

