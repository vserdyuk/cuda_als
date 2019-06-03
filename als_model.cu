#include <memory>

#include <mma.h>
using namespace nvcuda;

#include "als_model.h"
#include "cuda_runtime.h"
#include "cuda_common.h"

#ifdef USE_LOGGER
#include "logger.h"
#endif

#include <fstream>
#include <limits>
#include <iostream>

#define FIND_NAN
//#define DEBUG_SAVE
#define CALC_RSME

#ifdef DEBUG_SAVE
static void save_host_array_float(const float *h_arr, const size_t size, const std::string &path) {
	std::ofstream out;

	out.open(path);

	out << std::fixed;

	out.precision(std::numeric_limits<double>::max_digits10);

	for(size_t i = 0; i < size; ++i) {
		out << h_arr[i] << std::endl;
	}
}

static void save_device_array_float(const float *d_arr, const size_t size, const std::string &path) {
	float *h_arr;

	CUDA_CHECK(cudaMallocHost((void **)&h_arr, size * sizeof(h_arr[0])));

	CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(d_arr[0]), cudaMemcpyDeviceToHost));

	save_host_array_float(h_arr, size, path);

	CUDA_CHECK(cudaFreeHost(h_arr));
}
#endif	// DEBUG_SAVE

__global__
void float2half_array(float *float_arr, half *half_arr, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		half_arr[i] =  __float2half(float_arr[i]);
	}
}

__global__
void calculate_square_error(float *coo_vals, int *coo_row_idxs, int *coo_col_idxs, int val_cnt, float *UT, float *VT, int f, float *err_arr, int err_size) {
	int val_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (val_id < val_cnt) {
		int row = coo_row_idxs[val_id];
		int col = coo_col_idxs[val_id];

		float actual = coo_vals[val_id];

		float predicted = 0;

		for(int i = 0; i < f; ++i) {
			float u = UT[f * row + i];
			float v = VT[f * col + i];

			if(u != u || v != v) {
				break;
			} else {
				predicted += u * v;
			}
		}

		float err = actual - predicted;

		atomicAdd(&err_arr[val_id % err_size], err * err);
	}
}

__global__
void calculate_vtvs(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int m_batch_offset) {
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
		int user_idx = blockIdx.x + m_batch_offset;

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

			vtvs[blockIdx.x * f * f + out_row + out_col * f] = out;

			++left_row;
			++out_row;
		}
	}
}

__global__
void calculate_vtvs_smem_row_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x + m_batch_offset;

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

#ifdef FIND_NAN
				if(smem[f * smem_col + threadIdx.x] != smem[f * smem_col + threadIdx.x]) {
					printf("smem nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
#endif
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

				vtvs[blockIdx.x * f * f + out_row + out_col * f] += out;

				++left_row;
				++out_row;
			}

			__syncthreads();
		}

		// regularization
		vtvs[blockIdx.x * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

__global__
void calculate_vtvs_smem_row_major_no_calc(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x + m_batch_offset;

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
void calculate_vtvs_smem_col_major(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x + m_batch_offset;

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

				vtvs[blockIdx.x * f * f + out_row + out_col * f] += out;

				++left_row;
				++out_row;
			}

			__syncthreads();
		}

		// regularization
		vtvs[blockIdx.x * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

__global__
void calculate_vtvs_smem_col_major_two_threads(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, float *VT, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ float smem [];

	if(threadIdx.x < f) {
		int user_idx = blockIdx.x + m_batch_offset;

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

				vtvs[blockIdx.x * f * f + out_row + out_col * f] += out;	// how bad is += for performance?

				++left_row;
				++out_row;
			}

			__syncthreads();
		}

		// regularization
		vtvs[blockIdx.x * f * f + out_col + out_col * f] += items_cnt * lambda;
	}
}

// add float to half converter

// multiple warps, each computes one piece
__global__
void calculate_vtvs_smem_row_major_tensor(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, half *VT_half, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ half smem_half [];

	const unsigned int warp_id = threadIdx.x / 32;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> vt_frag;	// actually col-major
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;	// vt interpreted as row-major -> vt transposed -> v
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

	wmma::fill_fragment(acc_frag, 0.0f);

	int user_idx = blockIdx.x + m_batch_offset;

	int start = csr_row_ptrs[user_idx];
	int end = csr_row_ptrs[user_idx + 1];

	int items_cnt = end - start;

	int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;

	int tile_row = warp_id / (f / 16);	// 4 = f(64)/wmma rows(16)
	int tile_col = warp_id % (f / 16);	// 4 = f(64)/wmma cols(16)

	for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
		int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

		// First 64(f) threads read smem_col_cnt columns, others are idle
		if(threadIdx.x < f) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + threadIdx.x] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x];
				++smem_col;

#ifdef FIND_NAN
				if(smem_half[f * smem_col + threadIdx.x] != smem_half[f * smem_col + threadIdx.x]) {
					printf("smem_half nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
#endif
			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + threadIdx.x], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		__syncthreads();

		// actual work

		for(int tile_iter = 0; tile_iter < smem_col_cnt / 16; ++tile_iter) {
			wmma::load_matrix_sync(vt_frag, smem_half + tile_row * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + tile_col * 16 + tile_iter * f * 16, f);

#ifdef FIND_NAN
			for(size_t i = 0; i < vt_frag.num_elements; ++i) {
				if(vt_frag.x[i] != vt_frag.x[i]) {
					printf("vt_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
			for(size_t i = 0; i < v_frag.num_elements; ++i) {
				if(v_frag.x[i] != v_frag.x[i]) {
					printf("v_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
#endif

			wmma::mma_sync(acc_frag, vt_frag, v_frag, acc_frag);

#ifdef FIND_NAN
			for(size_t i = 0; i < acc_frag.num_elements; ++i) {
				if(acc_frag.x[i] != acc_frag.x[i]) {
					printf("acc_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
#endif
		}

		__syncthreads();
	}

	// regularization
	float *smem_half_reg_term_buf = (float *)(smem_half + smem_col_cnt * f);


	if(threadIdx.x < 16 * 16) {
		smem_half_reg_term_buf[threadIdx.x] = 0;
	}

	__syncthreads();

	if(threadIdx.x < 16) {
		smem_half_reg_term_buf[threadIdx.x * 16 + threadIdx.x] = items_cnt * lambda;
	}

	__syncthreads(); // ?

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> reg_frag;

	if(tile_row == tile_col) {
		wmma::load_matrix_sync(reg_frag, smem_half_reg_term_buf, 16, wmma::mem_row_major);
		for(int i = 0; i < reg_frag.num_elements; ++i) {
			acc_frag.x[i] += reg_frag.x[i];
		}
	}

	// store

	// vtvs[user_idx * f * f + out_row + out_col * f] += out;
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_row * 16 * f + tile_col * 16, acc_frag, f, wmma::mem_row_major);
}

__global__
void calculate_vtvs_smem_row_major_tensor_symmetric(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, half *VT_half, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ half smem_half [];

	const unsigned int warp_id = threadIdx.x / 32;
	//const unsigned int lane_id = threadIdx.x % 32;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> vt_frag;	// actually col-major
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;	// vt interpreted as row-major -> vt transposed -> v
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

	wmma::fill_fragment(acc_frag, 0.0f);

	int user_idx = blockIdx.x + m_batch_offset;

	int start = csr_row_ptrs[user_idx];
	int end = csr_row_ptrs[user_idx + 1];

	int items_cnt = end - start;

	int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;

	//int tile_row = warp_id / (f / 16);	// 4 = f(64)/wmma rows(16)
	//int tile_col = warp_id % (f / 16);	// 4 = f(64)/wmma cols(16)

	int tile_row = 0;
	int tile_col = 0;

	int tile_dim = f / 16;	// tile_dim * tile_dim square

	for(int col = 0; col < tile_dim; ++col) {
		int next_col_start = (2 * tile_dim - col) * (col + 1) / 2;
		if(warp_id < next_col_start) {
			tile_row = tile_dim + warp_id - next_col_start;
			tile_col = col;
			break;
		}
	}

	for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
		int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

		// First 64(f) threads read smem_col_cnt columns, others are idle
		if(threadIdx.x < f) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + threadIdx.x] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x];
				++smem_col;

#ifdef FIND_NAN
				if(smem_half[f * smem_col + threadIdx.x] != smem_half[f * smem_col + threadIdx.x]) {
					printf("smem_half nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
#endif

			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + threadIdx.x], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		__syncthreads();

		// actual work

		for(int tile_iter = 0; tile_iter < smem_col_cnt / 16; ++tile_iter) {
			wmma::load_matrix_sync(vt_frag, smem_half + tile_row * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + tile_col * 16 + tile_iter * f * 16, f);

#ifdef FIND_NAN
			for(size_t i = 0; i < vt_frag.num_elements; ++i) {
				if(vt_frag.x[i] != vt_frag.x[i]) {
					printf("vt_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
			for(size_t i = 0; i < v_frag.num_elements; ++i) {
				if(v_frag.x[i] != v_frag.x[i]) {
					printf("v_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
#endif

			wmma::mma_sync(acc_frag, vt_frag, v_frag, acc_frag);

#ifdef FIND_NAN
			for(size_t i = 0; i < acc_frag.num_elements; ++i) {
				if(acc_frag.x[i] != acc_frag.x[i]) {
					printf("acc_frag nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
			}
#endif

		}

		__syncthreads();
	}

	// regularization

	float *smem_half_reg_term_buf = (float *)(smem_half + smem_col_cnt * f);


	if(threadIdx.x < 16 * 16) {
		smem_half_reg_term_buf[threadIdx.x] = 0;
	}

	__syncthreads();

	if(threadIdx.x < 16) {
		smem_half_reg_term_buf[threadIdx.x * 16 + threadIdx.x] = items_cnt * lambda;
	}

	__syncthreads(); // ?

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> reg_frag;

	if(tile_row == tile_col) {
		wmma::load_matrix_sync(reg_frag, smem_half_reg_term_buf, 16, wmma::mem_row_major);
		for(int i = 0; i < reg_frag.num_elements; ++i) {
			acc_frag.x[i] += reg_frag.x[i];
		}
	}


	// store

	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_col * 16 * f + tile_row * 16, acc_frag, f, wmma::mem_col_major);

	if(tile_row != tile_col) {
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_row * 16 * f + tile_col * 16, acc_frag, f, wmma::mem_row_major);
	}
}

// does not work, too many registers for one warp
// need to split into two warps

__global__
void calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f96(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, half *VT_half, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ half smem_half [];

	//const unsigned int warp_id = threadIdx.x / 32;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> vt_frag;	// actually col-major
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;	// vt interpreted as row-major -> vt transposed -> v
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag0_0;		// tile_row = 0, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_0;		// tile_row = 1, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_0;		// tile_row = 2, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_0;		// tile_row = 3, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_0;		// tile_row = 4, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_0;		// tile_row = 5, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_1;		// tile_row = 1, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_1;		// tile_row = 2, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_1;		// tile_row = 3, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_1;		// tile_row = 4, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_1;		// tile_row = 5, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_2;		// tile_row = 2, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_2;		// tile_row = 3, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_2;		// tile_row = 4, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_2;		// tile_row = 5, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_3;		// tile_row = 3, tile_col = 3
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_3;		// tile_row = 4, tile_col = 3
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_3;		// tile_row = 5, tile_col = 3
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_4;		// tile_row = 4, tile_col = 4
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_4;		// tile_row = 4, tile_col = 5
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_5;		// tile_row = 5, tile_col = 5


	//wmma::fill_fragment(acc_frag, 0.0f);
	wmma::fill_fragment(acc_frag0_0, 0.0f);
	wmma::fill_fragment(acc_frag1_0, 0.0f);
	wmma::fill_fragment(acc_frag2_0, 0.0f);
	wmma::fill_fragment(acc_frag3_0, 0.0f);
	wmma::fill_fragment(acc_frag4_0, 0.0f);
	wmma::fill_fragment(acc_frag5_0, 0.0f);
	wmma::fill_fragment(acc_frag1_1, 0.0f);
	wmma::fill_fragment(acc_frag2_1, 0.0f);
	wmma::fill_fragment(acc_frag3_1, 0.0f);
	wmma::fill_fragment(acc_frag4_1, 0.0f);
	wmma::fill_fragment(acc_frag5_1, 0.0f);
	wmma::fill_fragment(acc_frag2_2, 0.0f);
	wmma::fill_fragment(acc_frag3_2, 0.0f);
	wmma::fill_fragment(acc_frag4_2, 0.0f);
	wmma::fill_fragment(acc_frag5_2, 0.0f);
	wmma::fill_fragment(acc_frag3_3, 0.0f);
	wmma::fill_fragment(acc_frag4_3, 0.0f);
	wmma::fill_fragment(acc_frag5_3, 0.0f);
	wmma::fill_fragment(acc_frag4_4, 0.0f);
	wmma::fill_fragment(acc_frag5_3, 0.0f);
	wmma::fill_fragment(acc_frag5_5, 0.0f);

	int user_idx = blockIdx.x + m_batch_offset;

	int start = csr_row_ptrs[user_idx];
	int end = csr_row_ptrs[user_idx + 1];

	int items_cnt = end - start;

	int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;

//	int tile_row = 0;
//	int tile_col = 0;
//
//	int tile_dim = f / 16;	// tile_dim * tile_dim square
//
//	for(int col = 0; col < tile_dim; ++col) {
//		int next_col_start = (2 * tile_dim - col) * (col + 1) / 2;
//		if(warp_id < next_col_start) {
//			tile_row = tile_dim + warp_id - next_col_start;
//			tile_col = col;
//			break;
//		}
//	}

	for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
		int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

		// First 64(f) threads read smem_col_cnt columns, others are idle
		if(threadIdx.x < f) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + threadIdx.x] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x];
				++smem_col;

#ifdef FIND_NAN
				if(smem_half[f * smem_col + threadIdx.x] != smem_half[f * smem_col + threadIdx.x]) {
					printf("smem_half nan found blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
				}
#endif

			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + threadIdx.x], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		__syncthreads();

		// actual work

		for(int tile_iter = 0; tile_iter < smem_col_cnt / 16; ++tile_iter) {

			//wmma::load_matrix_sync(vt_frag, smem_half + tile_row * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + tile_col * 16 + tile_iter * f * 16, f);

			//wmma::mma_sync(acc_frag, vt_frag, v_frag, acc_frag);

			wmma::load_matrix_sync(vt_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag0_0, vt_frag, v_frag, acc_frag0_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_0, vt_frag, v_frag, acc_frag1_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_0, vt_frag, v_frag, acc_frag2_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_0, vt_frag, v_frag, acc_frag3_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_0, vt_frag, v_frag, acc_frag4_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_0, vt_frag, v_frag, acc_frag5_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_1, vt_frag, v_frag, acc_frag1_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_1, vt_frag, v_frag, acc_frag2_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_1, vt_frag, v_frag, acc_frag3_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_1, vt_frag, v_frag, acc_frag4_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_1, vt_frag, v_frag, acc_frag5_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_2, vt_frag, v_frag, acc_frag2_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_2, vt_frag, v_frag, acc_frag3_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_2, vt_frag, v_frag, acc_frag4_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_2, vt_frag, v_frag, acc_frag5_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_3, vt_frag, v_frag, acc_frag3_3);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_3, vt_frag, v_frag, acc_frag4_3);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_3, vt_frag, v_frag, acc_frag5_3);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_4, vt_frag, v_frag, acc_frag4_4);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_4, vt_frag, v_frag, acc_frag5_4);

			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag5_5, vt_frag, v_frag, acc_frag5_5);
		}

		__syncthreads();
	}

	// store

	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_col * 16 * f + tile_row * 16, acc_frag, f, wmma::mem_col_major);

	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 1 * 16, acc_frag1_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 2 * 16, acc_frag2_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 3 * 16, acc_frag3_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 4 * 16, acc_frag4_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 5 * 16, acc_frag5_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 2 * 16, acc_frag2_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 3 * 16, acc_frag3_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 4 * 16, acc_frag4_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 5 * 16, acc_frag5_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 3 * 16, acc_frag3_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 4 * 16, acc_frag4_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 5 * 16, acc_frag5_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 4 * 16, acc_frag4_3, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 5 * 16, acc_frag5_3, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 5 * 16, acc_frag5_4, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_col_major);

	//if(tile_row != tile_col) {
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_row * 16 * f + tile_col * 16, acc_frag, f, wmma::mem_row_major);

		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 0 * 16, acc_frag1_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 0 * 16, acc_frag2_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 0 * 16, acc_frag3_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 0 * 16, acc_frag4_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 0 * 16, acc_frag5_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 1 * 16, acc_frag2_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 1 * 16, acc_frag3_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 1 * 16, acc_frag4_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 1 * 16, acc_frag5_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 2 * 16, acc_frag3_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 2 * 16, acc_frag4_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 2 * 16, acc_frag5_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 3 * 16, acc_frag4_3, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 3 * 16, acc_frag5_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 4 * 16, acc_frag5_4, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_row_major);
//	}
}

__global__
void calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f80(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, half *VT_half, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ half smem_half [];

	//const unsigned int warp_id = threadIdx.x / 32;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> vt_frag;	// actually col-major
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;	// vt interpreted as row-major -> vt transposed -> v
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag0_0;		// tile_row = 0, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_0;		// tile_row = 1, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_0;		// tile_row = 2, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_0;		// tile_row = 3, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_0;		// tile_row = 4, tile_col = 0
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_0;		// tile_row = 5, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_1;		// tile_row = 1, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_1;		// tile_row = 2, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_1;		// tile_row = 3, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_1;		// tile_row = 4, tile_col = 1
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_1;		// tile_row = 5, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_2;		// tile_row = 2, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_2;		// tile_row = 3, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_2;		// tile_row = 4, tile_col = 2
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_2;		// tile_row = 5, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_3;		// tile_row = 3, tile_col = 3
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_3;		// tile_row = 4, tile_col = 3
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_3;		// tile_row = 5, tile_col = 3
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_4;		// tile_row = 4, tile_col = 4
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_4;		// tile_row = 4, tile_col = 5
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_5;		// tile_row = 5, tile_col = 5

	//wmma::fill_fragment(acc_frag, 0.0f);
	wmma::fill_fragment(acc_frag0_0, 0.0f);
	wmma::fill_fragment(acc_frag1_0, 0.0f);
	wmma::fill_fragment(acc_frag2_0, 0.0f);
	wmma::fill_fragment(acc_frag3_0, 0.0f);
	wmma::fill_fragment(acc_frag4_0, 0.0f);
	//wmma::fill_fragment(acc_frag5_0, 0.0f);
	wmma::fill_fragment(acc_frag1_1, 0.0f);
	wmma::fill_fragment(acc_frag2_1, 0.0f);
	wmma::fill_fragment(acc_frag3_1, 0.0f);
	wmma::fill_fragment(acc_frag4_1, 0.0f);
	//wmma::fill_fragment(acc_frag5_1, 0.0f);
	wmma::fill_fragment(acc_frag2_2, 0.0f);
	wmma::fill_fragment(acc_frag3_2, 0.0f);
	wmma::fill_fragment(acc_frag4_2, 0.0f);
	//wmma::fill_fragment(acc_frag5_2, 0.0f);
	wmma::fill_fragment(acc_frag3_3, 0.0f);
	wmma::fill_fragment(acc_frag4_3, 0.0f);
	//wmma::fill_fragment(acc_frag5_3, 0.0f);
	wmma::fill_fragment(acc_frag4_4, 0.0f);
	//wmma::fill_fragment(acc_frag5_3, 0.0f);
	//wmma::fill_fragment(acc_frag5_5, 0.0f);

	int user_idx = blockIdx.x + m_batch_offset;

	int start = csr_row_ptrs[user_idx];
	int end = csr_row_ptrs[user_idx + 1];

	int items_cnt = end - start;

	int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;

	for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
		int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

		// f=80: all 32 threads read smem_col_cnt columns, 2 rows per thread (64 rows)
		// then 16 threads read smem_col_cnt columns, 1 row per thread (16)
		if(threadIdx.x < 64) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + threadIdx.x * 2] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x * 2];
				smem_half[f * smem_col + threadIdx.x * 2 + 1] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x * 2 + 1];
				++smem_col;
			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + threadIdx.x * 2], 0, sizeof(smem_half[0]));
				memset(&smem_half[f * smem_col + threadIdx.x * 2 + 1], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		if(threadIdx.x < 16) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + 64 + threadIdx.x] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + 64 + threadIdx.x];
				++smem_col;
			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + 64 + threadIdx.x], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		__syncthreads();

		// actual work

		for(int tile_iter = 0; tile_iter < smem_col_cnt / 16; ++tile_iter) {
			//wmma::load_matrix_sync(vt_frag, smem_half + tile_row * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + tile_col * 16 + tile_iter * f * 16, f);

			//wmma::mma_sync(acc_frag, vt_frag, v_frag, acc_frag);

			wmma::load_matrix_sync(vt_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);

			wmma::mma_sync(acc_frag0_0, vt_frag, v_frag, acc_frag0_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_0, vt_frag, v_frag, acc_frag1_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_0, vt_frag, v_frag, acc_frag2_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_0, vt_frag, v_frag, acc_frag3_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_0, vt_frag, v_frag, acc_frag4_0);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_0, vt_frag, v_frag, acc_frag5_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_1, vt_frag, v_frag, acc_frag1_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_1, vt_frag, v_frag, acc_frag2_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_1, vt_frag, v_frag, acc_frag3_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_1, vt_frag, v_frag, acc_frag4_1);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_1, vt_frag, v_frag, acc_frag5_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_2, vt_frag, v_frag, acc_frag2_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_2, vt_frag, v_frag, acc_frag3_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_2, vt_frag, v_frag, acc_frag4_2);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_2, vt_frag, v_frag, acc_frag5_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_3, vt_frag, v_frag, acc_frag3_3);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_3, vt_frag, v_frag, acc_frag4_3);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_3, vt_frag, v_frag, acc_frag5_3);

			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag4_4, vt_frag, v_frag, acc_frag4_4);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_4, vt_frag, v_frag, acc_frag5_4);
//
//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			wmma::load_matrix_sync(v_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_5, vt_frag, v_frag, acc_frag5_5);3
			__syncthreads();
		}

		__syncthreads();
	}

	// regularization

	float *smem_half_reg_term_buf = (float *)(smem_half + smem_col_cnt * f);

	// 16 * 16 = 256 items in regularization fragment
	// we have 32 threads so 256 / 32 = 8 items per thread
	// needs update in case of multiple warps per block

	smem_half_reg_term_buf[threadIdx.x * 8] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 1] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 2] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 3] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 4] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 5] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 6] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 7] = 0;

	__syncthreads();

	if(threadIdx.x < 16) {
		smem_half_reg_term_buf[threadIdx.x * 16 + threadIdx.x] = items_cnt * lambda;
	}

	__syncthreads();

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> reg_frag;

	wmma::load_matrix_sync(reg_frag, smem_half_reg_term_buf, 16, wmma::mem_row_major);
	for(int i = 0; i < reg_frag.num_elements; ++i) {
		acc_frag0_0.x[i] += reg_frag.x[i];
		acc_frag1_1.x[i] += reg_frag.x[i];
		acc_frag2_2.x[i] += reg_frag.x[i];
		acc_frag3_3.x[i] += reg_frag.x[i];
		acc_frag4_4.x[i] += reg_frag.x[i];
	}

	// store

	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_col * 16 * f + tile_row * 16, acc_frag, f, wmma::mem_col_major);

	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 1 * 16, acc_frag1_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 2 * 16, acc_frag2_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 3 * 16, acc_frag3_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 4 * 16, acc_frag4_0, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 5 * 16, acc_frag5_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 2 * 16, acc_frag2_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 3 * 16, acc_frag3_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 4 * 16, acc_frag4_1, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 5 * 16, acc_frag5_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 3 * 16, acc_frag3_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 4 * 16, acc_frag4_2, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 5 * 16, acc_frag5_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 4 * 16, acc_frag4_3, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 5 * 16, acc_frag5_3, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 5 * 16, acc_frag5_4, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_col_major);

	//if(tile_row != tile_col) {
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_row * 16 * f + tile_col * 16, acc_frag, f, wmma::mem_row_major);

		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 0 * 16, acc_frag1_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 0 * 16, acc_frag2_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 0 * 16, acc_frag3_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 0 * 16, acc_frag4_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 0 * 16, acc_frag5_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 1 * 16, acc_frag2_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 1 * 16, acc_frag3_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 1 * 16, acc_frag4_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 1 * 16, acc_frag5_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 2 * 16, acc_frag3_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 2 * 16, acc_frag4_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 2 * 16, acc_frag5_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 3 * 16, acc_frag4_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 3 * 16, acc_frag5_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 4 * 16, acc_frag5_4, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_row_major);
//	}
}

__global__
void calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f64(float *vtvs, int *csr_row_ptrs, int *csr_col_idxs, float lambda, int m, int f, half *VT_half, int smem_col_cnt, int m_batch_offset) {
	extern __shared__ half smem_half [];

	//const unsigned int warp_id = threadIdx.x / 32;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> vt_frag;	// actually col-major
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;	// vt interpreted as row-major -> vt transposed -> v
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag0_0;		// tile_row = 0, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_0;		// tile_row = 1, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_0;		// tile_row = 2, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_0;		// tile_row = 3, tile_col = 0
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_0;		// tile_row = 4, tile_col = 0
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_0;		// tile_row = 5, tile_col = 0
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag1_1;		// tile_row = 1, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_1;		// tile_row = 2, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_1;		// tile_row = 3, tile_col = 1
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_1;		// tile_row = 4, tile_col = 1
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_1;		// tile_row = 5, tile_col = 1
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag2_2;		// tile_row = 2, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_2;		// tile_row = 3, tile_col = 2
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_2;		// tile_row = 4, tile_col = 2
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_2;		// tile_row = 5, tile_col = 2
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag3_3;		// tile_row = 3, tile_col = 3
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_3;		// tile_row = 4, tile_col = 3
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_3;		// tile_row = 5, tile_col = 3
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag4_4;		// tile_row = 4, tile_col = 4
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_4;		// tile_row = 4, tile_col = 5
	//wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag5_5;		// tile_row = 5, tile_col = 5

	//wmma::fill_fragment(acc_frag, 0.0f);
	wmma::fill_fragment(acc_frag0_0, 0.0f);
	wmma::fill_fragment(acc_frag1_0, 0.0f);
	wmma::fill_fragment(acc_frag2_0, 0.0f);
	wmma::fill_fragment(acc_frag3_0, 0.0f);
	//wmma::fill_fragment(acc_frag4_0, 0.0f);
	//wmma::fill_fragment(acc_frag5_0, 0.0f);
	wmma::fill_fragment(acc_frag1_1, 0.0f);
	wmma::fill_fragment(acc_frag2_1, 0.0f);
	wmma::fill_fragment(acc_frag3_1, 0.0f);
	//wmma::fill_fragment(acc_frag4_1, 0.0f);
	//wmma::fill_fragment(acc_frag5_1, 0.0f);
	wmma::fill_fragment(acc_frag2_2, 0.0f);
	wmma::fill_fragment(acc_frag3_2, 0.0f);
	//wmma::fill_fragment(acc_frag4_2, 0.0f);
	//wmma::fill_fragment(acc_frag5_2, 0.0f);
	wmma::fill_fragment(acc_frag3_3, 0.0f);
	//wmma::fill_fragment(acc_frag4_3, 0.0f);
	//wmma::fill_fragment(acc_frag5_3, 0.0f);
	//wmma::fill_fragment(acc_frag4_4, 0.0f);
	//wmma::fill_fragment(acc_frag5_3, 0.0f);
	//wmma::fill_fragment(acc_frag5_5, 0.0f);

	int user_idx = blockIdx.x + m_batch_offset;

	int start = csr_row_ptrs[user_idx];
	int end = csr_row_ptrs[user_idx + 1];

	int items_cnt = end - start;

	int smem_iters = (items_cnt - 1) / smem_col_cnt + 1;

	for(int smem_iter = 0; smem_iter < smem_iters; ++smem_iter) {
		int smem_items = smem_col_cnt * (smem_iter + 1) < items_cnt ? smem_col_cnt : items_cnt - smem_col_cnt * smem_iter;

		// f=64: all 32 threads read smem_col_cnt columns, 2 rows per thread
		if(threadIdx.x < f / 2) {
			int smem_col = 0;
			while(smem_col < smem_items) {
				smem_half[f * smem_col + threadIdx.x * 2] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x * 2];
				smem_half[f * smem_col + threadIdx.x * 2 + 1] = VT_half[f * csr_col_idxs[start + smem_iter * smem_col_cnt + smem_col] + threadIdx.x * 2 + 1];
				++smem_col;
			}
			// if this smem_iter has less than 16 cols left
			while(smem_col < smem_col_cnt) {
				memset(&smem_half[f * smem_col + threadIdx.x * 2], 0, sizeof(smem_half[0]));
				memset(&smem_half[f * smem_col + threadIdx.x * 2 + 1], 0, sizeof(smem_half[0]));
				++smem_col;
			}
		}

		__syncthreads();

		// actual work

		for(int tile_iter = 0; tile_iter < smem_col_cnt / 16; ++tile_iter) {
			//wmma::load_matrix_sync(vt_frag, smem_half + tile_row * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + tile_col * 16 + tile_iter * f * 16, f);

			//wmma::mma_sync(acc_frag, vt_frag, v_frag, acc_frag);

			wmma::load_matrix_sync(vt_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);

			wmma::mma_sync(acc_frag0_0, vt_frag, v_frag, acc_frag0_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_0, vt_frag, v_frag, acc_frag1_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_0, vt_frag, v_frag, acc_frag2_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_0, vt_frag, v_frag, acc_frag3_0);

//			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag4_0, vt_frag, v_frag, acc_frag4_0);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 0 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_0, vt_frag, v_frag, acc_frag5_0);

			wmma::load_matrix_sync(vt_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag1_1, vt_frag, v_frag, acc_frag1_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_1, vt_frag, v_frag, acc_frag2_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_1, vt_frag, v_frag, acc_frag3_1);

//			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag4_1, vt_frag, v_frag, acc_frag4_1);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 1 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_1, vt_frag, v_frag, acc_frag5_1);

			wmma::load_matrix_sync(vt_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag2_2, vt_frag, v_frag, acc_frag2_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_2, vt_frag, v_frag, acc_frag3_2);

//			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag4_2, vt_frag, v_frag, acc_frag4_2);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 2 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_2, vt_frag, v_frag, acc_frag5_2);

			wmma::load_matrix_sync(vt_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
			wmma::mma_sync(acc_frag3_3, vt_frag, v_frag, acc_frag3_3);

//			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag4_3, vt_frag, v_frag, acc_frag4_3);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 3 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_3, vt_frag, v_frag, acc_frag5_3);

//			wmma::load_matrix_sync(vt_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag4_4, vt_frag, v_frag, acc_frag4_4);

//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			//wmma::load_matrix_sync(v_frag, smem_half + 4 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_4, vt_frag, v_frag, acc_frag5_4);
//
//			wmma::load_matrix_sync(vt_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			wmma::load_matrix_sync(v_frag, smem_half + 5 * 16 + tile_iter * f * 16, f);
//			wmma::mma_sync(acc_frag5_5, vt_frag, v_frag, acc_frag5_5);3
			__syncthreads();
		}

		__syncthreads();
	}

	// regularization

	float *smem_half_reg_term_buf = (float *)(smem_half + smem_col_cnt * f);

	// 16 * 16 = 256 items in regularization fragment
	// we have 32 threads so 256 / 32 = 8 items per thread
	// needs update in case of multiple warps per block

	smem_half_reg_term_buf[threadIdx.x * 8] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 1] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 2] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 3] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 4] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 5] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 6] = 0;
	smem_half_reg_term_buf[threadIdx.x * 8 + 7] = 0;

	__syncthreads();

	if(threadIdx.x < 16) {
		smem_half_reg_term_buf[threadIdx.x * 16 + threadIdx.x] = items_cnt * lambda;
	}

	__syncthreads();

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> reg_frag;

	wmma::load_matrix_sync(reg_frag, smem_half_reg_term_buf, 16, wmma::mem_row_major);
	for(int i = 0; i < reg_frag.num_elements; ++i) {
		acc_frag0_0.x[i] += reg_frag.x[i];
		acc_frag1_1.x[i] += reg_frag.x[i];
		acc_frag2_2.x[i] += reg_frag.x[i];
		acc_frag3_3.x[i] += reg_frag.x[i];
	}

	// store

	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_col * 16 * f + tile_row * 16, acc_frag, f, wmma::mem_col_major);

	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 1 * 16, acc_frag1_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 2 * 16, acc_frag2_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 3 * 16, acc_frag3_0, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 4 * 16, acc_frag4_0, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 5 * 16, acc_frag5_0, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 2 * 16, acc_frag2_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 3 * 16, acc_frag3_1, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 4 * 16, acc_frag4_1, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 5 * 16, acc_frag5_1, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 3 * 16, acc_frag3_2, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 4 * 16, acc_frag4_2, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 5 * 16, acc_frag5_2, f, wmma::mem_col_major);
	wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 4 * 16, acc_frag4_3, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 5 * 16, acc_frag5_3, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 5 * 16, acc_frag5_4, f, wmma::mem_col_major);
	//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_col_major);

	//if(tile_row != tile_col) {
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + tile_row * 16 * f + tile_col * 16, acc_frag, f, wmma::mem_row_major);

		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 0 * 16 * f + 0 * 16, acc_frag0_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 0 * 16, acc_frag1_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 0 * 16, acc_frag2_0, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 0 * 16, acc_frag3_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 0 * 16, acc_frag4_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 0 * 16, acc_frag5_0, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 1 * 16 * f + 1 * 16, acc_frag1_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 1 * 16, acc_frag2_1, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 1 * 16, acc_frag3_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 1 * 16, acc_frag4_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 1 * 16, acc_frag5_1, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 2 * 16 * f + 2 * 16, acc_frag2_2, f, wmma::mem_row_major);
		wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 2 * 16, acc_frag3_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 2 * 16, acc_frag4_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 2 * 16, acc_frag5_2, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 3 * 16 * f + 3 * 16, acc_frag3_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 3 * 16, acc_frag4_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 3 * 16, acc_frag5_3, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 4 * 16 * f + 4 * 16, acc_frag4_4, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 4 * 16, acc_frag5_4, f, wmma::mem_row_major);
		//wmma::store_matrix_sync(vtvs + blockIdx.x * f * f + 5 * 16 * f + 5 * 16, acc_frag5_5, f, wmma::mem_row_major);
//	}
}

// TODO: warps nr = f / 32
// e.g. f = 64. within smsem iter each warp loops through its half of vvt
// and stores current smem part in shared memory
// less warps needed, but adds loop through hafl of vvt


std::string als_model::to_string(CALCULATE_VVTS_TYPE calculate_vvts_type) {
	switch(calculate_vvts_type) {
		case CALCULATE_VVTS_TYPE::SIMPLE: return "SIMPLE";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR: return "SMEM_ROW_MAJOR";
		case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR: return "SMEM_COL_MAJOR";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC: return "SMEM_ROW_MAJOR_NO_CALC";
		case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS: return "SMEM_COL_MAJOR_TWO_THREADS";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR: return "SMEM_ROW_MAJOR_TENSOR";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC: return "SMEM_ROW_MAJOR_TENSOR_SYMMETRIC";
		case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: return "SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG";
		default: return "UNKNOWN";
	}
}

als_model::als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f,
		float lambda, int iters, CALCULATE_VVTS_TYPE calculate_vvts_type, int smem_col_cnt,
		int m_batches, int n_batches):
		train_ratings(train_ratings), test_ratings(test_ratings), f(f), lambda(lambda), iters(iters),
		calculate_vvts_type(calculate_vvts_type), smem_col_cnt(smem_col_cnt), m_batches(m_batches),
		n_batches(n_batches) {
	m = train_ratings.row_cnt;
	n = train_ratings.col_cnt;

	CUDA_CHECK(cudaMallocHost((void **)&h_VT, n * f * sizeof(h_VT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VT, n * f * sizeof(d_VT[0])));

	CUDA_CHECK(cudaMallocHost((void **)&h_UT, m * f * sizeof(h_UT[0])));
	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_UT, m * f * sizeof(d_UT[0])));

	int m_first_batch_size = (m - 1) / m_batches + 1;
	int n_first_batch_size = (n - 1) / n_batches + 1;

	int x_size = m_first_batch_size > n_first_batch_size ? m_first_batch_size : n_first_batch_size;

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_xtxs, f * f * x_size * sizeof(d_xtxs[0])));

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VTRT, f * m * sizeof(d_VTRT[0])));

	CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_UTR, f * n * sizeof(d_UTR[0])));

	CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
	CUBLAS_CHECK(cublasCreate_v2(&cublas_handle));

	switch (calculate_vvts_type) {
	case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
	case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
	case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
		CUDA_CHECK(cudaMemset(d_xtxs, 0, f * f * m_first_batch_size * sizeof(d_xtxs[0])));
	}
}

als_model::~als_model() {
	CUDA_CHECK(cudaFreeHost(h_VT));
	CUDA_CHECK(cudaFree(d_VT));

	CUDA_CHECK(cudaFreeHost(h_UT));
	CUDA_CHECK(cudaFree(d_UT));

	CUDA_CHECK(cudaFree(d_xtxs));

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

#ifdef DEBUG_SAVE
			save_host_array_float(h_VT, f * n, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_4_VT");
#endif

	CUDA_CHECK(cudaMemcpy(d_VT, h_VT, n * f * sizeof(h_VT[0]), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_UT, h_UT, m * f * sizeof(h_UT[0]), cudaMemcpyHostToDevice));

#ifdef USE_LOGGER
	g_logger.log("factors initialization done", true);
#endif

	for(size_t it = 0; it < iters; ++it) {

#ifdef USE_LOGGER
		g_logger.als_iter = it + 1;
		g_logger.event_started(logger::EVENT_TYPE::ALS_ITER);
#endif

		// ---------- update U ----------
		{

#ifdef USE_LOGGER
			g_logger.event_started(logger::EVENT_TYPE::ALS_UPDATE_U);
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
			half *d_VT_half = 0;
			int symmetric_tiles_cnt = 0;

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
				smem_size = smem_col_cnt * f * sizeof(d_VT[0]);

#ifdef USE_LOGGER
				g_logger.log("vtvs smem_col_cnt=" + std::to_string(smem_col_cnt) + " smem_size=" + std::to_string(smem_size), true);
#endif

			}
			switch (calculate_vvts_type) {
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
					CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_VT_half, n * f * sizeof(d_VT_half[0])));
					float2half_array<<<(n*f-1)/1024 + 1, 1024>>>(d_VT, d_VT_half, f*n);
			}

			for(int m_batch = 0; m_batch < m_batches; ++m_batch) {
				int m_first_batch_size = (m - 1) / m_batches + 1;

				int m_batch_size = m_first_batch_size * (m_batch + 1) < m ? m_first_batch_size : m - m_first_batch_size * m_batch;

				int m_batch_offset = m_batch * m_first_batch_size;

				switch (calculate_vvts_type) {
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
					calculate_vtvs_smem_row_major<<<m_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
					calculate_vtvs_smem_row_major_no_calc<<<m_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
					if(f < smem_col_cnt) {
						throw std::runtime_error("SMEM_COL_MAJOR: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt("
								+ std::to_string(smem_col_cnt) + ")"
						);
					}
					calculate_vtvs_smem_col_major<<<m_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
					if(f < smem_col_cnt * 2) {
						throw std::runtime_error("SMEM_COL_MAJOR_TWO_THREADS: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt * 2 ("
								+ std::to_string(smem_col_cnt) + " * 2 = " + std::to_string(smem_col_cnt * 2) + ")"
						);
					}
					calculate_vtvs_smem_col_major_two_threads<<<m_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}
					calculate_vtvs_smem_row_major_tensor<<<m_batch_size, (f * f / 16 / 16) * 32, smem_size + 16 * 16>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT_half, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}

					symmetric_tiles_cnt = (f / 16) * (f / 16 + 1) / 2;

#ifdef USE_LOGGER
					g_logger.log("vtvs symmetric_tiles_cnt=" + std::to_string(symmetric_tiles_cnt) + " m_batch=" + std::to_string(m_batch), true);
#endif

					calculate_vtvs_smem_row_major_tensor_symmetric<<<m_batch_size, symmetric_tiles_cnt * 32, smem_size>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT_half, smem_col_cnt, m_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}
					if(f == 80) {
						calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f80<<<m_batch_size, 32, smem_size + 16 * 16>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
								train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT_half, smem_col_cnt, m_batch_offset
						);
					} else if (f == 64) {
						calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f64<<<m_batch_size, 32, smem_size + 16 * 16>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
								train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT_half, smem_col_cnt, m_batch_offset
						);
					} else {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: f(" + std::to_string(f) + ") not supported");
					}
					break;
				case CALCULATE_VVTS_TYPE::SIMPLE:
				default:
					calculate_vtvs<<<m_batch_size, f>>>(d_xtxs, train_ratings.d_csr_row_ptrs,
							train_ratings.d_csr_coo_col_idxs, lambda, m, f, d_VT, m_batch_offset
					);
				}
				CUDA_CHECK(cudaPeekAtLastError());

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
				g_logger.log("vtvs calculation done type=" + to_string(calculate_vvts_type) + " m_batch=" + std::to_string(m_batch), true);
#endif

#ifdef DEBUG_SAVE
				save_device_array_float(d_xtxs + f * f * 5, f * f, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_1_vtv_user_5_m_batch=" + std::to_string(m_batch));
				//save_device_array_float(d_xtxs + f * f * 11, f * f, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_1_vtv_user_11_m_batch=" + std::to_string(m_batch));
				//save_device_array_float(d_xtxs + f * f * 12, f * f, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_1_vtv_user_12_m_batch=" + std::to_string(m_batch));
				//save_device_array_float(d_xtxs + f * f * 13, f * f, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_1_vtv_user_13_m_batch=" + std::to_string(m_batch));
				//save_device_array_float(d_xtxs, f * f * m_batch_size, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_1_vtvs_m_batch=" + std::to_string(m_batch));
#endif

				// TODO: single array of max(m, n) allocated in model constructor

				// host array of pointers to each device vtv
				float **h_d_vtvs_ptrs;

				CUDA_CHECK(cudaMallocHost((void **)&h_d_vtvs_ptrs, m_batch_size * sizeof(h_d_vtvs_ptrs[0])));

				for(size_t i = 0; i < m_batch_size; ++i) {
					h_d_vtvs_ptrs[i] = &d_xtxs[i * f * f];
				}

				// device array of pointers to each device vtv
				float **d_d_vtvs_ptrs;

				CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_vtvs_ptrs, m_batch_size * sizeof(d_d_vtvs_ptrs[0])));
				CUDA_CHECK(cudaMemcpy(d_d_vtvs_ptrs, h_d_vtvs_ptrs, m_batch_size * sizeof(h_d_vtvs_ptrs[0]), cudaMemcpyHostToDevice));

				// required by cublasSgetrfBatched but not used for now
				int *d_getrf_infos;

				CUDA_CHECK(CUDA_MALLOC_DEVICE(&d_getrf_infos, m_batch_size * sizeof(d_getrf_infos[0])));

				// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
				CUBLAS_CHECK(cublasSgetrfBatched(cublas_handle, f, d_d_vtvs_ptrs, f, NULL, d_getrf_infos, m_batch_size));

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
				g_logger.log("vtvs batched LU factorization done m_batch=" + std::to_string(m_batch), true);
#endif

				int getrs_info;

				// host array of pointers to each device VTRT column
				float **h_d_VTRT_ptrs;

				CUDA_CHECK(cudaMallocHost((void **)&h_d_VTRT_ptrs, m_batch_size * sizeof(h_d_VTRT_ptrs[0])));

				for(size_t i = 0; i < m_batch_size; ++i) {
					h_d_VTRT_ptrs[i] = &d_VTRT[(m_batch_offset + i) * f];
				}

				// device array of pointers to each device VTRT column
				float **d_d_VTRT_ptrs;

				CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_VTRT_ptrs, m_batch_size * sizeof(d_d_VTRT_ptrs[0])));
				CUDA_CHECK(cudaMemcpy(d_d_VTRT_ptrs, h_d_VTRT_ptrs, m_batch_size * sizeof(h_d_VTRT_ptrs[0]), cudaMemcpyHostToDevice));

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
						m_batch_size
				));

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

				// write result
				CUDA_CHECK(cudaMemcpy(d_UT + f * m_batch_offset, d_VTRT + f * m_batch_offset, m_batch_size * f * sizeof(d_VTRT[0]), cudaMemcpyDeviceToDevice));

#ifdef USE_LOGGER
				g_logger.log("U batched solve done m_batch=" + std::to_string(m_batch), true);
#endif

				switch (calculate_vvts_type) {
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
					CUDA_CHECK(cudaMemset(d_xtxs, 0, f * f * m_batch_size * sizeof(d_xtxs[0])));
				}

				CUDA_CHECK(cudaFreeHost(h_d_vtvs_ptrs));
				CUDA_CHECK(cudaFree(d_d_vtvs_ptrs));
				CUDA_CHECK(cudaFree(d_getrf_infos));
				CUDA_CHECK(cudaFreeHost(h_d_VTRT_ptrs));
				CUDA_CHECK(cudaFree(d_d_VTRT_ptrs));
			}	// m_batch loop

			CUDA_CHECK(cudaFree(d_VT_half));

#ifdef USE_LOGGER
			g_logger.event_finished(logger::EVENT_TYPE::ALS_UPDATE_U, true);
#endif

#ifdef DEBUG_SAVE
			save_device_array_float(d_UT, f * m, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_2_UT");
#endif

		}	// update U block
		// ---------- update V ----------
		{

#ifdef USE_LOGGER
			g_logger.event_started(logger::EVENT_TYPE::ALS_UPDATE_V);
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
			g_logger.log("UTR via cuSPARSE and cuBLAS done", true);
#endif

#ifdef USE_LOGGER
			g_logger.log("utus calculation started type=" + to_string(calculate_vvts_type), true);
#endif

			// Function is named calculate_vtvs but here we actually calculate utus.
			// Naming is kept for U update for easier debugging

			int smem_size = 0;
			half *d_UT_half = 0;
			int symmetric_tiles_cnt = 0;

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
			case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
				smem_size = smem_col_cnt * f * sizeof(d_UT[0]);

#ifdef USE_LOGGER
				g_logger.log("utus smem_col_cnt=" + std::to_string(smem_col_cnt) + " smem_size=" + std::to_string(smem_size), true);
#endif

			}

			switch (calculate_vvts_type) {
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
			case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
				CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_UT_half, m * f * sizeof(d_UT_half[0])));
				float2half_array<<<(m*f-1)/1024 + 1, 1024>>>(d_UT, d_UT_half, f*m);
			}

			for(int n_batch = 0; n_batch < n_batches; ++n_batch) {
				int n_first_batch_size = (n - 1) / n_batches + 1;

				int n_batch_size = n_first_batch_size * (n_batch + 1) < n ? n_first_batch_size : n - n_first_batch_size * n_batch;

				int n_batch_offset = n_batch * n_first_batch_size;

				switch (calculate_vvts_type) {
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
					calculate_vtvs_smem_row_major<<<n_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC:
					calculate_vtvs_smem_row_major_no_calc<<<n_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
					if(f < smem_col_cnt) {
						throw std::runtime_error("SMEM_COL_MAJOR: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt("
								+ std::to_string(smem_col_cnt) + ")"
						);
					}
					calculate_vtvs_smem_col_major<<<n_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
					if(f < smem_col_cnt * 2) {
						throw std::runtime_error("SMEM_COL_MAJOR_TWO_THREADS: f(" + std::to_string(f) + ") should be greater than or equal to smem_col_cnt * 2 ("
								+ std::to_string(smem_col_cnt) + " * 2 = " + std::to_string(smem_col_cnt * 2) + ")"
						);
					}
					calculate_vtvs_smem_col_major_two_threads<<<n_batch_size, f, smem_size>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}
					calculate_vtvs_smem_row_major_tensor<<<n_batch_size, (f * f / 16 / 16) * 32, smem_size + 16 * 16>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT_half, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}

					symmetric_tiles_cnt = (f / 16) * (f / 16 + 1) / 2;

#ifdef USE_LOGGER
					g_logger.log("utus symmetric_tiles_cnt=" + std::to_string(symmetric_tiles_cnt) + " n_batch=" + std::to_string(n_batch), true);
#endif

					calculate_vtvs_smem_row_major_tensor_symmetric<<<n_batch_size, symmetric_tiles_cnt * 32, smem_size>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT_half, smem_col_cnt, n_batch_offset
					);
					break;
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG:
					if(f % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: f(" + std::to_string(f) + ") % 16 should be equal to 0");
					}
					if(smem_col_cnt % 16 != 0) {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: smem_col_cnt(" + std::to_string(smem_col_cnt) + ") % 16 should be equal to 0");
					}

					if(f == 80) {
						calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f80<<<n_batch_size, 32, smem_size + 16 * 16>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
								train_ratings.d_csc_row_idxs, lambda, n, f, d_UT_half, smem_col_cnt, n_batch_offset
						);
					} else if (f == 64) {
						calculate_vtvs_smem_row_major_tensor_symmetric_mult_frag_f64<<<n_batch_size, 32, smem_size  + 16 * 16>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
								train_ratings.d_csc_row_idxs, lambda, n, f, d_UT_half, smem_col_cnt, n_batch_offset
						);
					} else {
						throw std::runtime_error("SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG: f(" + std::to_string(f) + ") not supported");
					}
					break;
				case CALCULATE_VVTS_TYPE::SIMPLE:
				default:
					calculate_vtvs<<<n_batch_size, f>>>(d_xtxs, train_ratings.d_csc_col_ptrs,
							train_ratings.d_csc_row_idxs, lambda, n, f, d_UT, n_batch_offset
					);
				}

				CUDA_CHECK(cudaPeekAtLastError());

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
				g_logger.log("utus calculation done type=" + to_string(calculate_vvts_type) + " n_batch=" + std::to_string(n_batch), true);
#endif

#ifdef DEBUG_SAVE
				//save_device_array_float(d_xtxs + f * f * 9, f * f, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_3_utu_item_9_n_batch=" + std::to_string(n_batch));
				//save_device_array_float(d_xtxs, f * f * n_batch_size, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_3_utus_n_batch=" + std::to_string(n_batch));
#endif

				// TODO: single array of max(m, n) allocated in model constructor

				// host array of pointers to each device utu
				float **h_d_utus_ptrs;

				CUDA_CHECK(cudaMallocHost((void **)&h_d_utus_ptrs, n_batch_size * sizeof(h_d_utus_ptrs[0])));

				for(size_t i = 0; i < n_batch_size; ++i) {
					h_d_utus_ptrs[i] = &d_xtxs[i * f * f];
				}

				// device array of pointers to each device utu
				float **d_d_utus_ptrs;

				CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_utus_ptrs, n_batch_size * sizeof(d_d_utus_ptrs[0])));
				CUDA_CHECK(cudaMemcpy(d_d_utus_ptrs, h_d_utus_ptrs, n_batch_size * sizeof(h_d_utus_ptrs[0]), cudaMemcpyHostToDevice));

				// required by cublasSgetrfBatched but not used for now
				int *d_getrf_infos;

				CUDA_CHECK(CUDA_MALLOC_DEVICE(&d_getrf_infos, n_batch_size * sizeof(d_getrf_infos[0])));

				// stepping here in Nsight debug session causes GDB crash so don't put breakpoints here
				CUBLAS_CHECK(cublasSgetrfBatched(cublas_handle, f, d_d_utus_ptrs, f, NULL, d_getrf_infos, n_batch_size));

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

#ifdef USE_LOGGER
				g_logger.log("utus batched LU factorization done n_batch=" + std::to_string(n_batch), true);
#endif

				int getrs_info;

				// host array of pointers to each device UTR column
				float **h_d_UTR_ptrs;

				CUDA_CHECK(cudaMallocHost((void **)&h_d_UTR_ptrs, n_batch_size * sizeof(h_d_UTR_ptrs[0])));

				for(size_t i = 0; i < n_batch_size; ++i) {
					h_d_UTR_ptrs[i] = &d_UTR[(n_batch_offset + i) * f];
				}

				// device array of pointers to each device UTR column
				float **d_d_UTR_ptrs;

				CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_d_UTR_ptrs, n_batch_size * sizeof(d_d_utus_ptrs[0])));
				CUDA_CHECK(cudaMemcpy(d_d_UTR_ptrs, h_d_UTR_ptrs, n_batch_size * sizeof(h_d_UTR_ptrs[0]), cudaMemcpyHostToDevice));

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
						n_batch_size
				));

#if defined (DEBUG) || defined(USE_LOGGER)
				CUDA_CHECK(cudaDeviceSynchronize());
#endif

				// write result
				CUDA_CHECK(cudaMemcpy(d_VT + f * n_batch_offset, d_UTR + f * n_batch_offset, n_batch_size * f * sizeof(d_UTR[0]), cudaMemcpyDeviceToDevice));

#ifdef USE_LOGGER
				g_logger.log("V batched solve done n_batch=" + std::to_string(n_batch), true);
#endif

				switch (calculate_vvts_type) {
				case CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR:
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR:
				case CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR_TWO_THREADS:
					CUDA_CHECK(cudaMemset(d_xtxs, 0, f * f * n_batch_size * sizeof(d_xtxs[0])));
				}

				CUDA_CHECK(cudaFreeHost(h_d_utus_ptrs));
				CUDA_CHECK(cudaFree(d_d_utus_ptrs));
				CUDA_CHECK(cudaFree(d_getrf_infos));
				CUDA_CHECK(cudaFreeHost(h_d_UTR_ptrs));
				CUDA_CHECK(cudaFree(d_d_UTR_ptrs));
			}	// n_batch loop

			CUDA_CHECK(cudaFree(d_UT_half));

#ifdef USE_LOGGER
			g_logger.event_finished(logger::EVENT_TYPE::ALS_UPDATE_V, true);
#endif

#ifdef DEBUG_SAVE
			save_device_array_float(d_VT, f * n, "/home/vladimir/src/cuda_als/tmp/run_" + std::to_string(g_logger.run_iter) + "_iter_" + std::to_string(g_logger.als_iter) + "_4_VT");
#endif
		}	// update V block

#ifdef USE_LOGGER
		g_logger.event_finished(logger::EVENT_TYPE::ALS_ITER, true);
#endif

#ifdef CALC_RSME

		float *d_err_arr = 0;
		int err_size = 1000;
		CUDA_CHECK(CUDA_MALLOC_DEVICE((void **)&d_err_arr, err_size * sizeof(d_err_arr[0])));

		CUDA_CHECK(cudaMemset(d_err_arr, 0, err_size * sizeof(float)));

		calculate_square_error<<<(train_ratings.val_cnt - 1) / 256 + 1, 256>>>(train_ratings.d_csr_coo_vals, train_ratings.d_coo_row_idxs,
				train_ratings.d_csr_coo_col_idxs, train_ratings.val_cnt, d_UT, d_VT, f, d_err_arr, err_size
		);

		float square_err_train = 0;

		CUBLAS_CHECK(cublasSasum(cublas_handle, err_size, d_err_arr, 1, &square_err_train));

		CUDA_CHECK(cudaDeviceSynchronize());

#ifdef USE_LOGGER
		g_logger.log("train root-mean-square error: " + std::to_string(sqrt(square_err_train / train_ratings.val_cnt)), true);
#endif

		CUDA_CHECK(cudaMemset(d_err_arr, 0, err_size * sizeof(d_err_arr[0])));

		calculate_square_error<<<(test_ratings.val_cnt - 1) / 256 + 1, 256>>>(test_ratings.d_csr_coo_vals, test_ratings.d_coo_row_idxs,
				test_ratings.d_csr_coo_col_idxs, test_ratings.val_cnt, d_UT, d_VT, f, d_err_arr, err_size
		);

		float square_err_test = 0;

		CUBLAS_CHECK(cublasSasum(cublas_handle, err_size, d_err_arr, 1, &square_err_test));

		CUDA_CHECK(cudaDeviceSynchronize());

#ifdef USE_LOGGER
		g_logger.log("test root-mean-square error: " + std::to_string(sqrt(square_err_test / test_ratings.val_cnt)), true);
#endif

		CUDA_CHECK(cudaFree(d_err_arr));

#endif	// CALC_RSME
	}	// iters loop

#ifdef USE_LOGGER
	g_logger.als_iter = 0;
#endif

	// final result from device to host

	CUDA_CHECK(cudaMemcpy(h_VT, d_VT, n * f * sizeof(d_VT[0]), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_UT, d_UT, m * f * sizeof(d_UT[0]), cudaMemcpyDeviceToHost));

#ifdef USE_LOGGER
	g_logger.log("als model training done", true);
#endif
}

