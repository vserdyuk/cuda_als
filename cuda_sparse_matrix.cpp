#include "cuda_sparse_matrix.h"
#include "cuda_common.h"
#include <fstream>
#include <cstring>

void host_sparse_matrix::malloc_load_file(std::string path, void **host_ptr, size_t buf_size) {
	std::ifstream ifs(path, std::ios::binary|std::ios::ate);

	if(!ifs)
		throw std::runtime_error(path + ": " + std::strerror(errno));

	auto end = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	auto file_size = std::size_t(end - ifs.tellg());

	if(file_size == 0)
		throw std::runtime_error(path + "is empty");

	if(file_size != buf_size)
		throw std::runtime_error("file size does not match buffer size");

	CUDA_CHECK(cudaMallocHost(host_ptr, buf_size));

	if(!ifs.read((char*)*host_ptr, buf_size))
		throw std::runtime_error(path + ": " + std::strerror(errno));
}

host_sparse_matrix::host_sparse_matrix(size_t row_cnt, size_t col_cnt, size_t val_cnt):
		row_cnt(row_cnt), col_cnt(col_cnt), val_cnt(val_cnt)
{}

host_sparse_matrix::~host_sparse_matrix() {
	CUDA_CHECK(cudaFreeHost(h_csr_coo_vals));
	CUDA_CHECK(cudaFreeHost(h_csr_row_ptrs));
	CUDA_CHECK(cudaFreeHost(h_csr_coo_col_idxs));
	CUDA_CHECK(cudaFreeHost(h_coo_row_idxs));
	CUDA_CHECK(cudaFreeHost(h_csc_vals));
	CUDA_CHECK(cudaFreeHost(h_csc_row_idxs));
	CUDA_CHECK(cudaFreeHost(h_csc_col_ptrs));
}

void host_sparse_matrix::load_csr(std::string vals_path, std::string row_ptrs_path, std::string col_idxs_path) {
	if(!has_coo) {
		malloc_load_file(vals_path, (void**)&h_csr_coo_vals, val_cnt * sizeof(h_csr_coo_vals[0]));
		malloc_load_file(col_idxs_path, (void**)&h_csr_coo_col_idxs, val_cnt * sizeof(h_csr_coo_col_idxs[0]));
	}
	malloc_load_file(row_ptrs_path, (void**)&h_csr_row_ptrs, (row_cnt + 1) * sizeof(h_csr_row_ptrs[0]));

	has_csr = true;
}

void host_sparse_matrix::load_csc(std::string vals_path, std::string row_idxs_path, std::string col_ptrs_path) {
	malloc_load_file(vals_path, (void**)&h_csc_vals, val_cnt * sizeof(h_csc_vals[0]));
	malloc_load_file(row_idxs_path, (void**)&h_csc_row_idxs, val_cnt * sizeof(h_csc_row_idxs[0]));
	malloc_load_file(col_ptrs_path, (void**)&h_csc_col_ptrs, (col_cnt + 1) * sizeof(h_csc_col_ptrs[0]));

	has_csc = true;
}

void host_sparse_matrix::load_coo(std::string vals_path, std::string row_idxs_path, std::string col_idxs_path) {
	if(!has_csr) {
		malloc_load_file(vals_path, (void**)&h_csr_coo_vals, val_cnt * sizeof(h_csr_coo_vals[0]));
		malloc_load_file(col_idxs_path, (void**)&h_csr_coo_col_idxs, val_cnt * sizeof(h_csr_coo_col_idxs[0]));
	}
	malloc_load_file(row_idxs_path, (void**)&h_coo_row_idxs, val_cnt * sizeof(h_coo_row_idxs[0]));

	has_coo = true;
}

void device_sparse_matrix::malloc_cpy_from_host(void **host_ptr, void **dev_ptr, size_t buf_size) {
	CUDA_CHECK(CUDA_MALLOC_DEVICE(dev_ptr, buf_size));
	CUDA_CHECK(cudaMemcpy(*dev_ptr, *host_ptr, buf_size, cudaMemcpyHostToDevice));
}

device_sparse_matrix::device_sparse_matrix(host_sparse_matrix &host_matrix) {
	CUSPARSE_CHECK(cusparseCreateMatDescr(&cusparse_descr));
	CUSPARSE_CHECK(cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK(cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO));

	row_cnt = host_matrix.row_cnt;
	col_cnt = host_matrix.col_cnt;
	val_cnt = host_matrix.val_cnt;

	if(host_matrix.has_csr) {
		if(!has_coo) {
			malloc_cpy_from_host((void**)&host_matrix.h_csr_coo_vals, (void**)&d_csr_coo_vals, val_cnt * sizeof(host_matrix.h_csr_coo_vals[0]));
			malloc_cpy_from_host((void**)&host_matrix.h_csr_coo_col_idxs, (void**)&d_csr_coo_col_idxs, val_cnt * sizeof(host_matrix.h_csr_coo_col_idxs[0]));
		}
		malloc_cpy_from_host((void**)&(host_matrix.h_csr_row_ptrs), (void**)(&d_csr_row_ptrs), (row_cnt + 1) * sizeof(host_matrix.h_csr_row_ptrs[0]));

		has_csr = true;
	}
	if(host_matrix.has_csc) {
		malloc_cpy_from_host((void**)&host_matrix.h_csc_vals, (void**)&d_csc_vals, val_cnt * sizeof(host_matrix.h_csc_vals[0]));
		malloc_cpy_from_host((void**)&host_matrix.h_csc_row_idxs, (void**)&d_csc_row_idxs, val_cnt * sizeof(host_matrix.h_csc_row_idxs[0]));
		malloc_cpy_from_host((void**)&host_matrix.h_csc_col_ptrs, (void**)&d_csc_col_ptrs, (col_cnt + 1) * sizeof(host_matrix.h_csc_col_ptrs[0]));

		has_csc = true;
	}
	if(host_matrix.has_coo) {
		if(!has_csr) {
			malloc_cpy_from_host((void**)&host_matrix.h_csr_coo_vals, (void**)&d_csr_coo_vals, val_cnt * sizeof(host_matrix.h_csr_coo_vals[0]));
			malloc_cpy_from_host((void**)&host_matrix.h_csr_coo_col_idxs, (void**)&d_csr_coo_col_idxs, val_cnt * sizeof(host_matrix.h_csr_coo_col_idxs[0]));
		}
		malloc_cpy_from_host((void**)&host_matrix.h_coo_row_idxs, (void**)&d_coo_row_idxs, val_cnt * sizeof(host_matrix.h_coo_row_idxs[0]));

		has_coo = true;
	}
}

device_sparse_matrix::~device_sparse_matrix(){
	CUDA_CHECK(cudaFree(d_csr_coo_vals));
	CUDA_CHECK(cudaFree(d_csr_row_ptrs));
	CUDA_CHECK(cudaFree(d_csr_coo_col_idxs));
	CUDA_CHECK(cudaFree(d_coo_row_idxs));
	CUDA_CHECK(cudaFree(d_csc_vals));
	CUDA_CHECK(cudaFree(d_csc_row_idxs));
	CUDA_CHECK(cudaFree(d_csc_col_ptrs));

	CUSPARSE_CHECK(cusparseDestroyMatDescr(cusparse_descr));
}

