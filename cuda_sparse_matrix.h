#ifndef CUDA_SPARSE_MATRIX
#define CUDA_SPARSE_MATRIX

#include <string>

#include <cusparse.h>

struct host_sparse_matrix {
	host_sparse_matrix(size_t row_cnt, size_t col_cnt, size_t val_cnt);
	~host_sparse_matrix();

	// TODO: read .mtx to COO and convert to CSR using coo2csr (if needed, transpose CSR to get CSC)
	// TODO: look at new cusparseSpMat

	void load_csr(std::string vals_path, std::string row_ptrs_path, std::string col_idxs_path);
	void load_csc(std::string vals_path, std::string row_idxs_path, std::string col_ptrs_path);
	void load_coo(std::string vals_path, std::string row_idxs_path, std::string col_idxs_path);

	static void malloc_load_file(std::string path, void **host_ptr, size_t buf_size);

	bool has_coo = false;
	bool has_csr = false;
	bool has_csc = false;

	size_t row_cnt = 0;
	size_t col_cnt = 0;
	size_t val_cnt = 0;

	float *h_csr_coo_vals = nullptr;
	int *h_csr_row_ptrs = nullptr;
	int *h_csr_coo_col_idxs = nullptr;

	int *h_coo_row_idxs = nullptr;

	float *h_csc_vals = nullptr;
	int *h_csc_row_idxs = nullptr;
	int *h_csc_col_ptrs = nullptr;
};

struct device_sparse_matrix {
	device_sparse_matrix(host_sparse_matrix &host_matrix);
	~device_sparse_matrix();

	static void malloc_cpy_from_host(void **host_ptr, void **dev_ptr, size_t buf_size);

	bool has_coo = false;
	bool has_csr = false;
	bool has_csc = false;

	size_t row_cnt = 0;
	size_t col_cnt = 0;
	size_t val_cnt = 0;

	float *d_csr_coo_vals = nullptr;
	int *d_csr_row_ptrs = nullptr;
	int *d_csr_coo_col_idxs = nullptr;

	int *d_coo_row_idxs = nullptr;

	float *d_csc_vals = nullptr;
	int *d_csc_row_idxs = nullptr;
	int *d_csc_col_ptrs = nullptr;

	// there are cusparseXcoosort() and cusparseXcscsort() which take cusparseMatDescr_t but it is mostly used for CSR
	// so we assume we don't need separate descriptors for COO and CSC

	cusparseMatDescr_t cusparse_descr;
};

#endif // CUDA_SPARSE_MATRIX
