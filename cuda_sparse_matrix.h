#include <string>

struct cuda_sparse_matrix {
	cuda_sparse_matrix(size_t row_cnt, size_t col_cnt, size_t val_cnt);
	~cuda_sparse_matrix();

	void load_csr(std::string vals_path, std::string row_ptrs_path, std::string col_idxs_path);
	void load_csc(std::string vals_path, std::string row_idxs_path, std::string col_ptrs_path);
	void load_coo(std::string vals_path, std::string row_idxs_path, std::string col_idxs_path);

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

	float *d_csr_coo_vals = nullptr;
	int *d_csr_row_ptrs = nullptr;
	int *d_csr_coo_col_idxs = nullptr;

	int *d_coo_row_idxs = nullptr;

	float *d_csc_vals = nullptr;
	int *d_csc_row_idxs = nullptr;
	int *d_csc_col_ptrs = nullptr;
};
