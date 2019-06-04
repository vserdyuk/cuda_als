#ifndef ALS_MODEL_H
#define ALS_MODEL_H

#include <string>

#include <cusparse.h>
#include <cublas_v2.h>

#include "cuda_sparse_matrix.h"

struct als_model {
	enum class CALCULATE_VTVS_TYPE {
		SIMPLE = 0,
		SMEM_ROW_MAJOR,
		SMEM_COL_MAJOR,
		SMEM_ROW_MAJOR_NO_CALC,
		SMEM_COL_MAJOR_TWO_THREADS,
		SMEM_ROW_MAJOR_TENSOR,
		SMEM_ROW_MAJOR_TENSOR_SYMMETRIC,
		SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG
	};

	enum class SOLVE_TYPE {
		LU = 0,
		CUMF_ALS_CG_FP16,
		CUMF_ALS_CG_FP32
	};

	als_model(host_sparse_matrix &host_train_ratings, host_sparse_matrix &host_test_ratings, int f,
			float lambda, int iters, CALCULATE_VTVS_TYPE calculate_vtvs_type, SOLVE_TYPE solve_type,
			int smem_col_cnt, int m_batches, int n_batches
	);
	~als_model();

	void LU_solve_U(int m_batch_size, int m_batch_offset);
	void LU_solve_V(int n_batch_size, int n_batch_offset);

	void train();

	float rsme_train();
	float rsme_test();

	device_sparse_matrix train_ratings;
	device_sparse_matrix test_ratings;

	int m;			// number of users
	int n;			// number of items

	int m_batches;
	int n_batches;

	int f;			// number of factors
	float lambda;
	int iters;

	// dense matrices, column-major

	float *h_VT;	// host transposed global item factor matrix, n x f (thetaTHost)
	float *d_VT;	// device transposed global item factor matrix, n x f (thetaT)
	float *d_VTRT;	// device transposed global item factor matrix multiplied by transposed ratings, f x m (confusing name ythetaT, IMHO thetaTyT is clearer)

	float *h_UT;	// host transposed global user factor matrix, m x f (XTHost)
	float *d_UT;	// device transposed global user factor matrix, m x f (XT)
	float *d_UTR;	// device transposed global user factor matrix multiplied by ratings, f x n (confusing name yTXT, IMHO XTy is clearer)

	float *d_xtxs;	// device multiple vvt or uut regularized factor matrices each for single user, vvt (f x f) * n_first_batch_size or uut (f x f) * m_first_batch_size

	cusparseHandle_t cusparse_handle;
	cublasHandle_t cublas_handle;

	CALCULATE_VTVS_TYPE calculate_vtvs_type;
	SOLVE_TYPE solve_type;

	int smem_col_cnt;

	static std::string to_string(CALCULATE_VTVS_TYPE calculate_vtvs_type);
	static std::string to_string(SOLVE_TYPE solve_type);
};

#endif // ALS_MODEL_H
