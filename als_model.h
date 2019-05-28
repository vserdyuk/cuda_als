#ifndef ALS_MODEL_H
#define ALS_MODEL_H

#include <string>

#include <cusparse.h>
#include <cublas_v2.h>

#include "cuda_sparse_matrix.h"

struct als_model {
	enum class CALCULATE_VVTS_TYPE {
		SIMPLE = 0,
		SMEM_ROW_MAJOR,
		SMEM_COL_MAJOR,
		SMEM_ROW_MAJOR_NO_CALC,
		SMEM_COL_MAJOR_TWO_THREADS,
		SMEM_ROW_MAJOR_TENSOR,
		SMEM_ROW_MAJOR_TENSOR_SYMMETRIC
	};

	als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f, float lambda, int iters, CALCULATE_VVTS_TYPE calculate_vvts_type, int smem_col_cnt);
	~als_model();

	void train();

	cuda_sparse_matrix &train_ratings;
	cuda_sparse_matrix &test_ratings;

	int m;			// number of users
	int n;			// number of items

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

	float *d_vtvs;	// device multiple vvt regularized factor matrices each for single item, (f x f) * n (tt)
	float *d_utus;	// device multiple uut regularized factor matrices each for single user, (f x f) * m (xx)

	cusparseHandle_t cusparse_handle;
	cublasHandle_t cublas_handle;

	CALCULATE_VVTS_TYPE calculate_vvts_type;

	int smem_col_cnt;

	static std::string to_string(CALCULATE_VVTS_TYPE calculate_vvts_type);
};

#endif // ALS_MODEL_H
