#ifndef ALS_MODEL_H
#define ALS_MODEL_H

#include <cusparse.h>
#include <cublas_v2.h>

#include "cuda_sparse_matrix.h"

struct als_model {
	als_model(cuda_sparse_matrix &train_ratings, cuda_sparse_matrix &test_ratings, int f, float lambda, int iters);
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
	float *d_RVT;	// device ratings multiplied by transposed global item factor matrix, f x m (ythetaT)

	float *h_U;		// host global user factor matrix, m x f (XTHost(XHost?))
	float *d_U;		// device global user factor matrix, m x f (XT(X?))
	float *d_RTUT;	// device transposed ratings multiplied by transposed global user factor matrix f x n (yTXT)

	float *d_vvts;	// device multiple vvt factor matrices each for single item, (f x f) * n (tt)
	float *d_uuts;	// device multiple uut factor matrices each for single user, (f x f) * m (xx)

	cusparseHandle_t cusparse_handle;
	cublasHandle_t cublas_handle;
};

#endif // ALS_MODEL_H
