#include <string>
#include <iostream>

#include "cuda_runtime.h"

#include "cuda_sparse_matrix.h"
#include "als_model.h"

#define CUDA_DEVICE_ID 0

int main(int argc, char **argv) {
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int f = atoi(argv[3]);
	long nnz_train = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	float lambda = atof(argv[6]);
	int iters = atoi(argv[7]);
	std::string data_path = argv[8];

	cudaSetDevice(CUDA_DEVICE_ID);

	cuda_sparse_matrix train_ratings(m, n, nnz_train);
	train_ratings.load_csc(data_path + "/R_train_csc.data.bin", data_path + "/R_train_csc.indices.bin", data_path + "/R_train_csc.indptr.bin");
	train_ratings.load_csr(data_path + "/R_train_csr.data.bin", data_path + "/R_train_csr.indptr.bin", data_path + "/R_train_csr.indices.bin");
	train_ratings.load_coo(data_path + "/R_train_csr.data.bin", data_path + "/R_train_coo.row.bin", data_path + "/R_train_csr.indices.bin");

	cuda_sparse_matrix test_ratings(m, n, nnz_test);

	test_ratings.load_coo(data_path + "/R_test_coo.data.bin", data_path + "/R_test_coo.row.bin", data_path + "/R_test_coo.col.bin");

	als_model model(train_ratings, test_ratings, f, lambda, iters);
	model.train();
}
