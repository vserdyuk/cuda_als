#include <string>

#include "cuda_runtime.h"
#include "cuda_sparse_matrix.h"
#include "iostream"

#define CUDA_DEVICE_ID 0

int main(int argc, char **argv) {
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int f = atoi(argv[3]);
	long nnz_train = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	float lambda = atof(argv[6]);
	std::string data_path = argv[7];

	cudaSetDevice(CUDA_DEVICE_ID);

	cuda_sparse_matrix test(m, n, nnz_test);

	test.load_coo(data_path + "/R_test_coo.data.bin", data_path + "/R_test_coo.row.bin", data_path + "/R_test_coo.col.bin");

	cuda_sparse_matrix train(m, n, nnz_train);
	train.load_csc(data_path + "/R_train_csc.data.bin", data_path + "/R_train_csc.indices.bin", data_path + "/R_train_csc.indptr.bin");
	train.load_csr(data_path + "/R_train_csr.data.bin", data_path + "/R_train_csr.indptr.bin", data_path + "/R_train_csr.indices.bin");
	train.load_coo(data_path + "/R_train_csr.data.bin", data_path + "/R_train_coo.row.bin", data_path + "/R_train_csr.indices.bin");
}
