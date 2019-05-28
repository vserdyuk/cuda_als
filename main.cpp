#include <string>
#include <iostream>

#include "cuda_runtime.h"

#include "cuda_sparse_matrix.h"
#include "als_model.h"

#ifdef USE_LOGGER
#include "logger.h"
logger g_logger;
#endif

#define CUDA_DEVICE_ID 0


int main(int argc, char **argv) {
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	//int f = atoi(argv[3]);
	//int f = 80;
	int f = 64;
	long nnz_train = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	float lambda = atof(argv[6]);
	int als_iters = atoi(argv[7]);
	std::string data_folder = argv[8];
	int als_runs = atoi(argv[9]);
	//als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = static_cast<als_model::CALCULATE_VVTS_TYPE>(atoi(argv[10]));
	//als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = als_model::CALCULATE_VVTS_TYPE::SIMPLE;	// easier debugging
	//als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = als_model::CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR;	// easier debugging
	//als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = als_model::CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_NO_CALC;	// easier debugging
	//als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = als_model::CALCULATE_VVTS_TYPE::SMEM_COL_MAJOR;	// easier debugging
	als_model::CALCULATE_VVTS_TYPE als_calculate_vvts_type = als_model::CALCULATE_VVTS_TYPE::SMEM_ROW_MAJOR_TENSOR;	// easier debugging
	//int smem_col_cnt = 112;	// shoud be calculated based on device shared memory size
	int smem_col_cnt = 32;	// shoud be calculated based on device shared memory size
	//int smem_col_cnt = 112;	// shoud be calculated based on device shared memory size

#ifdef USE_LOGGER
	std::string log_folder = argv[11];
#endif

	std::cout << std::fixed;

#ifdef USE_LOGGER
	g_logger.init(log_folder);
#endif

	std::stringstream ss;

	ss << "m=" << m << " n=" << n << " f=" << f << " nnz_train=" << nnz_train << " nnz_test=" << nnz_test << " lambda=" << lambda
			<< " als_iters=" << als_iters << " data_folder=" << data_folder << " als_runs=" << als_runs << " log_folder=" << log_folder
			<< " als_calculate_vvts_type=" << als_model::to_string(als_calculate_vvts_type) << " smem_col_cnt=" << smem_col_cnt
	;

	g_logger.log(ss.str(), true);

	cudaSetDevice(CUDA_DEVICE_ID);

	cuda_sparse_matrix train_ratings(m, n, nnz_train);
	train_ratings.load_csc(data_folder + "/R_train_csc.data.bin", data_folder + "/R_train_csc.indices.bin", data_folder + "/R_train_csc.indptr.bin");
	train_ratings.load_csr(data_folder + "/R_train_csr.data.bin", data_folder + "/R_train_csr.indptr.bin", data_folder + "/R_train_csr.indices.bin");
	train_ratings.load_coo(data_folder + "/R_train_csr.data.bin", data_folder + "/R_train_coo.row.bin", data_folder + "/R_train_csr.indices.bin");

	cuda_sparse_matrix test_ratings(m, n, nnz_test);

	test_ratings.load_coo(data_folder + "/R_test_coo.data.bin", data_folder + "/R_test_coo.row.bin", data_folder + "/R_test_coo.col.bin");

	als_model model(train_ratings, test_ratings, f, lambda, als_iters, als_calculate_vvts_type, smem_col_cnt);

#ifdef USE_LOGGER
	g_logger.log("als model constructor done", true);
#endif

	// run in loop to measure performance
	for(size_t i = 0; i < als_runs; ++i) {

#ifdef USE_LOGGER
		g_logger.run_iter = i + 1;
#endif
		model.train();
	}

#ifdef USE_LOGGER
	g_logger.run_iter = 0;

	g_logger.log("all als runs done", true);

	g_logger.save();
#endif
}
