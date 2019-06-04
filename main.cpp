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
	//int f = 32;
	//int f = 64;
	int f = 80;
	//int f = 96;
	//int f = 128;
	long nnz_train = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	float lambda = atof(argv[6]);
	int als_iters = atoi(argv[7]);
	std::string data_folder = argv[8];
	int als_runs = atoi(argv[9]);

	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = static_cast<als_model::CALCULATE_VTVS_TYPE>(atoi(argv[10]));
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SIMPLE;	// easier debugging
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_ROW_MAJOR;	// easier debugging
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_ROW_MAJOR_NO_CALC;	// easier debugging
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_COL_MAJOR;	// easier debugging
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_ROW_MAJOR_TENSOR;	// easier debugging
	//als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC;	// easier debugging
	als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = als_model::CALCULATE_VTVS_TYPE::SMEM_ROW_MAJOR_TENSOR_SYMMETRIC_MULT_FRAG;	// easier debugging

	//als_model::SOLVE_TYPE als_solve_type = static_cast<als_model::SOLVE_TYPE>(atoi(argv[11]));
	//als_model::SOLVE_TYPE als_solve_type = als_model::SOLVE_TYPE::LU;	// easier debugging
	als_model::SOLVE_TYPE als_solve_type = als_model::SOLVE_TYPE::CUMF_ALS_CG_FP32;	// easier debugging

	int smem_col_cnt = 32;	// shoud be calculated based on device shared memory sized
	//int smem_col_cnt = 112;	// shoud be calculated based on device shared memory size

	int m_batches = atoi(argv[12]);
	int n_batches = atoi(argv[13]);

#ifdef USE_LOGGER
	std::string log_folder = argv[14];

	std::cout << std::fixed;

	g_logger.init(log_folder);
	std::stringstream ss;

	ss << "m=" << m << " n=" << n << " f=" << f << " nnz_train=" << nnz_train << " nnz_test=" << nnz_test << " lambda=" << lambda
			<< " als_iters=" << als_iters << " data_folder=" << data_folder << " als_runs=" << als_runs << " log_folder=" << log_folder
			<< " als_calculate_vtvs_type=" << als_model::to_string(als_calculate_vtvs_type) << " als_solve_type=" << als_model::to_string(als_solve_type)
			<< " smem_col_cnt=" << smem_col_cnt
	;

	g_logger.log(ss.str(), true);
#endif

	cudaSetDevice(CUDA_DEVICE_ID);

	host_sparse_matrix host_train_ratings(m, n, nnz_train);
	host_train_ratings.load_csc(data_folder + "/R_train_csc.data.bin", data_folder + "/R_train_csc.indices.bin", data_folder + "/R_train_csc.indptr.bin");
	host_train_ratings.load_csr(data_folder + "/R_train_csr.data.bin", data_folder + "/R_train_csr.indptr.bin", data_folder + "/R_train_csr.indices.bin");
	host_train_ratings.load_coo(data_folder + "/R_train_csr.data.bin", data_folder + "/R_train_coo.row.bin", data_folder + "/R_train_csr.indices.bin");

	host_sparse_matrix host_test_ratings(m, n, nnz_test);

	host_test_ratings.load_coo(data_folder + "/R_test_coo.data.bin", data_folder + "/R_test_coo.row.bin", data_folder + "/R_test_coo.col.bin");

#ifdef USE_LOGGER
	g_logger.log("host train and test ratings loaded", true);
#endif

	// run in loop to measure performance
	for(size_t i = 0; i < als_runs; ++i) {

#ifdef USE_LOGGER
		// MF_MODEL_TRAINING event is meant for comparing performance with cumf_als and cumf_sgd
		// started after host loading train and test ratings from disk, before first cudaMemcpy to device
		g_logger.event_started(logger::EVENT_TYPE::MF_MODEL_TRAINING);

		g_logger.run_iter = i + 1;
#endif

		als_model model(host_train_ratings, host_test_ratings, f, lambda, als_iters, als_calculate_vtvs_type, als_solve_type, smem_col_cnt, m_batches, n_batches);

#ifdef USE_LOGGER
	g_logger.log("als model constructor done", true);
#endif

		model.train();

#ifdef USE_LOGGER
		// MF_MODEL_TRAINING event finishes after model (user and item factors) calculation before its saving to disk
		g_logger.event_finished(logger::EVENT_TYPE::MF_MODEL_TRAINING, true);

		//g_logger.log("final RMSE train: " + std::to_string(model.rsme_train()), true);
		g_logger.log("final RMSE test: " + std::to_string(model.rsme_test()), true);
#endif

	}

#ifdef USE_LOGGER
	g_logger.run_iter = 0;

	g_logger.log("all als runs done", true);

	g_logger.save();
#endif
}
