#include <string>
#include <iostream>

#include "cuda_runtime.h"

#include "cuda_sparse_matrix.h"
#include "als_model.h"

#include "INIReader.h"

#ifdef USE_LOGGER
#include "logger.h"
logger g_logger;
#endif

#define CUDA_DEVICE_ID 0

int main(int argc, char **argv) {
	std::string ini_path = argv[1];

	INIReader ini_reader(ini_path);

	if (ini_reader.ParseError() != 0) {
		std::cout << "Can't load " << ini_path << "\n";
		return 1;
	}

	int m = ini_reader.GetInteger("data", "m", 0);
	int n = ini_reader.GetInteger("data", "n", 0);
	int f = ini_reader.GetInteger("als", "f", 0);

	long nnz_train = ini_reader.GetInteger("data", "nnz_train", 0);
	long nnz_test = ini_reader.GetInteger("data", "nnz_test", 0);
	float lambda = ini_reader.GetFloat("als", "lambda", 0);
	int als_iters = ini_reader.GetInteger("als", "als_iters", 0);
	std::string data_folder = ini_reader.Get("data", "data_folder", "");
	int als_runs = ini_reader.GetInteger("general", "als_runs", 0);

	als_model::CALCULATE_VTVS_TYPE als_calculate_vtvs_type = static_cast<als_model::CALCULATE_VTVS_TYPE>(ini_reader.GetInteger("als", "als_calculate_vtvs_type", 0));

	als_model::SOLVE_TYPE als_solve_type = static_cast<als_model::SOLVE_TYPE>(ini_reader.GetInteger("als", "als_solve_type", 0));

	int smem_col_cnt = 32;	// shoud be calculated based on device shared memory sized
	//int smem_col_cnt = 112;	// shoud be calculated based on device shared memory size

	int m_batches = ini_reader.GetInteger("als", "m_batches", 0);
	int n_batches = ini_reader.GetInteger("als", "n_batches", 0);

#ifdef USE_LOGGER
	std::string log_folder = ini_reader.Get("general", "log_folder", "");

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

		if(als_runs == 1) {
			//model.save_libmf("/home/vladimir/src/cuda_als/data/netflix/netflix.model");
			model.save_libmf("/home/vladimir/src/cumf_sgd/data/netflix/netflix_train.bin.model");

			g_logger.log("saved model in cumf_sgd libmf format", true);
		}
	}

#ifdef USE_LOGGER
	g_logger.run_iter = 0;

	g_logger.log("all als runs done", true);

	g_logger.save();
#endif
}
