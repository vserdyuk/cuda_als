#include <fstream>
#include <iostream>
#include "logger.h"

static inline double seconds(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_sec + (double)tv.tv_usec * 1.e-6);
}

static std::string get_date() {
	std::time_t now= std::time(0);
	std::tm* now_tm= std::gmtime(&now);
	char buf[42];
	std::strftime(buf, 42, "%Y%m%d_%H%M%S", now_tm);
	return buf;
}

void logger::init(const std::string &log_folder) {
	this->log_folder = log_folder;

	log_name = get_date();

#ifdef DEBUG
	log_name += "_debug=true";
#else
	log_name += "_debug=false";
#endif

	log_name += ".csv";

	prev_ts = seconds();
	curr_ts = prev_ts;
	ss << std::fixed;
	ss << "log_name,id,run_iter,als_iter,prev_ts,curr_ts,elapsed,text" << std::endl;
	log("init logger", false);
}

void logger::log(const std::string &text, bool write_cout) {
	prev_ts = curr_ts;
	curr_ts = seconds();

	double elapsed = curr_ts - prev_ts;

	ss << log_name << "," << next_id++ << "," << run_iter << "," << als_iter << "," << prev_ts << "," << curr_ts << "," << elapsed << "," << text << std::endl;

	if(write_cout) {
		std::cout << "run iter: " << run_iter << " als iter: " << als_iter << " elapsed: " << elapsed << " " << text << std::endl;
	}
}

void logger::save(const std::string &log_path) {
	std::ofstream out;
	out.open(log_path);
	out << ss.rdbuf();
}

void logger::save() {
	save(log_folder + "/" + log_name);
}
