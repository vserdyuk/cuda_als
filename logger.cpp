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
	ss << std::fixed;
	ss << "log_name,id,run_iter,als_iter,prev_ts,curr_ts,elapsed,event_elapsed,text" << std::endl;
	log("init logger", false);
}

void logger::log(const std::string &text, bool write_cout) {
	double curr_ts = seconds();

	double elapsed = curr_ts - prev_ts;

	prev_ts = curr_ts;

	ss << log_name << "," << next_id++ << "," << run_iter << "," << als_iter << "," << prev_ts << "," << curr_ts << "," << elapsed << "," << "," << text << std::endl;

	if(write_cout) {
		std::cout << "run iter: " << run_iter << " als iter: " << als_iter << " elapsed: " << elapsed << " " << text << std::endl;
	}
}

void logger::event_started(EVENT_TYPE event_type) {
	switch(event_type) {
		case EVENT_TYPE::ALS_UPDATE_U:
			als_update_u_started_ts = seconds();
			break;
		case EVENT_TYPE::ALS_UPDATE_V:
			als_update_v_started_ts = seconds();
			break;
		case EVENT_TYPE::ALS_ITER:
			als_iter_started_ts = seconds();
			break;
		default:
			break;
	}
}

void logger::event_finished(EVENT_TYPE event_type, bool write_cout) {
	double curr_ts = seconds();

	double log_elapsed = curr_ts - prev_ts;

	double event_elapsed = 0;

	switch(event_type) {
		case EVENT_TYPE::ALS_UPDATE_U:
			event_elapsed = curr_ts - als_update_u_started_ts;
			als_update_u_started_ts = 0;
			break;
		case EVENT_TYPE::ALS_UPDATE_V:
			event_elapsed = curr_ts - als_update_v_started_ts;
			als_update_v_started_ts = 0;
			break;
		case EVENT_TYPE::ALS_ITER:
			event_elapsed = curr_ts - als_iter_started_ts;
			als_update_v_started_ts = 0;
			break;
		default:
			break;
	}

	std::string text = "event " + to_string(event_type) + " finished";

	ss << log_name << "," << next_id++ << "," << run_iter << "," << als_iter << "," << prev_ts << "," << curr_ts << "," << log_elapsed << "," << event_elapsed << "," << text << std::endl;

	if(write_cout) {
		std::cout << "run iter: " << run_iter << " als iter: " << als_iter << " elapsed: " << log_elapsed << " event elapsed: " << event_elapsed << " " << text << std::endl;
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

std::string logger::to_string(EVENT_TYPE event_type) {
	switch(event_type) {
		case EVENT_TYPE::ALS_UPDATE_U: return "ALS_UPDATE_U";
		case EVENT_TYPE::ALS_UPDATE_V: return "ALS_UPDATE_V";
		case EVENT_TYPE::ALS_ITER: return "ALS_ITER";
		default: return "UNKNOWN";
	}
}
