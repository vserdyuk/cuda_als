#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <sstream>

struct logger {
	enum class EVENT_TYPE {
		ALS_UPDATE_U = 0,
		ALS_UPDATE_V,
		ALS_ITER
	};

	void init(const std::string &log_folder);

	void log(const std::string &text, bool write_cout);

	void event_started(EVENT_TYPE event_type);
	void event_finished(EVENT_TYPE event_type, bool write_cout);

	void save();
	void save(const std::string &log_path);

	double prev_ts = 0;

	double als_update_u_started_ts = 0;

	double als_update_v_started_ts = 0;

	double als_iter_started_ts = 0;

	std::string log_folder;
	std::string log_name;

	std::stringstream ss;

	size_t next_id = 0;

	size_t run_iter = 0;
	size_t als_iter = 0;

	static std::string to_string(EVENT_TYPE event_type);
};

extern logger g_logger;

#endif // LOGGER_H
