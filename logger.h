#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <sstream>
#include <ctime>

#include <sys/time.h>

struct logger {
	void init(const std::string &log_folder);

	void log(const std::string &text, bool write_cout);

	void save();
	void save(const std::string &log_path);

	double prev_ts;
	double curr_ts;

	std::string log_folder;
	std::string log_name;

	std::stringstream ss;

	size_t next_id = 0;

	size_t run_iter = 0;
	size_t als_iter = 0;
};

extern logger g_logger;

#endif // LOGGER_H
