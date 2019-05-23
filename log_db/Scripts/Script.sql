SELECT DISTINCT
	log_name
FROM log;


WITH u_started AS (
	SELECT
		log_name,
		run_iter,
		als_iter,
		curr_ts
	FROM log
	WHERE text LIKE "update U started"
),
u_done AS (
	SELECT
		log_name,
		run_iter,
		als_iter,
		curr_ts
	FROM log
	WHERE text LIKE "update U done"
)
SELECT
	u_started.log_name,
	u_started.run_iter,
	u_started.als_iter,
	u_started.curr_ts AS start_ts,
	u_done.curr_ts AS done_ts,
	u_done.curr_ts - u_started.curr_ts AS elapsed
FROM
	u_started
	LEFT JOIN u_done
		ON u_started.log_name = u_done.log_name
		AND u_started.run_iter = u_done.run_iter
		AND u_started.als_iter = u_done.als_iter
;

