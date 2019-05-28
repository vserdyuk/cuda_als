SELECT DISTINCT
	log_name
FROM ml10M_log;

CREATE TABLE ml10M_log AS SELECT * FROM toy_data_log WHERE 1 = 2; 

WITH started AS (
	SELECT
		log_name,
		run_iter,
		als_iter,
		curr_ts,
		CASE WHEN text LIKE "update U started" THEN "U" ELSE "V" END AS factor,
		text
	FROM ml10M_log
	WHERE 
		text LIKE "update U started"
		OR text LIKE "update V started"
),
done AS (
	SELECT
		log_name,
		run_iter,
		als_iter,
		curr_ts,
		CASE WHEN text LIKE "update U done" THEN "U" ELSE "V" END AS factor
	FROM ml10M_log
	WHERE 
		text LIKE "update U done"
		OR text LIKE "update V done"
),
smem_config AS (
	SELECT
		log_name,
		run_iter,
		als_iter,
		CASE WHEN text LIKE "vtvs smem%" THEN "U" ELSE "V" END AS factor,
		text as smem_config
	FROM ml10M_log
	WHERE
		text LIKE "vtvs smem%"
		OR text LIKE "utus smem%"
),
elapsed as (
SELECT
	started.log_name,
	started.run_iter,
	started.als_iter,
	started.factor,
	started.curr_ts AS start_ts,
	done.curr_ts AS done_ts,
	done.curr_ts - started.curr_ts AS elapsed
FROM
	started
	LEFT JOIN done
		ON started.log_name = done.log_name
		AND started.run_iter = done.run_iter
		AND started.als_iter = done.als_iter
		AND started.factor = done.factor
), 
avg_elapsed AS ( 
SELECT
	log_name,
	factor,
	AVG(elapsed) as avg_elapsed
FROM elapsed
GROUP BY
	log_name,
	factor
)
SELECT
	avg_elapsed.log_name,
	avg_elapsed.factor,
	avg_elapsed.avg_elapsed,
	smem_config_distinct.smem_config,
	main_params.text
FROM
	avg_elapsed
	LEFT JOIN 
	(
		SELECT
			log_name,
			text
		FROM ml10M_log
		WHERE id = 1
	) AS main_params
		ON avg_elapsed.log_name = main_params.log_name
	LEFT JOIN (
		SELECT DISTINCT
			log_name,
			factor,
			smem_config
		FROM smem_config
	) AS smem_config_distinct
		ON avg_elapsed.log_name = smem_config_distinct.log_name
		AND avg_elapsed.factor = smem_config_distinct.factor
;

SELECT DISTINCT * FROM toy_data_log; 


SELECT COUNT(DISTINCT movieId)
FROM "ml-latest_ratings";
-- 53889

SELECT COUNT(DISTINCT userId)
FROM "ml-latest_ratings";
-- 283228

SELECT COUNT(*)
FROM "ml-latest_ratings";
-- 27753444

SELECT MAX(movieId)
FROM "ml-latest_ratings";

WITH distinct_user_id as (
	SELECT DISTINCT userId
	FROM "ml-latest_ratings"
)
SELECT
	userId,
	_rowid_ as row_id
FROM distinct_user_id;




