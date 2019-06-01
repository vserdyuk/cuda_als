SELECT DISTINCT
	log_name
FROM ml10M_log;

CREATE TABLE ml10M_log AS SELECT * FROM toy_data_log WHERE 1 = 2; 

DELETE FROM ml10M_log;

SELECT
	log_name,
	text,
	AVG(elapsed) as avg_elapsed
FROM ml10M_log
WHERE text LIKE "% calculation done type%"
GROUP BY
	log_name,
	text
;

SELECT
	log_name,
	text,
	AVG(elapsed) as avg_elapsed
FROM ml10M_log
WHERE text LIKE "%batched LU factorization done%"
GROUP BY
	log_name,
	text
;

SELECT
	log_name,
	text,
	AVG(elapsed) as avg_elapsed
FROM ml10M_log
WHERE text LIKE "% batched solve done%"
GROUP BY
	log_name,
	text
;

SELECT
--	ml10M_log.log_name,
	COUNT(DISTINCT log_name) AS log_name_distinct_cnt,
	COUNT(run_iter) AS run_iter_cnt,
	COUNT(als_iter) AS als_iter_cnt,
	"text",
	AVG(event_elapsed) AS event_elapsed_avg
FROM ml10M_log
WHERE LENGTH(event_elapsed) > 0
GROUP BY
--	ml10M_log.log_name,
	"text"
;

DROP TABLE ml10M_log;


SELECT
--	COUNT(DISTINCT ml10M_log.log_name) AS log_name_distinct_cnt,
	ml10M_log.log_name,	
	COUNT(run_iter) AS run_iter_cnt,
	COUNT(als_iter) AS als_iter_cnt,
	text,
	AVG(elapsed) as elapsed_avg
FROM ml10M_log
WHERE
	text LIKE "%via cuSPARSE and cuBLAS done%"
	OR text LIKE "%calculation done type=%"
	OR text LIKE "%batched LU factorization%"
	OR text LIKE "%batched solve done%"
GROUP BY
	text,
	ml10M_log.log_name
;




















