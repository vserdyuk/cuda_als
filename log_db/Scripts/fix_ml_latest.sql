SELECT COUNT(DISTINCT userId)
FROM ml_latest_ratings;
-- 283228

SELECT MAX(userId)
FROM ml_latest_ratings;
-- 283228

SELECT COUNT(DISTINCT movieId)
FROM ml_latest_ratings;
-- 53889

SELECT MAX(movieId)
FROM ml_latest_ratings;
-- 193886 !!!

SELECT COUNT(*)
FROM ml_latest_ratings;
-- 27753444

WITH distinct_user_id as (
	SELECT DISTINCT userId
	FROM ml_latest_ratings
)
SELECT
	userId,
	ROW_NUMBER() OVER () as rownum
FROM distinct_user_id;


DROP TABLE ml_latest_users;
CREATE TABLE ml_latest_users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id_ml INTEGER);
INSERT INTO ml_latest_users(user_id_ml) SELECT DISTINCT userId FROM ml_latest_ratings ORDER BY userId;

DROP TABLE ml_latest_movies;
CREATE TABLE ml_latest_movies (movie_id INTEGER PRIMARY KEY AUTOINCREMENT, movie_id_ml INTEGER);
INSERT INTO ml_latest_movies(movie_id_ml) SELECT DISTINCT movieId FROM ml_latest_ratings ORDER BY movieId;

CREATE TABLE ml_latest_ratings_corrected AS
SELECT
	u.user_id AS userId,
	m.movie_id AS movieId,
	r.rating,
	r."timestamp"
FROM 
	ml_latest_ratings r
	LEFT JOIN ml_latest_users AS u
		ON r.userId = u.user_id_ml
	LEFT JOIN ml_latest_movies AS m
		ON r.movieId = m.movie_id_ml
;	
	
	
SELECT COUNT(DISTINCT userId)
FROM ml_latest_ratings_corrected;
-- 283228

SELECT MAX(userId)
FROM ml_latest_ratings_corrected;
-- 283228

SELECT COUNT(DISTINCT movieId)
FROM ml_latest_ratings_corrected;
-- 53889

SELECT MAX(movieId)
FROM ml_latest_ratings_corrected;
-- 53889 ok

SELECT COUNT(*)
FROM ml_latest_ratings_corrected;
-- 27753444

DROP TABLE ml_latest_ratings;
DROP TABLE ml_latest_users;
DROP TABLE ml_latest_movies;
DROP TABLE ml_latest_ratings_corrected;

VACUUM;
