SELECT COUNT(DISTINCT UserID)
FROM ml10m_ratings;
-- 69878

SELECT MAX(UserID)
FROM ml10m_ratings;
-- 71567 !!!

SELECT COUNT(DISTINCT MovieID)
FROM ml10m_ratings;
-- 10677

SELECT MAX(MovieID)
FROM ml10m_ratings;
-- 65133 !!!

SELECT COUNT(*)
FROM ml10m_ratings;
-- 10000054

DROP TABLE ml10m_users;
CREATE TABLE ml10m_users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id_ml INTEGER);
INSERT INTO ml10m_users(user_id_ml) SELECT DISTINCT UserID FROM ml10m_ratings ORDER BY UserID;

DROP TABLE ml10m_movies;
CREATE TABLE ml10m_movies (movie_id INTEGER PRIMARY KEY AUTOINCREMENT, movie_id_ml INTEGER);
INSERT INTO ml10m_movies(movie_id_ml) SELECT DISTINCT MovieID FROM ml10m_ratings ORDER BY MovieID;

CREATE TABLE ml10m_ratings_corrected AS
SELECT
	u.user_id AS UserID,
	m.movie_id AS MovieID,
	r.Rating,
	r."Timestamp"
FROM 
	ml10m_ratings r
	LEFT JOIN ml10m_users AS u
		ON r.UserID = u.user_id_ml
	LEFT JOIN ml10m_movies AS m
		ON r.MovieID = m.movie_id_ml
;	
	
	
SELECT COUNT(DISTINCT UserID)
FROM ml10m_ratings_corrected;
-- 69878

SELECT MAX(UserID)
FROM ml10m_ratings_corrected;
-- 69878 ok

SELECT COUNT(DISTINCT MovieID)
FROM ml10m_ratings_corrected;
-- 10677

SELECT MAX(MovieID)
FROM ml10m_ratings_corrected;
-- 10677 ok

SELECT COUNT(*)
FROM ml10m_ratings_corrected;
-- 10000054

DROP TABLE ml10m_ratings;
DROP TABLE ml10m_users;
DROP TABLE ml10m_movies;
DROP TABLE ml10m_ratings_corrected;

VACUUM;
