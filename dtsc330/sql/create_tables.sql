CREATE TABLE IF NOT EXISTS grants(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  application_id VARCHAR(20) NOT NULL,
  start_at DATE,
  grant_type VARCHAR(10),
  total_cost INTEGER
);