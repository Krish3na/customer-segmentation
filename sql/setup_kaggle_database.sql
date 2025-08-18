-- Customer Segmentation & LTV Analysis - Kaggle Database Setup Script
-- This script sets up the database and imports Kaggle dataset data

-- Create database (run this separately if needed)
-- CREATE DATABASE customer_segmentation_kaggle;

-- Connect to the database
\c customer_segmentation_kaggle;

-- Create tables for Kaggle data
CREATE TABLE IF NOT EXISTS kaggle_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    age INTEGER,
    annual_income_k DECIMAL(10,2),
    spending_score_1_100 INTEGER,
    customer_city VARCHAR(100),
    customer_state VARCHAR(50),
    country VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS kaggle_transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    transaction_date DATE,
    total_amount DECIMAL(10,2),
    product_category VARCHAR(100),
    status VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES kaggle_customers(customer_id)
);

CREATE TABLE IF NOT EXISTS kaggle_rfm_scores (
    customer_id VARCHAR(50) PRIMARY KEY,
    recency INTEGER,
    frequency INTEGER,
    monetary DECIMAL(10,2),
    R_score INTEGER,
    F_score INTEGER,
    M_score INTEGER,
    RFM_score INTEGER,
    segment VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES kaggle_customers(customer_id)
);

CREATE TABLE IF NOT EXISTS kaggle_customer_clusters (
    customer_id VARCHAR(50) PRIMARY KEY,
    cluster INTEGER,
    recency INTEGER,
    frequency INTEGER,
    monetary DECIMAL(10,2),
    RFM_score INTEGER,
    FOREIGN KEY (customer_id) REFERENCES kaggle_customers(customer_id)
);

CREATE TABLE IF NOT EXISTS kaggle_ltv_predictions (
    customer_id VARCHAR(50) PRIMARY KEY,
    predicted_ltv DECIMAL(12,2),
    actual_ltv DECIMAL(12,2),
    ltv_difference DECIMAL(12,2),
    ltv_accuracy DECIMAL(5,4),
    segment VARCHAR(20),
    RFM_score INTEGER,
    FOREIGN KEY (customer_id) REFERENCES kaggle_customers(customer_id)
);

-- Import data from CSV files (adjust paths as needed)
-- Note: Make sure the CSV files are accessible to PostgreSQL

-- Import customers data
\copy kaggle_customers FROM 'data/kaggle/raw/kaggle_retail_customers.csv' WITH (FORMAT csv, HEADER true);

-- Import transactions data
\copy kaggle_transactions(customer_id, transaction_date, total_amount, product_category, status) FROM 'data/kaggle/raw/kaggle_retail_transactions.csv' WITH (FORMAT csv, HEADER true);

-- Import RFM scores
\copy kaggle_rfm_scores FROM 'data/kaggle/processed/kaggle_rfm_scores.csv' WITH (FORMAT csv, HEADER true);

-- Import customer clusters
\copy kaggle_customer_clusters FROM 'data/kaggle/processed/kaggle_customer_clusters.csv' WITH (FORMAT csv, HEADER true);

-- Import LTV predictions
\copy kaggle_ltv_predictions FROM 'data/kaggle/processed/kaggle_ltv_predictions.csv' WITH (FORMAT csv, HEADER true);

-- Create indexes for better performance
CREATE INDEX idx_kaggle_transactions_customer_id ON kaggle_transactions(customer_id);
CREATE INDEX idx_kaggle_transactions_date ON kaggle_transactions(transaction_date);
CREATE INDEX idx_kaggle_rfm_segment ON kaggle_rfm_scores(segment);
CREATE INDEX idx_kaggle_clusters_cluster ON kaggle_customer_clusters(cluster);

-- Display table statistics
SELECT 
    'kaggle_customers' as table_name,
    COUNT(*) as row_count
FROM kaggle_customers
UNION ALL
SELECT 
    'kaggle_transactions' as table_name,
    COUNT(*) as row_count
FROM kaggle_transactions
UNION ALL
SELECT 
    'kaggle_rfm_scores' as table_name,
    COUNT(*) as row_count
FROM kaggle_rfm_scores
UNION ALL
SELECT 
    'kaggle_customer_clusters' as table_name,
    COUNT(*) as row_count
FROM kaggle_customer_clusters
UNION ALL
SELECT 
    'kaggle_ltv_predictions' as table_name,
    COUNT(*) as row_count
FROM kaggle_ltv_predictions
ORDER BY table_name;
