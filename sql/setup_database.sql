-- Customer Segmentation & LTV Analysis - Database Setup Script
-- This script creates the database schema and imports CSV data

-- Create database (uncomment if needed)
-- CREATE DATABASE customer_segmentation;
-- USE customer_segmentation;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS rfm_scores;
DROP TABLE IF EXISTS customer_segments;
DROP TABLE IF EXISTS customer_clusters;
DROP TABLE IF EXISTS ltv_predictions;

-- Create customers table
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    city VARCHAR(50),
    customer_type VARCHAR(20),
    registration_date DATE
);

-- Create transactions table
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    transaction_date DATE,
    product_category VARCHAR(50),
    price DECIMAL(10,2),
    quantity INTEGER,
    payment_method VARCHAR(20),
    status VARCHAR(20),
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create RFM scores table
CREATE TABLE rfm_scores (
    customer_id INTEGER PRIMARY KEY,
    recency INTEGER,
    frequency INTEGER,
    monetary DECIMAL(10,2),
    recency_score INTEGER,
    frequency_score INTEGER,
    monetary_score INTEGER,
    rfm_score INTEGER,
    rfm_weighted_score DECIMAL(10,2),
    segment VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create customer segments table
CREATE TABLE customer_segments (
    customer_id INTEGER PRIMARY KEY,
    segment VARCHAR(20),
    segment_description TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create customer clusters table
CREATE TABLE customer_clusters (
    customer_id INTEGER PRIMARY KEY,
    cluster INTEGER,
    cluster_label VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create LTV predictions table
CREATE TABLE ltv_predictions (
    customer_id INTEGER PRIMARY KEY,
    actual_ltv DECIMAL(10,2),
    predicted_ltv DECIMAL(10,2),
    prediction_error DECIMAL(10,2),
    prediction_error_percent DECIMAL(5,2),
    model_used VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes for performance
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_category ON transactions(product_category);
CREATE INDEX idx_customers_type ON customers(customer_type);
CREATE INDEX idx_customers_city ON customers(city);
CREATE INDEX idx_rfm_segment ON rfm_scores(segment);
CREATE INDEX idx_clusters_cluster ON customer_clusters(cluster);

-- Import data from CSV files
-- Note: Adjust the file paths based on your system

-- Import customers
COPY customers FROM 'data/raw/customers_large.csv' WITH (FORMAT csv, HEADER true);

-- Import transactions
COPY transactions FROM 'data/raw/transactions_large.csv' WITH (FORMAT csv, HEADER true);

-- Import RFM scores
COPY rfm_scores FROM 'data/processed/rfm_scores.csv' WITH (FORMAT csv, HEADER true);

-- Import customer segments
COPY customer_segments FROM 'data/processed/customer_segments.csv' WITH (FORMAT csv, HEADER true);

-- Import customer clusters
COPY customer_clusters FROM 'data/processed/customer_clusters.csv' WITH (FORMAT csv, HEADER true);

-- Import LTV predictions
COPY ltv_predictions FROM 'data/processed/ltv_predictions.csv' WITH (FORMAT csv, HEADER true);

-- Verify data import
SELECT 'customers' as table_name, COUNT(*) as row_count FROM customers
UNION ALL
SELECT 'transactions' as table_name, COUNT(*) as row_count FROM transactions
UNION ALL
SELECT 'rfm_scores' as table_name, COUNT(*) as row_count FROM rfm_scores
UNION ALL
SELECT 'customer_segments' as table_name, COUNT(*) as row_count FROM customer_segments
UNION ALL
SELECT 'customer_clusters' as table_name, COUNT(*) as row_count FROM customer_clusters
UNION ALL
SELECT 'ltv_predictions' as table_name, COUNT(*) as row_count FROM ltv_predictions;

-- Sample queries to verify data integrity
SELECT 
    'Total Revenue' as metric,
    SUM(total_amount) as value
FROM transactions
UNION ALL
SELECT 
    'Average Transaction Value' as metric,
    AVG(total_amount) as value
FROM transactions
UNION ALL
SELECT 
    'Total Customers' as metric,
    COUNT(*) as value
FROM customers
UNION ALL
SELECT 
    'High-Value Customers' as metric,
    COUNT(*) as value
FROM rfm_scores
WHERE segment = 'High-Value';

-- Sample customer analysis query
SELECT 
    c.customer_type,
    COUNT(*) as customer_count,
    AVG(r.monetary) as avg_monetary,
    AVG(l.predicted_ltv) as avg_ltv
FROM customers c
JOIN rfm_scores r ON c.customer_id = r.customer_id
JOIN ltv_predictions l ON c.customer_id = l.customer_id
GROUP BY c.customer_type
ORDER BY avg_ltv DESC;
