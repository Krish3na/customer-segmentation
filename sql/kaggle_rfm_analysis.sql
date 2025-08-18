-- Customer Segmentation & LTV Analysis - Kaggle RFM Analysis
-- Comprehensive SQL analysis for Kaggle datasets

-- 1. CUSTOMER DEMOGRAPHICS ANALYSIS
-- =================================

-- Customer age distribution (Optimized)
SELECT 
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 44 THEN '35-44'
        WHEN age BETWEEN 45 AND 54 THEN '45-54'
        WHEN age >= 55 THEN '55+'
    END as age_group,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM kaggle_customers WHERE age IS NOT NULL), 2) as percentage
FROM kaggle_customers 
WHERE age IS NOT NULL
GROUP BY age_group
ORDER BY age_group;

-- Gender distribution (Optimized)
SELECT 
    gender,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM kaggle_customers WHERE gender IS NOT NULL), 2) as percentage
FROM kaggle_customers 
WHERE gender IS NOT NULL
GROUP BY gender
ORDER BY customer_count DESC;

-- Income distribution
SELECT 
    CASE 
        WHEN annual_income_k < 30 THEN 'Low Income (<$30k)'
        WHEN annual_income_k BETWEEN 30 AND 60 THEN 'Medium Income ($30k-$60k)'
        WHEN annual_income_k BETWEEN 60 AND 100 THEN 'High Income ($60k-$100k)'
        WHEN annual_income_k >= 100 THEN 'Very High Income ($100k+)'
    END as income_group,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM kaggle_customers 
WHERE annual_income_k IS NOT NULL
GROUP BY income_group
ORDER BY income_group;

-- 2. TRANSACTION ANALYSIS
-- =======================

-- Monthly transaction trends
SELECT 
    DATE_TRUNC('month', transaction_date) as month,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value
FROM kaggle_transactions
WHERE transaction_date IS NOT NULL
GROUP BY month
ORDER BY month;

-- Product category performance
SELECT 
    product_category,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value,
    ROUND(SUM(total_amount) * 100.0 / SUM(SUM(total_amount)) OVER (), 2) as revenue_percentage
FROM kaggle_transactions
WHERE product_category IS NOT NULL
GROUP BY product_category
ORDER BY total_revenue DESC;

-- Customer transaction frequency
SELECT 
    customer_id,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_spent,
    AVG(total_amount) as avg_transaction_value,
    MIN(transaction_date) as first_purchase,
    MAX(transaction_date) as last_purchase
FROM kaggle_transactions
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 20;

-- 3. RFM ANALYSIS
-- ===============

-- RFM Score Distribution
SELECT 
    R_score,
    F_score,
    M_score,
    RFM_score,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM kaggle_rfm_scores
GROUP BY R_score, F_score, M_score, RFM_score
ORDER BY RFM_score DESC;

-- Segment Analysis
SELECT 
    segment,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as customer_percentage,
    AVG(recency) as avg_recency_days,
    AVG(frequency) as avg_frequency,
    AVG(monetary) as avg_monetary,
    SUM(monetary) as total_revenue,
    ROUND(SUM(monetary) * 100.0 / SUM(SUM(monetary)) OVER (), 2) as revenue_percentage
FROM kaggle_rfm_scores
GROUP BY segment
ORDER BY total_revenue DESC;

-- RFM Score Ranges by Segment
SELECT 
    segment,
    MIN(RFM_score) as min_rfm_score,
    MAX(RFM_score) as max_rfm_score,
    AVG(RFM_score) as avg_rfm_score,
    STDDEV(RFM_score) as rfm_score_std
FROM kaggle_rfm_scores
GROUP BY segment
ORDER BY avg_rfm_score DESC;

-- 4. CLUSTER ANALYSIS
-- ===================

-- Cluster Distribution
SELECT 
    cluster,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as customer_percentage,
    AVG(recency) as avg_recency,
    AVG(frequency) as avg_frequency,
    AVG(monetary) as avg_monetary,
    SUM(monetary) as total_revenue
FROM kaggle_customer_clusters
GROUP BY cluster
ORDER BY cluster;

-- Cluster vs Segment Comparison
SELECT 
    c.cluster,
    r.segment,
    COUNT(*) as customer_count,
    AVG(c.recency) as avg_recency,
    AVG(c.frequency) as avg_frequency,
    AVG(c.monetary) as avg_monetary
FROM kaggle_customer_clusters c
JOIN kaggle_rfm_scores r ON c.customer_id = r.customer_id
GROUP BY c.cluster, r.segment
ORDER BY c.cluster, customer_count DESC;

-- 5. LTV ANALYSIS
-- ===============

-- LTV by Segment
SELECT 
    segment,
    COUNT(*) as customer_count,
    AVG(predicted_ltv) as avg_predicted_ltv,
    SUM(predicted_ltv) as total_predicted_ltv,
    AVG(ltv_accuracy) as avg_ltv_accuracy,
    MIN(predicted_ltv) as min_ltv,
    MAX(predicted_ltv) as max_ltv
FROM kaggle_ltv_predictions
WHERE segment IS NOT NULL
GROUP BY segment
ORDER BY avg_predicted_ltv DESC;

-- LTV by Cluster
SELECT 
    c.cluster,
    COUNT(*) as customer_count,
    AVG(l.predicted_ltv) as avg_predicted_ltv,
    SUM(l.predicted_ltv) as total_predicted_ltv,
    AVG(l.ltv_accuracy) as avg_ltv_accuracy
FROM kaggle_customer_clusters c
JOIN kaggle_ltv_predictions l ON c.customer_id = l.customer_id
GROUP BY c.cluster
ORDER BY avg_predicted_ltv DESC;

-- Top Customers by Predicted LTV
SELECT 
    l.customer_id,
    c.gender,
    c.age,
    c.annual_income_k,
    r.segment,
    c2.cluster,
    l.predicted_ltv,
    l.ltv_accuracy
FROM kaggle_ltv_predictions l
JOIN kaggle_customers c ON l.customer_id = c.customer_id
JOIN kaggle_rfm_scores r ON l.customer_id = r.customer_id
JOIN kaggle_customer_clusters c2 ON l.customer_id = c2.customer_id
ORDER BY l.predicted_ltv DESC
LIMIT 20;

-- 6. CROSS-ANALYSIS
-- ==================

-- Age vs LTV Analysis
SELECT 
    CASE 
        WHEN c.age < 25 THEN '18-24'
        WHEN c.age BETWEEN 25 AND 34 THEN '25-34'
        WHEN c.age BETWEEN 35 AND 44 THEN '35-44'
        WHEN c.age BETWEEN 45 AND 54 THEN '45-54'
        WHEN c.age >= 55 THEN '55+'
    END as age_group,
    COUNT(*) as customer_count,
    AVG(l.predicted_ltv) as avg_predicted_ltv,
    AVG(r.monetary) as avg_monetary,
    AVG(r.frequency) as avg_frequency
FROM kaggle_ltv_predictions l
JOIN kaggle_customers c ON l.customer_id = c.customer_id
JOIN kaggle_rfm_scores r ON l.customer_id = r.customer_id
WHERE c.age IS NOT NULL
GROUP BY age_group
ORDER BY avg_predicted_ltv DESC;

-- Income vs Segment Analysis
SELECT 
    CASE 
        WHEN c.annual_income_k < 30 THEN 'Low Income (<$30k)'
        WHEN c.annual_income_k BETWEEN 30 AND 60 THEN 'Medium Income ($30k-$60k)'
        WHEN c.annual_income_k BETWEEN 60 AND 100 THEN 'High Income ($60k-$100k)'
        WHEN c.annual_income_k >= 100 THEN 'Very High Income ($100k+)'
    END as income_group,
    r.segment,
    COUNT(*) as customer_count,
    AVG(l.predicted_ltv) as avg_predicted_ltv
FROM kaggle_ltv_predictions l
JOIN kaggle_customers c ON l.customer_id = c.customer_id
JOIN kaggle_rfm_scores r ON l.customer_id = r.customer_id
WHERE c.annual_income_k IS NOT NULL
GROUP BY income_group, r.segment
ORDER BY income_group, avg_predicted_ltv DESC;

-- 7. BUSINESS INSIGHTS
-- ====================

-- Revenue Opportunity by Segment
SELECT 
    segment,
    COUNT(*) as customer_count,
    SUM(monetary) as current_revenue,
    AVG(predicted_ltv) as avg_predicted_ltv,
    SUM(predicted_ltv) as total_predicted_ltv,
    ROUND((SUM(predicted_ltv) - SUM(monetary)) / SUM(monetary) * 100, 2) as growth_potential_percent
FROM kaggle_rfm_scores r
JOIN kaggle_ltv_predictions l ON r.customer_id = l.customer_id
GROUP BY segment
ORDER BY growth_potential_percent DESC;

-- Customer Retention Risk Analysis
SELECT 
    segment,
    COUNT(*) as customer_count,
    AVG(recency) as avg_recency_days,
    CASE 
        WHEN AVG(recency) > 90 THEN 'High Risk'
        WHEN AVG(recency) > 60 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as retention_risk
FROM kaggle_rfm_scores
GROUP BY segment
ORDER BY avg_recency_days DESC;
