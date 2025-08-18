-- Customer Segmentation & LTV Analysis - RFM Analysis SQL Queries
-- This file contains SQL queries for performing RFM analysis on customer transaction data

-- ============================================================================
-- 1. BASIC DATA EXPLORATION
-- ============================================================================

-- Total customers and transactions
SELECT 
    COUNT(DISTINCT customer_id) as total_customers,
    COUNT(*) as total_transactions,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value
FROM transactions;

-- Transaction date range
SELECT 
    MIN(transaction_date) as earliest_transaction,
    MAX(transaction_date) as latest_transaction,
    DATEDIFF(day, MIN(transaction_date), MAX(transaction_date)) as days_span
FROM transactions;

-- Customer distribution by city
SELECT 
    city,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) as percentage
FROM customers
GROUP BY city
ORDER BY customer_count DESC;

-- Product category performance
SELECT 
    product_category,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) as percentage
FROM transactions
GROUP BY product_category
ORDER BY total_revenue DESC;

-- ============================================================================
-- 2. RFM ANALYSIS - CALCULATE RFM METRICS
-- ============================================================================

-- Calculate RFM metrics for each customer
WITH rfm_calculation AS (
    SELECT 
        customer_id,
        -- Recency: Days since last purchase
        DATEDIFF(day, MAX(transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        -- Frequency: Number of purchases
        COUNT(*) as frequency,
        -- Monetary: Total amount spent
        SUM(total_amount) as monetary,
        -- Additional metrics
        AVG(total_amount) as avg_order_value,
        MIN(transaction_date) as first_purchase_date,
        MAX(transaction_date) as last_purchase_date
    FROM transactions
    GROUP BY customer_id
)
SELECT * FROM rfm_calculation
ORDER BY monetary DESC;

-- ============================================================================
-- 3. RFM SCORING - QUINTILE-BASED SCORING
-- ============================================================================

-- Create RFM scores using quintiles
WITH rfm_calculation AS (
    SELECT 
        customer_id,
        DATEDIFF(day, MAX(transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(total_amount) as monetary,
        AVG(total_amount) as avg_order_value,
        MIN(transaction_date) as first_purchase_date,
        MAX(transaction_date) as last_purchase_date
    FROM transactions
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT 
        *,
        -- Recency score (1-5, lower recency = higher score)
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        -- Frequency score (1-5, higher frequency = higher score)
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        -- Monetary score (1-5, higher monetary = higher score)
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
)
SELECT 
    *,
    (R_score + F_score + M_score) as RFM_score,
    ROUND((R_score * 0.5 + F_score * 0.3 + M_score * 0.2), 2) as RFM_weighted_score
FROM rfm_scored
ORDER BY RFM_score DESC;

-- ============================================================================
-- 4. CUSTOMER SEGMENTATION BASED ON RFM SCORES
-- ============================================================================

-- Segment customers using RFM scores
WITH rfm_calculation AS (
    SELECT 
        customer_id,
        DATEDIFF(day, MAX(transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(total_amount) as monetary,
        AVG(total_amount) as avg_order_value
    FROM transactions
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        *,
        (R_score + F_score + M_score) as RFM_score,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT 
    segment,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM rfm_segmented), 2) as percentage,
    ROUND(AVG(monetary), 2) as avg_monetary,
    ROUND(AVG(frequency), 2) as avg_frequency,
    ROUND(AVG(recency), 2) as avg_recency,
    ROUND(AVG(RFM_score), 2) as avg_rfm_score
FROM rfm_segmented
GROUP BY segment
ORDER BY avg_monetary DESC;

-- ============================================================================
-- 5. CUSTOMER SEGMENT ANALYSIS WITH DEMOGRAPHICS
-- ============================================================================

-- Segment analysis with customer demographics
WITH rfm_calculation AS (
    SELECT 
        t.customer_id,
        DATEDIFF(day, MAX(t.transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(t.total_amount) as monetary,
        AVG(t.total_amount) as avg_order_value,
        c.customer_type,
        c.age,
        c.gender,
        c.city
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    GROUP BY t.customer_id, c.customer_type, c.age, c.gender, c.city
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        *,
        (R_score + F_score + M_score) as RFM_score,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT 
    segment,
    customer_type,
    gender,
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as customer_count,
    ROUND(AVG(monetary), 2) as avg_monetary,
    ROUND(AVG(frequency), 2) as avg_frequency,
    ROUND(AVG(RFM_score), 2) as avg_rfm_score
FROM rfm_segmented
GROUP BY segment, customer_type, gender, 
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END
ORDER BY segment, avg_monetary DESC;

-- ============================================================================
-- 6. PRODUCT CATEGORY ANALYSIS BY SEGMENT
-- ============================================================================

-- Product category performance by customer segment
WITH rfm_calculation AS (
    SELECT 
        t.customer_id,
        DATEDIFF(day, MAX(t.transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(t.total_amount) as monetary
    FROM transactions t
    GROUP BY t.customer_id
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        customer_id,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT 
    s.segment,
    t.product_category,
    COUNT(*) as transaction_count,
    SUM(t.total_amount) as total_revenue,
    ROUND(AVG(t.total_amount), 2) as avg_transaction_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.segment), 2) as percentage_of_segment
FROM rfm_segmented s
JOIN transactions t ON s.customer_id = t.customer_id
GROUP BY s.segment, t.product_category
ORDER BY s.segment, total_revenue DESC;

-- ============================================================================
-- 7. TIME-BASED ANALYSIS
-- ============================================================================

-- Monthly transaction trends
SELECT 
    YEAR(transaction_date) as year,
    MONTH(transaction_date) as month,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value,
    COUNT(DISTINCT customer_id) as unique_customers
FROM transactions
GROUP BY YEAR(transaction_date), MONTH(transaction_date)
ORDER BY year, month;

-- Customer acquisition and retention analysis
WITH customer_first_purchase AS (
    SELECT 
        customer_id,
        MIN(transaction_date) as first_purchase_date,
        MAX(transaction_date) as last_purchase_date,
        COUNT(*) as total_transactions
    FROM transactions
    GROUP BY customer_id
),
monthly_cohorts AS (
    SELECT 
        customer_id,
        YEAR(first_purchase_date) * 100 + MONTH(first_purchase_date) as cohort_month,
        first_purchase_date,
        last_purchase_date,
        total_transactions
    FROM customer_first_purchase
)
SELECT 
    cohort_month,
    COUNT(*) as cohort_size,
    COUNT(CASE WHEN DATEDIFF(day, first_purchase_date, last_purchase_date) > 30 THEN 1 END) as retained_30_days,
    COUNT(CASE WHEN DATEDIFF(day, first_purchase_date, last_purchase_date) > 90 THEN 1 END) as retained_90_days,
    COUNT(CASE WHEN DATEDIFF(day, first_purchase_date, last_purchase_date) > 180 THEN 1 END) as retained_180_days,
    ROUND(AVG(total_transactions), 2) as avg_transactions_per_customer
FROM monthly_cohorts
GROUP BY cohort_month
ORDER BY cohort_month;

-- ============================================================================
-- 8. PAYMENT METHOD ANALYSIS
-- ============================================================================

-- Payment method usage by segment
WITH rfm_calculation AS (
    SELECT 
        t.customer_id,
        DATEDIFF(day, MAX(t.transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(t.total_amount) as monetary
    FROM transactions t
    GROUP BY t.customer_id
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        customer_id,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT 
    s.segment,
    t.payment_method,
    COUNT(*) as transaction_count,
    SUM(t.total_amount) as total_revenue,
    ROUND(AVG(t.total_amount), 2) as avg_transaction_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.segment), 2) as percentage_of_segment
FROM rfm_segmented s
JOIN transactions t ON s.customer_id = t.customer_id
GROUP BY s.segment, t.payment_method
ORDER BY s.segment, total_revenue DESC;

-- ============================================================================
-- 9. TOP CUSTOMERS ANALYSIS
-- ============================================================================

-- Top 20 customers by RFM score
WITH rfm_calculation AS (
    SELECT 
        t.customer_id,
        c.customer_type,
        c.city,
        DATEDIFF(day, MAX(t.transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(t.total_amount) as monetary,
        AVG(t.total_amount) as avg_order_value
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    GROUP BY t.customer_id, c.customer_type, c.city
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        *,
        (R_score + F_score + M_score) as RFM_score,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT TOP 20
    customer_id,
    customer_type,
    city,
    recency,
    frequency,
    monetary,
    avg_order_value,
    RFM_score,
    segment
FROM rfm_segmented
ORDER BY RFM_score DESC, monetary DESC;

-- ============================================================================
-- 10. SUMMARY STATISTICS
-- ============================================================================

-- Overall RFM summary statistics
WITH rfm_calculation AS (
    SELECT 
        customer_id,
        DATEDIFF(day, MAX(transaction_date), (SELECT MAX(transaction_date) FROM transactions)) as recency,
        COUNT(*) as frequency,
        SUM(total_amount) as monetary
    FROM transactions
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency) as F_score,
        NTILE(5) OVER (ORDER BY monetary) as M_score
    FROM rfm_calculation
),
rfm_segmented AS (
    SELECT 
        *,
        (R_score + F_score + M_score) as RFM_score,
        CASE 
            WHEN R_score >= 4 AND F_score >= 4 AND M_score >= 4 THEN 'High-Value'
            WHEN F_score >= 4 AND M_score >= 4 AND R_score >= 3 THEN 'Loyal'
            WHEN R_score <= 2 AND F_score >= 3 AND M_score >= 3 THEN 'At-Risk'
            WHEN R_score >= 4 AND F_score <= 2 AND M_score <= 2 THEN 'New'
            WHEN R_score >= 4 AND F_score >= 3 AND M_score <= 2 THEN 'Promising'
            WHEN R_score <= 2 AND F_score <= 2 AND M_score <= 2 THEN 'Lost'
            ELSE 'Regular'
        END as segment
    FROM rfm_scored
)
SELECT 
    'Overall' as metric,
    COUNT(*) as total_customers,
    ROUND(AVG(recency), 2) as avg_recency,
    ROUND(AVG(frequency), 2) as avg_frequency,
    ROUND(AVG(monetary), 2) as avg_monetary,
    ROUND(AVG(RFM_score), 2) as avg_rfm_score,
    ROUND(SUM(monetary), 2) as total_revenue
FROM rfm_segmented
UNION ALL
SELECT 
    segment as metric,
    COUNT(*) as total_customers,
    ROUND(AVG(recency), 2) as avg_recency,
    ROUND(AVG(frequency), 2) as avg_frequency,
    ROUND(AVG(monetary), 2) as avg_monetary,
    ROUND(AVG(RFM_score), 2) as avg_rfm_score,
    ROUND(SUM(monetary), 2) as total_revenue
FROM rfm_segmented
GROUP BY segment
ORDER BY total_revenue DESC;
