# 🗄️ SQL Analysis Guide - Customer Segmentation & LTV Analysis

## 📋 Overview

This guide provides step-by-step instructions for running SQL analysis on both synthetic and Kaggle datasets. The SQL analysis includes comprehensive RFM analysis, customer segmentation, LTV analysis, and business insights.

## 🛠️ Prerequisites

### 1. PostgreSQL Installation
- Install PostgreSQL (version 12 or higher)
- Ensure `psql` command-line tool is available
- Have appropriate permissions to create databases and tables

### 2. Data Files
Ensure the following data files are available:
- **Synthetic Data**: `data/raw/` and `data/processed/` folders
- **Kaggle Data**: `data/kaggle/raw/` and `data/kaggle/processed/` folders

## 🚀 Quick Start

### Step 1: Setup Database for Synthetic Data
```bash
# Create database
createdb customer_segmentation

# Import data and run analysis
psql -d customer_segmentation -f sql/setup_database.sql
psql -d customer_segmentation -f sql/rfm_analysis.sql
```

### Step 2: Setup Database for Kaggle Data
```bash
# Create database
createdb customer_segmentation_kaggle

# Import data and run analysis
psql -d customer_segmentation_kaggle -f sql/setup_kaggle_database.sql
psql -d customer_segmentation_kaggle -f sql/kaggle_rfm_analysis.sql
```

## 📊 Analysis Categories

### 1. Customer Demographics Analysis
- **Age Distribution**: Customer age groups and percentages
- **Gender Distribution**: Gender-based customer breakdown
- **Income Analysis**: Income brackets and customer distribution
- **Geographic Analysis**: Customer location patterns

### 2. Transaction Analysis
- **Monthly Trends**: Transaction volume and revenue over time
- **Product Performance**: Category-wise revenue analysis
- **Customer Behavior**: Purchase frequency and patterns
- **Seasonal Analysis**: Time-based transaction patterns

### 3. RFM Analysis
- **Score Distribution**: R, F, M score breakdowns
- **Segment Analysis**: Customer segment characteristics
- **Revenue by Segment**: Segment-wise revenue contribution
- **Score Ranges**: RFM score statistics by segment

### 4. Cluster Analysis
- **Cluster Distribution**: K-means cluster breakdown
- **Cluster vs Segment**: Comparison between ML clusters and RFM segments
- **Cluster Characteristics**: Feature analysis by cluster
- **Revenue by Cluster**: Cluster-wise revenue analysis

### 5. LTV Analysis
- **LTV by Segment**: Segment-wise lifetime value predictions
- **LTV by Cluster**: Cluster-wise LTV analysis
- **Top Customers**: Highest LTV customers identification
- **Accuracy Analysis**: Model performance metrics

### 6. Cross-Analysis
- **Age vs LTV**: Age group LTV correlation
- **Income vs Segment**: Income-based segment analysis
- **Demographics vs Behavior**: Customer profile analysis
- **Geographic Patterns**: Location-based insights

### 7. Business Insights
- **Revenue Opportunities**: Growth potential by segment
- **Retention Risk**: Customer churn risk analysis
- **Marketing ROI**: Segment-wise marketing effectiveness
- **Budget Allocation**: Data-driven budget recommendations

## 🔍 Key SQL Queries Explained

### RFM Score Calculation
```sql
-- Recency: Days since last purchase
SELECT customer_id, 
       EXTRACT(DAY FROM (CURRENT_DATE - MAX(transaction_date))) as recency
FROM transactions 
GROUP BY customer_id;

-- Frequency: Number of transactions
SELECT customer_id, 
       COUNT(*) as frequency
FROM transactions 
GROUP BY customer_id;

-- Monetary: Total amount spent
SELECT customer_id, 
       SUM(total_amount) as monetary
FROM transactions 
GROUP BY customer_id;
```

### Segment Assignment
```sql
-- Rule-based segmentation
SELECT customer_id,
       CASE 
           WHEN RFM_score >= 13 THEN 'High-Value'
           WHEN RFM_score >= 11 THEN 'Loyal'
           WHEN RFM_score >= 9 THEN 'At-Risk'
           WHEN RFM_score >= 7 THEN 'New'
           ELSE 'Lost'
       END as segment
FROM rfm_scores;
```

### LTV Analysis
```sql
-- Average LTV by segment
SELECT segment,
       AVG(predicted_ltv) as avg_ltv,
       SUM(predicted_ltv) as total_ltv
FROM ltv_predictions
GROUP BY segment
ORDER BY avg_ltv DESC;
```

## 📈 Business Intelligence Queries

### Revenue Opportunity Analysis
```sql
-- Growth potential by segment
SELECT segment,
       COUNT(*) as customer_count,
       SUM(monetary) as current_revenue,
       SUM(predicted_ltv) as predicted_revenue,
       ROUND((SUM(predicted_ltv) - SUM(monetary)) / SUM(monetary) * 100, 2) as growth_potential_percent
FROM rfm_scores r
JOIN ltv_predictions l ON r.customer_id = l.customer_id
GROUP BY segment
ORDER BY growth_potential_percent DESC;
```

### Customer Retention Risk
```sql
-- Retention risk by segment
SELECT segment,
       COUNT(*) as customer_count,
       AVG(recency) as avg_recency_days,
       CASE 
           WHEN AVG(recency) > 90 THEN 'High Risk'
           WHEN AVG(recency) > 60 THEN 'Medium Risk'
           ELSE 'Low Risk'
       END as retention_risk
FROM rfm_scores
GROUP BY segment
ORDER BY avg_recency_days DESC;
```

## 🎯 Marketing Insights Queries

### High-Value Customer Analysis
```sql
-- High-value customer characteristics
SELECT 
    c.gender,
    c.age_group,
    c.income_group,
    COUNT(*) as customer_count,
    AVG(l.predicted_ltv) as avg_ltv
FROM customers c
JOIN ltv_predictions l ON c.customer_id = l.customer_id
WHERE l.segment = 'High-Value'
GROUP BY c.gender, c.age_group, c.income_group
ORDER BY avg_ltv DESC;
```

### At-Risk Customer Identification
```sql
-- At-risk customers for retention campaigns
SELECT 
    c.customer_id,
    c.email,
    r.recency,
    r.frequency,
    r.monetary,
    l.predicted_ltv
FROM customers c
JOIN rfm_scores r ON c.customer_id = r.customer_id
JOIN ltv_predictions l ON c.customer_id = l.customer_id
WHERE r.segment = 'At-Risk'
ORDER BY l.predicted_ltv DESC
LIMIT 100;
```

## 🔧 Performance Optimization

### Index Creation
```sql
-- Create indexes for better query performance
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_rfm_segment ON rfm_scores(segment);
CREATE INDEX idx_ltv_segment ON ltv_predictions(segment);
```

### Query Optimization Tips
1. **Use EXPLAIN ANALYZE** to understand query performance
2. **Limit results** when exploring large datasets
3. **Use appropriate data types** for better storage efficiency
4. **Create materialized views** for frequently accessed data
5. **Partition large tables** by date for better performance

## 📊 Visualization Integration

### Export Results for Dashboard
```sql
-- Export segment analysis for dashboard
\copy (
    SELECT segment, customer_count, total_revenue, avg_ltv
    FROM segment_analysis
) TO 'segment_analysis.csv' WITH CSV HEADER;

-- Export LTV predictions for visualization
\copy (
    SELECT customer_id, segment, predicted_ltv, ltv_accuracy
    FROM ltv_predictions
) TO 'ltv_predictions.csv' WITH CSV HEADER;
```

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Grant necessary permissions
   GRANT ALL PRIVILEGES ON DATABASE customer_segmentation TO your_user;
   ```

2. **File Not Found**
   ```bash
   # Check file paths and permissions
   ls -la data/processed/
   chmod 644 data/processed/*.csv
   ```

3. **Memory Issues**
   ```sql
   -- Increase work memory for large queries
   SET work_mem = '256MB';
   SET maintenance_work_mem = '512MB';
   ```

4. **Connection Issues**
   ```bash
   # Test database connection
   psql -h localhost -U username -d customer_segmentation
   ```

## 📈 Advanced Analysis

### Cohort Analysis
```sql
-- Customer cohort analysis
WITH cohort_data AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(transaction_date)) as cohort_month,
        DATE_TRUNC('month', transaction_date) as order_month
    FROM transactions
    GROUP BY customer_id, transaction_date
)
SELECT 
    cohort_month,
    order_month,
    COUNT(DISTINCT customer_id) as customers,
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / 
          FIRST_VALUE(COUNT(DISTINCT customer_id)) OVER (PARTITION BY cohort_month ORDER BY order_month), 2) as retention_rate
FROM cohort_data
GROUP BY cohort_month, order_month
ORDER BY cohort_month, order_month;
```

### Predictive Analytics
```sql
-- Customer churn prediction
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    CASE 
        WHEN recency > 90 THEN 'High Churn Risk'
        WHEN recency > 60 THEN 'Medium Churn Risk'
        ELSE 'Low Churn Risk'
    END as churn_risk
FROM rfm_scores
ORDER BY recency DESC;
```

## 🎉 Success Metrics

### Key Performance Indicators
- **Customer Segmentation Accuracy**: 95%+ segment assignment accuracy
- **LTV Prediction Accuracy**: 90%+ R² score
- **Query Performance**: <5 seconds for complex queries
- **Data Completeness**: <1% missing values

### Business Impact
- **Revenue Optimization**: 20%+ improvement in campaign ROI
- **Customer Retention**: 15%+ reduction in churn rate
- **Marketing Efficiency**: 25%+ improvement in targeting accuracy
- **Budget Allocation**: Data-driven optimization leading to 30%+ cost savings

This SQL analysis provides the foundation for data-driven customer segmentation and LTV analysis, enabling strategic marketing decisions and business growth.
