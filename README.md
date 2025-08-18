# 🎯 Customer Segmentation & Lifetime Value (LTV) Analysis

A comprehensive data science project that performs customer segmentation using RFM analysis and K-Means clustering, predicts customer lifetime value using machine learning, and provides interactive dashboards for business insights.

## 📊 Project Overview

This project demonstrates end-to-end customer analytics including:
- **RFM Analysis**: Recency, Frequency, Monetary customer segmentation
- **K-Means Clustering**: Machine learning-based customer clustering
- **LTV Prediction**: Customer lifetime value forecasting using multiple ML models
- **Interactive Dashboards**: Streamlit-based visualization and analysis
- **SQL Analysis**: Comprehensive database queries and business intelligence
- **Multiple Datasets**: Support for both synthetic and Kaggle-style datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL (for SQL analysis)
- Required Python packages (see `python/requirements.txt`)

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd CustomerSegmentation

# Install Python dependencies
pip install -r python/requirements.txt

# Run the unified dashboard
streamlit run python/unified_dashboard.py
```

## 📁 Project Structure

```
CustomerSegmentation/
├── python/                          # Python scripts
│   ├── unified_dashboard.py         # Main Streamlit dashboard
│   ├── simple_large_data_generation.py  # Synthetic data generation
│   ├── rfm_analysis.py              # RFM analysis for synthetic data
│   ├── simple_clustering.py         # K-Means clustering for synthetic data
│   ├── simple_ltv_prediction.py     # LTV prediction for synthetic data
│   ├── kaggle_data_processor.py     # Kaggle dataset processing
│   ├── kaggle_rfm_analysis.py       # RFM analysis for Kaggle data
│   ├── kaggle_clustering.py         # K-Means clustering for Kaggle data
│   ├── kaggle_ltv_prediction.py     # LTV prediction for Kaggle data
│   └── requirements.txt             # Python dependencies
├── sql/                             # SQL scripts
│   ├── setup_database.sql           # Database setup for synthetic data
│   ├── rfm_analysis.sql             # SQL analysis for synthetic data
│   ├── setup_kaggle_database.sql    # Database setup for Kaggle data
│   └── kaggle_rfm_analysis.sql      # SQL analysis for Kaggle data
├── data/                            # Data files
│   ├── raw/                         # Raw data files
│   ├── processed/                   # Processed analysis results
│   └── kaggle/                      # Kaggle dataset files
│       ├── raw/                     # Raw Kaggle data
│       └── processed/               # Processed Kaggle results
├── COMPLETE_PROJECT_SUMMARY.md      # Detailed project documentation
├── SQL_ANALYSIS_GUIDE.md           # SQL analysis guide
└── README.md                       # This file
```

## 🎯 Key Features

### 1. **Dual Dataset Support**
- **Synthetic Dataset**: 50,000 customers with realistic patterns
- **Kaggle Dataset**: 500 customers simulating real-world e-commerce data

### 2. **Comprehensive Analytics**
- **RFM Analysis**: Customer segmentation based on Recency, Frequency, Monetary
- **K-Means Clustering**: Machine learning clustering with optimal cluster detection
- **LTV Prediction**: Multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM)
- **Business Intelligence**: Revenue optimization and retention strategies

### 3. **Interactive Dashboard**
- **6 Main Sections**: Executive Summary, RFM Analysis, Customer Segmentation, LTV Prediction, Marketing Insights, Data Overview
- **Dataset Switching**: Seamless switching between synthetic and Kaggle datasets
- **Modern UI**: Beautiful, responsive design with interactive charts
- **Real-time Analytics**: Dynamic visualizations and insights

### 4. **SQL Analysis**
- **Database Integration**: PostgreSQL setup and data import
- **Comprehensive Queries**: Customer demographics, transaction analysis, RFM analysis, segment analysis
- **Business Intelligence**: Revenue opportunities, customer insights, performance metrics

## 📊 Dashboard Sections

### 1. **Executive Summary**
- Key business metrics (customers, transactions, revenue, LTV)
- Customer segment distribution
- Predicted LTV distribution

### 2. **RFM Analysis**
- Recency, Frequency, Monetary metrics
- RFM score distribution
- Customer segment analysis
- 3D RFM visualization

### 3. **Customer Segmentation**
- K-Means cluster distribution
- Revenue by cluster
- Cluster characteristics analysis

### 4. **LTV Prediction**
- Predicted lifetime value metrics
- LTV by customer segment
- LTV distribution analysis
- LTV vs RFM score relationship

### 5. **Marketing Insights**
- Segment-specific recommendations
- Revenue optimization opportunities
- Customer retention strategies

### 6. **Data Overview**
- Dataset statistics and quality assessment
- Customer demographics
- Missing value analysis

## 🔧 Usage Instructions

### Running the Dashboard
```bash
# Start the unified dashboard
streamlit run python/unified_dashboard.py

# Access the dashboard at http://localhost:8501
```

### Dataset Selection
1. Use the sidebar dropdown to select "Synthetic Dataset" or "Kaggle Dataset"
2. Navigate between sections using the horizontal tabs
3. Explore interactive visualizations and insights

### SQL Analysis
```bash
# Set up PostgreSQL database
psql -f sql/setup_database.sql          # For synthetic data
psql -f sql/setup_kaggle_database.sql   # For Kaggle data

# Run analysis queries
psql -f sql/rfm_analysis.sql            # For synthetic data
psql -f sql/kaggle_rfm_analysis.sql     # For Kaggle data
```

## 📈 Key Insights

### Customer Segments
- **High-Value**: Premium customers with high spending
- **Loyal**: Regular customers with good retention
- **At-Risk**: Customers showing declining engagement
- **New**: Recently acquired customers
- **Lost**: Inactive customers requiring reactivation

### Business Opportunities
- **Revenue Optimization**: 20% potential increase through upselling
- **Customer Retention**: Targeted campaigns for at-risk customers
- **Customer Acquisition**: Focus on high-value customer profiles
- **LTV Maximization**: Personalized strategies based on predicted values

## 🛠️ Technical Stack

- **Python**: Data processing, ML models, analytics
- **Streamlit**: Interactive dashboard framework
- **PostgreSQL**: Database management and SQL analysis
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **XGBoost/LightGBM**: Advanced ML models

## 📋 Requirements

### Python Packages
```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.15.0
xgboost==1.7.6
lightgbm==4.0.0
matplotlib==3.7.2
seaborn==0.12.2
```

### System Requirements
- Python 3.8 or higher
- PostgreSQL 12 or higher
- 4GB RAM minimum
- 2GB disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please open an issue in the repository or contact the project maintainers.

---

**Note**: This project is designed for educational and demonstration purposes. The datasets are synthetic and should not be used for production business decisions without proper validation.
