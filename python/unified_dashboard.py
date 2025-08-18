"""
Unified Customer Segmentation & LTV Analysis Dashboard

A single dashboard that properly handles both synthetic and Kaggle datasets
without mixing them up. Routes to the correct data based on user selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime
import psutil
import gc

# Page configuration
st.set_page_config(
    page_title="🎯 Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .segment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(79, 172, 254, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(67, 233, 123, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def create_metric_card(value, label, prefix="", suffix=""):
    """Creates a beautiful metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{prefix}{value}{suffix}</h3>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)

def optimize_dataframe(df, max_rows=10000):
    """Optimize dataframe for better performance"""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

def create_optimized_chart(fig, height=400, show_modebar=False):
    """Create optimized plotly chart with performance settings"""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )
    return st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={'displayModeBar': show_modebar}
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_synthetic_data():
    """Load synthetic dataset with caching for better performance"""
    try:
        # Load only essential columns to reduce memory usage
        customers = pd.read_csv('data/raw/customers_large.csv', usecols=['customer_id', 'age', 'gender'])
        transactions = pd.read_csv('data/raw/transactions_large.csv', usecols=['customer_id', 'transaction_date', 'total_amount'])
        rfm_data = pd.read_csv('data/processed/rfm_scores.csv')
        customer_segments = pd.read_csv('data/processed/customer_segments.csv')
        clustering_data = pd.read_csv('data/processed/customer_clusters.csv')
        ltv_data = pd.read_csv('data/processed/ltv_predictions.csv')
        
        return {
            'customers': customers,
            'transactions': transactions,
            'rfm_data': rfm_data,
            'customer_segments': customer_segments,
            'clustering_data': clustering_data,
            'ltv_data': ltv_data,
            'dataset_type': 'synthetic'
        }
    except FileNotFoundError as e:
        st.error(f"❌ Synthetic data files not found: {e}")
        st.info("Please run the synthetic data analysis scripts first.")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_kaggle_data():
    """Load Kaggle dataset with caching for better performance"""
    try:
        # Load only essential columns to reduce memory usage
        customers = pd.read_csv('data/kaggle/raw/kaggle_retail_customers.csv', usecols=['CustomerID', 'Age', 'Gender', 'Annual_Income_k'])
        transactions = pd.read_csv('data/kaggle/raw/kaggle_retail_transactions.csv', usecols=['CustomerID', 'TransactionDate', 'TotalAmount'])
        rfm_data = pd.read_csv('data/kaggle/processed/kaggle_rfm_scores.csv')
        customer_segments = pd.read_csv('data/kaggle/processed/kaggle_customer_segments.csv')
        clustering_data = pd.read_csv('data/kaggle/processed/kaggle_customer_clusters.csv')
        ltv_data = pd.read_csv('data/kaggle/processed/kaggle_ltv_predictions.csv')
        
        return {
            'customers': customers,
            'transactions': transactions,
            'rfm_data': rfm_data,
            'customer_segments': customer_segments,
            'clustering_data': clustering_data,
            'ltv_data': ltv_data,
            'dataset_type': 'kaggle'
        }
    except FileNotFoundError as e:
        st.error(f"❌ Kaggle data files not found: {e}")
        st.info("Please run the Kaggle data analysis scripts first.")
        return None

def show_executive_summary(data):
    """Modern executive summary with key metrics"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>🏠 Executive Summary - {dataset_name} Dataset</h1><p>Key Business Metrics & Insights</p></div>', unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use RFM data for accurate customer count (not LTV data which might be sampled)
        total_customers = len(data['rfm_data'])
        create_metric_card(f"{total_customers:,}", "Total Customers")
    
    with col2:
        total_transactions = len(data['transactions'])
        create_metric_card(f"{total_transactions:,}", "Total Transactions")
    
    with col3:
        avg_ltv = data['ltv_data']['predicted_ltv'].mean()
        create_metric_card(f"${avg_ltv:,.0f}", "Avg Predicted LTV", prefix="$")
    
    with col4:
        total_revenue = data['rfm_data']['monetary'].sum()
        create_metric_card(f"${total_revenue:,.0f}", "Total Revenue", prefix="$")
    
    # Customer segments distribution
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Customer Segment Distribution")
    
    segment_counts = data['rfm_data']['segment'].value_counts()
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Distribution by Segment",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    create_optimized_chart(fig, height=400, show_modebar=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # LTV distribution with sampling for large datasets
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("💰 Predicted LTV Distribution")
    
    # Sample data if too large for better performance
    ltv_sample = data['ltv_data'].sample(min(1000, len(data['ltv_data'])), random_state=42)
    
    fig = px.histogram(
        ltv_sample, 
        x='predicted_ltv',
        nbins=30,
        title="Distribution of Predicted LTV",
        labels={'predicted_ltv': 'Predicted LTV ($)', 'count': 'Number of Customers'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(bargap=0.1, bargroupgap=0.05)
    create_optimized_chart(fig, height=400, show_modebar=False)
    st.markdown('</div>', unsafe_allow_html=True)

def show_rfm_analysis(data):
    """Modern RFM analysis with beautiful visualizations"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>📈 RFM Analysis - {dataset_name} Dataset</h1><p>Recency, Frequency, Monetary Customer Analysis</p></div>', unsafe_allow_html=True)
    
    # RFM metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_recency = data['rfm_data']['recency'].mean()
        create_metric_card(f"{avg_recency:.0f}", "Avg Days Since Last Purchase", suffix=" days")
    
    with col2:
        avg_frequency = data['rfm_data']['frequency'].mean()
        create_metric_card(f"{avg_frequency:.1f}", "Avg Purchase Frequency")
    
    with col3:
        avg_monetary = data['rfm_data']['monetary'].mean()
        create_metric_card(f"${avg_monetary:,.0f}", "Avg Customer Value", prefix="$")
    
    # RFM scores distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 RFM Score Distribution")
        
        fig = px.histogram(
            data['rfm_data'],
            x='RFM_score',
            nbins=15,
            title="RFM Score Distribution",
            labels={'RFM_score': 'RFM Score', 'count': 'Number of Customers'},
            color_discrete_sequence=['#f093fb']
        )
        fig.update_layout(height=400, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Segment Distribution")
        
        segment_counts = data['rfm_data']['segment'].value_counts()
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title="Customer Count by Segment",
            labels={'x': 'Segment', 'y': 'Number of Customers'},
            color=segment_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False, bargap=0.2, bargroupgap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RFM scatter plot
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 RFM 3D Scatter Plot")
    
    fig = px.scatter_3d(
        data['rfm_data'],
        x='recency',
        y='frequency',
        z='monetary',
        color='segment',
        title="RFM Analysis - 3D Visualization",
        labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_customer_segmentation(data):
    """Modern customer segmentation with cluster analysis"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>🎯 Customer Segmentation (K-Means) - {dataset_name} Dataset</h1><p>Machine Learning-Based Customer Clustering</p></div>', unsafe_allow_html=True)
    
    try:
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Cluster Distribution")
            
            cluster_counts = data['clustering_data']['cluster'].value_counts().sort_index()
            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Customer Distribution by Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("💰 Revenue by Cluster")
            
            # For both datasets, monetary column is already in clustering_data
            cluster_revenue = data['clustering_data'].groupby('cluster')['monetary'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=[f"Cluster {i}" for i in cluster_revenue.index],
                y=cluster_revenue.values,
                title="Total Revenue by Cluster",
                labels={'x': 'Cluster', 'y': 'Total Revenue ($)'},
                color=cluster_revenue.values,
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=400, showlegend=False, bargap=0.2, bargroupgap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cluster characteristics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Cluster Characteristics")
        
        # For both datasets, all RFM columns are already in clustering_data
        try:
            # Create subplot for cluster characteristics
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Recency by Cluster', 'Frequency by Cluster', 'Monetary by Cluster'),
                specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]]
            )
            
            for i, metric in enumerate(['recency', 'frequency', 'monetary']):
                for cluster in sorted(data['clustering_data']['cluster'].unique()):
                    cluster_data = data['clustering_data'][data['clustering_data']['cluster'] == cluster][metric]
                    fig.add_trace(
                        go.Box(y=cluster_data, name=f'Cluster {cluster}', showlegend=False),
                        row=1, col=i+1
                    )
            
            fig.update_layout(height=400, title_text="Cluster Characteristics Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
        except KeyError as e:
            st.error(f"❌ Missing column in data: {e}")
            st.info("Please ensure all required columns are present in the data files.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error in customer segmentation: {e}")
        st.info("Please ensure all data files are properly generated and contain the required columns.")

def show_ltv_prediction(data):
    """Modern LTV prediction with beautiful visualizations"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>💰 Customer Lifetime Value Prediction - {dataset_name} Dataset</h1><p>Machine Learning-Based LTV Forecasting</p></div>', unsafe_allow_html=True)
    
    try:
        # LTV metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_ltv = data['ltv_data']['predicted_ltv'].mean()
            create_metric_card(f"${avg_ltv:,.0f}", "Average Predicted LTV", prefix="$")
        
        with col2:
            total_ltv = data['ltv_data']['predicted_ltv'].sum()
            create_metric_card(f"${total_ltv:,.0f}", "Total Predicted LTV", prefix="$")
        
        with col3:
            max_ltv = data['ltv_data']['predicted_ltv'].max()
            create_metric_card(f"${max_ltv:,.0f}", "Highest Predicted LTV", prefix="$")
        
        # LTV by segment
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("💎 LTV by Customer Segment")
            
            try:
                # For both datasets, segment column is already in ltv_data
                segment_ltv = data['ltv_data'].groupby('segment')['predicted_ltv'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=segment_ltv.index,
                    y=segment_ltv.values,
                    title="Average LTV by Segment",
                    labels={'x': 'Segment', 'y': 'Average LTV ($)'},
                    color=segment_ltv.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400, showlegend=False, bargap=0.2, bargroupgap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            except KeyError as e:
                st.error(f"❌ Missing column in data: {e}")
                st.info("Please ensure segment information is available in LTV data.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 LTV Distribution by Segment")
            
            try:
                fig = px.box(
                    data['ltv_data'],
                    x='segment',
                    y='predicted_ltv',
                    title="LTV Distribution by Segment",
                    labels={'segment': 'Segment', 'predicted_ltv': 'Predicted LTV ($)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating LTV distribution: {e}")
                st.info("Please ensure segment information is properly available.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # LTV scatter plot
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 LTV vs RFM Score")
        
        try:
            # For both datasets, merge with clustering data which has RFM_score
            ltv_rfm = data['ltv_data'].merge(data['clustering_data'][['customer_id', 'RFM_score']], on='customer_id', how='inner')
            
            # Handle duplicate RFM_score columns (Kaggle data might have this issue)
            if 'RFM_score_x' in ltv_rfm.columns and 'RFM_score_y' in ltv_rfm.columns:
                # Use RFM_score_y (from clustering data) and drop RFM_score_x
                ltv_rfm['RFM_score'] = ltv_rfm['RFM_score_y']
                ltv_rfm = ltv_rfm.drop(['RFM_score_x', 'RFM_score_y'], axis=1)
            elif 'RFM_score_x' in ltv_rfm.columns:
                # Use RFM_score_x and drop it
                ltv_rfm['RFM_score'] = ltv_rfm['RFM_score_x']
                ltv_rfm = ltv_rfm.drop('RFM_score_x', axis=1)
            elif 'RFM_score_y' in ltv_rfm.columns:
                # Use RFM_score_y and drop it
                ltv_rfm['RFM_score'] = ltv_rfm['RFM_score_y']
                ltv_rfm = ltv_rfm.drop('RFM_score_y', axis=1)
            
            # Filter out negative LTV values for size parameter
            ltv_rfm_positive = ltv_rfm[ltv_rfm['predicted_ltv'] > 0].copy()
            
            fig = px.scatter(
                ltv_rfm_positive,
                x='RFM_score',
                y='predicted_ltv',
                title="LTV vs RFM Score Relationship",
                labels={'RFM_score': 'RFM Score', 'predicted_ltv': 'Predicted LTV ($)'},
                size='predicted_ltv',
                hover_data=['customer_id']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except KeyError as e:
            st.error(f"❌ Missing column in data: {e}")
            st.info("Please ensure RFM score information is available.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error in LTV prediction: {e}")
        st.info("Please ensure all data files are properly generated and contain the required columns.")

def show_marketing_insights(data):
    """Modern marketing insights with actionable recommendations"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>📊 Marketing Insights & Recommendations - {dataset_name} Dataset</h1><p>Data-Driven Marketing Strategies</p></div>', unsafe_allow_html=True)
    
    # Marketing recommendations by segment with expandable sections
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 Segment-Specific Marketing Recommendations")
    
    # Get unique segments from the data
    if dataset_type == 'synthetic':
        # For synthetic data, use segments from RFM data
        available_segments = data['rfm_data']['segment'].unique()
    else:
        # For Kaggle data, use predefined segments
        available_segments = ['High-Value', 'Loyal', 'At-Risk', 'New', 'Lost']
    
    # Show all segments with expandable sections
    for segment in available_segments:
        if segment in data['rfm_data']['segment'].values:
            segment_data = data['rfm_data'][data['rfm_data']['segment'] == segment]
            segment_count = len(segment_data)
            segment_revenue = segment_data['monetary'].sum()
            
            # Create expandable section for each segment
            with st.expander(f"🎯 {segment} Customers ({segment_count:,} customers) - ${segment_revenue:,.2f} Revenue", expanded=True):
                st.markdown(f"""
                <div class="segment-card">
                    <h4>📊 {segment} Segment Analysis</h4>
                    <p><strong>Total Revenue:</strong> ${segment_revenue:,.2f}</p>
                    <p><strong>Marketing Strategy:</strong></p>
                    <p style="white-space: pre-line; margin-top: 10px;">{get_marketing_strategy(segment)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Revenue optimization opportunities
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Revenue Optimization Opportunities")
    
    # Calculate potential revenue increase
    high_value_customers = data['rfm_data'][data['rfm_data']['segment'] == 'High-Value']
    current_revenue = high_value_customers['monetary'].sum()
    potential_increase = current_revenue * 0.2  # 20% potential increase
    
    st.write(f"**Current High-Value Customer Revenue:** ${current_revenue:,.2f}")
    st.write(f"**Potential 20% Revenue Increase:** ${potential_increase:,.2f}")
    st.write("**Strategy:** Focus on upselling and cross-selling to High-Value customers")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer retention opportunities
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.subheader("🔄 Customer Retention Opportunities")
    
    at_risk_customers = data['rfm_data'][data['rfm_data']['segment'] == 'At-Risk']
    at_risk_count = len(at_risk_customers)
    at_risk_revenue = at_risk_customers['monetary'].sum()
    
    st.write(f"**At-Risk Customers:** {at_risk_count:,}")
    st.write(f"**At-Risk Revenue:** ${at_risk_revenue:,.2f}")
    st.write("**Strategy:** Implement re-engagement campaigns and loyalty programs")
    st.markdown('</div>', unsafe_allow_html=True)

def get_marketing_strategy(segment):
    """Returns marketing strategy for each segment"""
    strategies = {
        'High-Value': '• Premium services & VIP treatment\n• Exclusive offers & early access\n• Upselling & cross-selling opportunities\n• Dedicated account managers',
        'Loyal': '• Loyalty rewards & points system\n• Referral programs & incentives\n• New product introductions\n• Exclusive member events',
        'At-Risk': '• Re-engagement campaigns\n• Win-back offers & discounts\n• Customer feedback surveys\n• Personalized retention strategies',
        'New': '• Welcome series & onboarding\n• Product education & tutorials\n• First purchase incentives\n• Social proof & testimonials',
        'Lost': '• Win-back campaigns\n• Special offers & reactivation\n• Customer feedback collection\n• Reactivation strategies',
        'Regular': '• Standard marketing campaigns\n• Product recommendations\n• Seasonal promotions\n• Email marketing',
        'Promising': '• Growth-focused campaigns\n• Upselling opportunities\n• Engagement strategies\n• Value proposition reinforcement'
    }
    return strategies.get(segment, '• Standard marketing approach\n• General campaigns\n• Basic customer service')

def show_data_overview(data):
    """Modern data overview with comprehensive statistics"""
    dataset_type = data.get('dataset_type', 'unknown')
    dataset_name = "Synthetic" if dataset_type == 'synthetic' else "Kaggle"
    
    st.markdown(f'<div class="main-header"><h1>🔧 Data Overview - {dataset_name} Dataset</h1><p>Comprehensive Dataset Statistics</p></div>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use processed data for accurate customer count
        customer_count = len(data['rfm_data'])
        create_metric_card(f"{customer_count:,}", "Total Customers")
    
    with col2:
        transaction_count = len(data['transactions'])
        create_metric_card(f"{transaction_count:,}", "Total Transactions")
    
    with col3:
        avg_transactions = transaction_count / customer_count
        create_metric_card(f"{avg_transactions:.1f}", "Avg Transactions per Customer")
    
    with col4:
        total_revenue = data['rfm_data']['monetary'].sum()
        create_metric_card(f"${total_revenue:,.0f}", "Total Revenue", prefix="$")
    
    # Data quality metrics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Data Quality Overview")
    
    # Check for missing values
    missing_data = {}
    for key, df in data.items():
        if key != 'dataset_type':
            missing_count = df.isnull().sum().sum()
            missing_data[key] = missing_count
    
    # Only show chart if there are missing values
    if sum(missing_data.values()) > 0:
        fig = px.bar(
            x=list(missing_data.keys()),
            y=list(missing_data.values()),
            title="Missing Values by Dataset",
            labels={'x': 'Dataset', 'y': 'Missing Values'},
            color=list(missing_data.values()),
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, bargap=0.2, bargroupgap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in any dataset!")
        st.info("All datasets are complete and ready for analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer demographics (if available)
    if 'age' in data['customers'].columns or 'Age' in data['customers'].columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("👥 Customer Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_col = 'age' if 'age' in data['customers'].columns else 'Age'
            fig = px.histogram(
                data['customers'],
                x=age_col,
                nbins=20,
                title="Age Distribution",
                labels={age_col: 'Age', 'count': 'Number of Customers'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=300, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution (if available)
            gender_col = 'gender' if 'gender' in data['customers'].columns else 'Gender'
            if gender_col in data['customers'].columns:
                gender_counts = data['customers'][gender_col].value_counts()
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main dashboard function with performance optimizations"""
    
    # Add performance monitoring
    start_time = datetime.now()
    
    # Sidebar for dataset selection
    st.sidebar.title("📋 Dataset Selection")
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset:",
        ["Synthetic Dataset", "Kaggle Dataset"],
        index=0
    )
    
    # Show loading spinner
    with st.spinner("🔄 Loading dataset..."):
        # Load data based on selection
        if dataset_choice == "Synthetic Dataset":
            data = load_synthetic_data()
            if data:
                st.sidebar.success("✅ Loaded: Synthetic Dataset")
        else:
            data = load_kaggle_data()
            if data:
                st.sidebar.success("✅ Loaded: Kaggle Dataset")
    
    if data is None:
        st.error("❌ Failed to load data. Please ensure the selected dataset files are available.")
        st.stop()
    
    # Show loading time and system info
    load_time = (datetime.now() - start_time).total_seconds()
    st.sidebar.info(f"⏱️ Load time: {load_time:.2f}s")
    
    # Show system performance info
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    st.sidebar.metric("💾 Memory Usage", f"{memory_usage:.1f}%")
    st.sidebar.metric("🖥️ CPU Usage", f"{cpu_usage:.1f}%")
    
    # Force garbage collection for better memory management
    gc.collect()
    
    # Horizontal Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Executive Summary", 
        "📈 RFM Analysis", 
        "🎯 Customer Segmentation", 
        "💰 LTV Prediction", 
        "📊 Marketing Insights", 
        "🔧 Data Overview"
    ])
    
    # Page routing with tabs
    with tab1: 
        try:
            show_executive_summary(data)
        except Exception as e:
            st.error(f"Error in Executive Summary: {e}")
    
    with tab2: 
        try:
            show_rfm_analysis(data)
        except Exception as e:
            st.error(f"Error in RFM Analysis: {e}")
    
    with tab3: 
        try:
            show_customer_segmentation(data)
        except Exception as e:
            st.error(f"Error in Customer Segmentation: {e}")
    
    with tab4: 
        try:
            show_ltv_prediction(data)
        except Exception as e:
            st.error(f"Error in LTV Prediction: {e}")
    
    with tab5: 
        try:
            show_marketing_insights(data)
        except Exception as e:
            st.error(f"Error in Marketing Insights: {e}")
    
    with tab6: 
        try:
            show_data_overview(data)
        except Exception as e:
            st.error(f"Error in Data Overview: {e}")

if __name__ == "__main__":
    main()
