"""
Customer Segmentation & LTV Analysis - RFM Analysis Script

This script performs RFM (Recency, Frequency, Monetary) analysis on customer transaction data.
RFM analysis is a marketing technique used to determine quantitatively which customers are the best ones
by examining how recently a customer has purchased (recency), how often they purchase (frequency),
and how much the customer spends (monetary).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data():
    """Load customer and transaction data"""
    print("Loading customer and transaction data...")
    
    try:
        customers_df = pd.read_csv('../data/raw/customers.csv')
        transactions_df = pd.read_csv('../data/raw/transactions.csv')
        print(f"Loaded {len(customers_df)} customers and {len(transactions_df)} transactions")
        return customers_df, transactions_df
    except FileNotFoundError:
        print("Error: Data files not found. Please run data_generation.py first.")
        return None, None

def calculate_rfm_scores(transactions_df, analysis_date=None):
    """
    Calculate RFM scores for each customer
    
    Parameters:
    - transactions_df: DataFrame with transaction data
    - analysis_date: Date to calculate recency from (default: max transaction date)
    
    Returns:
    - DataFrame with customer_id and RFM scores
    """
    
    print("Calculating RFM scores...")
    
    # Convert transaction_date to datetime
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    # Set analysis date (default to max transaction date)
    if analysis_date is None:
        analysis_date = transactions_df['transaction_date'].max()
    else:
        analysis_date = pd.to_datetime(analysis_date)
    
    print(f"Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
    
    # Calculate RFM metrics for each customer
    rfm_data = []
    
    for customer_id in transactions_df['customer_id'].unique():
        customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
        
        # Recency: Days since last purchase
        last_purchase_date = customer_transactions['transaction_date'].max()
        recency = (analysis_date - last_purchase_date).days
        
        # Frequency: Number of purchases
        frequency = len(customer_transactions)
        
        # Monetary: Total amount spent
        monetary = customer_transactions['total_amount'].sum()
        
        rfm_data.append({
            'customer_id': customer_id,
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'last_purchase_date': last_purchase_date,
            'first_purchase_date': customer_transactions['transaction_date'].min(),
            'avg_order_value': monetary / frequency
        })
    
    rfm_df = pd.DataFrame(rfm_data)
    
    print(f"Calculated RFM scores for {len(rfm_df)} customers")
    
    return rfm_df

def score_rfm(rfm_df, r_quintiles=5, f_quintiles=5, m_quintiles=5):
    """
    Score RFM values into quintiles (1-5 scale)
    
    Parameters:
    - rfm_df: DataFrame with RFM values
    - r_quintiles, f_quintiles, m_quintiles: Number of quintiles for scoring
    
    Returns:
    - DataFrame with RFM scores added
    """
    
    print("Scoring RFM values into quintiles...")
    
    # Create a copy to avoid modifying original
    rfm_scored = rfm_df.copy()
    
    # Score Recency (lower is better, so we reverse the quintiles)
    rfm_scored['R_score'] = pd.qcut(rfm_scored['recency'], 
                                   q=r_quintiles, 
                                   labels=range(r_quintiles, 0, -1))
    
    # Score Frequency (higher is better)
    rfm_scored['F_score'] = pd.qcut(rfm_scored['frequency'], 
                                   q=f_quintiles, 
                                   labels=range(1, f_quintiles + 1))
    
    # Score Monetary (higher is better)
    rfm_scored['M_score'] = pd.qcut(rfm_scored['monetary'], 
                                   q=m_quintiles, 
                                   labels=range(1, m_quintiles + 1))
    
    # Convert to numeric
    rfm_scored['R_score'] = pd.to_numeric(rfm_scored['R_score'])
    rfm_scored['F_score'] = pd.to_numeric(rfm_scored['F_score'])
    rfm_scored['M_score'] = pd.to_numeric(rfm_scored['M_score'])
    
    # Calculate RFM Score (combination of all three)
    rfm_scored['RFM_score'] = rfm_scored['R_score'] + rfm_scored['F_score'] + rfm_scored['M_score']
    
    # Calculate RFM Score (weighted combination)
    rfm_scored['RFM_weighted_score'] = (rfm_scored['R_score'] * 0.5 + 
                                       rfm_scored['F_score'] * 0.3 + 
                                       rfm_scored['M_score'] * 0.2)
    
    print("RFM scoring completed")
    
    return rfm_scored

def segment_customers_rfm(rfm_scored):
    """
    Segment customers based on RFM scores
    
    Parameters:
    - rfm_scored: DataFrame with RFM scores
    
    Returns:
    - DataFrame with customer segments added
    """
    
    print("Segmenting customers based on RFM scores...")
    
    rfm_segmented = rfm_scored.copy()
    
    # Define segmentation rules
    def assign_segment(row):
        r_score = row['R_score']
        f_score = row['F_score']
        m_score = row['M_score']
        rfm_score = row['RFM_score']
        
        # High-Value Customers (High RFM scores)
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            return 'High-Value'
        
        # Loyal Customers (High frequency and monetary, moderate recency)
        elif f_score >= 4 and m_score >= 4 and r_score >= 3:
            return 'Loyal'
        
        # At-Risk Customers (Low recency, moderate frequency/monetary)
        elif r_score <= 2 and f_score >= 3 and m_score >= 3:
            return 'At-Risk'
        
        # New Customers (High recency, low frequency/monetary)
        elif r_score >= 4 and f_score <= 2 and m_score <= 2:
            return 'New'
        
        # Promising Customers (High recency and frequency, low monetary)
        elif r_score >= 4 and f_score >= 3 and m_score <= 2:
            return 'Promising'
        
        # Lost Customers (Low scores across all dimensions)
        elif r_score <= 2 and f_score <= 2 and m_score <= 2:
            return 'Lost'
        
        # Regular Customers (moderate scores)
        else:
            return 'Regular'
    
    # Apply segmentation
    rfm_segmented['segment'] = rfm_segmented.apply(assign_segment, axis=1)
    
    # Add segment priority for marketing
    segment_priority = {
        'High-Value': 1,
        'Loyal': 2,
        'At-Risk': 3,
        'Promising': 4,
        'New': 5,
        'Regular': 6,
        'Lost': 7
    }
    
    rfm_segmented['segment_priority'] = rfm_segmented['segment'].map(segment_priority)
    
    print("Customer segmentation completed")
    
    return rfm_segmented

def generate_rfm_summary(rfm_segmented):
    """Generate summary statistics for RFM analysis"""
    
    print("\n=== RFM ANALYSIS SUMMARY ===")
    
    # Basic statistics
    print(f"Total Customers Analyzed: {len(rfm_segmented)}")
    print(f"Average Recency: {rfm_segmented['recency'].mean():.1f} days")
    print(f"Average Frequency: {rfm_segmented['frequency'].mean():.1f} purchases")
    print(f"Average Monetary: ${rfm_segmented['monetary'].mean():.2f}")
    
    # Segment distribution
    print("\n=== CUSTOMER SEGMENT DISTRIBUTION ===")
    segment_dist = rfm_segmented['segment'].value_counts()
    for segment, count in segment_dist.items():
        percentage = (count / len(rfm_segmented)) * 100
        avg_monetary = rfm_segmented[rfm_segmented['segment'] == segment]['monetary'].mean()
        print(f"{segment}: {count} customers ({percentage:.1f}%) - Avg Value: ${avg_monetary:.2f}")
    
    # RFM Score distribution
    print("\n=== RFM SCORE DISTRIBUTION ===")
    rfm_score_dist = rfm_segmented['RFM_score'].value_counts().sort_index()
    for score, count in rfm_score_dist.items():
        percentage = (count / len(rfm_segmented)) * 100
        print(f"RFM Score {score}: {count} customers ({percentage:.1f}%)")
    
    # Top customers by RFM score
    print("\n=== TOP 10 CUSTOMERS BY RFM SCORE ===")
    top_customers = rfm_segmented.nlargest(10, 'RFM_score')[['customer_id', 'RFM_score', 'segment', 'monetary']]
    for _, row in top_customers.iterrows():
        print(f"Customer {row['customer_id']}: RFM Score {row['RFM_score']}, {row['segment']}, ${row['monetary']:.2f}")

def save_rfm_results(rfm_segmented):
    """Save RFM analysis results"""
    
    # Create processed data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save RFM scores
    rfm_segmented.to_csv('data/processed/rfm_scores.csv', index=False)
    print("Saved rfm_scores.csv")
    
    # Save customer segments
    segments_df = rfm_segmented[['customer_id', 'segment', 'segment_priority', 'RFM_score', 'RFM_weighted_score']]
    segments_df.to_csv('data/processed/customer_segments.csv', index=False)
    print("Saved customer_segments.csv")
    
    # Create summary report
    summary_stats = {
        'total_customers': len(rfm_segmented),
        'avg_recency': rfm_segmented['recency'].mean(),
        'avg_frequency': rfm_segmented['frequency'].mean(),
        'avg_monetary': rfm_segmented['monetary'].mean(),
        'high_value_customers': len(rfm_segmented[rfm_segmented['segment'] == 'High-Value']),
        'loyal_customers': len(rfm_segmented[rfm_segmented['segment'] == 'Loyal']),
        'at_risk_customers': len(rfm_segmented[rfm_segmented['segment'] == 'At-Risk']),
        'new_customers': len(rfm_segmented[rfm_segmented['segment'] == 'New']),
        'lost_customers': len(rfm_segmented[rfm_segmented['segment'] == 'Lost'])
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('data/processed/rfm_summary.csv', index=False)
    print("Saved rfm_summary.csv")

def main():
    """Main function to perform RFM analysis"""
    
    print("Customer Segmentation & LTV Analysis - RFM Analysis")
    print("=" * 60)
    
    # Load data
    customers_df, transactions_df = load_data()
    if customers_df is None or transactions_df is None:
        return
    
    # Calculate RFM scores
    rfm_df = calculate_rfm_scores(transactions_df)
    
    # Score RFM values
    rfm_scored = score_rfm(rfm_df)
    
    # Segment customers
    rfm_segmented = segment_customers_rfm(rfm_scored)
    
    # Generate summary
    generate_rfm_summary(rfm_segmented)
    
    # Save results
    save_rfm_results(rfm_segmented)
    
    print("\n" + "=" * 60)
    print("RFM analysis completed successfully!")
    print("Results saved in ../data/processed/ directory")
    print("Next steps:")
    print("1. Run customer_clustering.py for K-Means segmentation")
    print("2. Run ltv_prediction.py for LTV modeling")

if __name__ == "__main__":
    main()
