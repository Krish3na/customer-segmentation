"""
Kaggle RFM Analysis for Customer Segmentation & LTV Analysis

Performs RFM analysis on Kaggle datasets with different data structures.
Handles multiple Kaggle dataset formats and provides unified analysis.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class KaggleRFMAnalysis:
    def __init__(self):
        self.dataset_type = None
        self.customers_df = None
        self.transactions_df = None
        
    def load_kaggle_data(self, dataset_type='retail_customer'):
        """Loads Kaggle dataset based on type"""
        self.dataset_type = dataset_type
        
        if dataset_type == 'retail_customer':
            return self._load_retail_customer_data()
        elif dataset_type == 'ecommerce_customer':
            return self._load_ecommerce_customer_data()
        elif dataset_type == 'online_retail':
            return self._load_online_retail_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_retail_customer_data(self):
        """Loads retail customer dataset"""
        try:
            customers_df = pd.read_csv('data/kaggle/raw/kaggle_retail_customers.csv')
            transactions_df = pd.read_csv('data/kaggle/raw/kaggle_retail_transactions.csv')
            
            # Convert date columns
            transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
            
            # Create a copy with renamed columns to avoid conflicts
            transactions_renamed = transactions_df.copy()
            transactions_renamed = transactions_renamed.rename(columns={
                'TransactionDate': 'transaction_date',
                'TotalAmount': 'total_amount',
                'CustomerID': 'customer_id'
            })
            
            # Rename customer ID in customers file to match
            customers_renamed = customers_df.copy()
            customers_renamed = customers_renamed.rename(columns={'CustomerID': 'customer_id'})
            
            self.customers_df = customers_renamed
            self.transactions_df = transactions_renamed
            
            print(f"✅ Loaded retail customer data: {len(customers_df)} customers, {len(transactions_df)} transactions")
            return True
            
        except FileNotFoundError:
            print("❌ Kaggle retail customer data not found. Please run kaggle_data_processor.py first.")
            return False
    
    def _load_ecommerce_customer_data(self):
        """Loads e-commerce customer dataset"""
        try:
            customers_df = pd.read_csv('data/kaggle/raw/kaggle_ecommerce_customers.csv')
            orders_df = pd.read_csv('data/kaggle/raw/kaggle_ecommerce_orders.csv')
            order_items_df = pd.read_csv('data/kaggle/raw/kaggle_ecommerce_order_items.csv')
            
            # Convert date columns
            orders_df['order_purchase_date'] = pd.to_datetime(orders_df['order_purchase_date'])
            
            # Merge orders with order items to get transaction data
            transactions_df = orders_df.merge(order_items_df, on='order_id', how='inner')
            
            # Calculate total amount per transaction
            transactions_df['total_amount'] = transactions_df['price'] + transactions_df['freight_value']
            
            # Rename columns to match our standard format
            transactions_df = transactions_df.rename(columns={
                'customer_id': 'customer_id',
                'order_purchase_date': 'transaction_date',
                'total_amount': 'total_amount'
            })
            
            self.customers_df = customers_df
            self.transactions_df = transactions_df
            
            print(f"✅ Loaded e-commerce data: {len(customers_df)} customers, {len(transactions_df)} transactions")
            return True
            
        except FileNotFoundError:
            print("❌ Kaggle e-commerce data not found. Please run kaggle_data_processor.py first.")
            return False
    
    def _load_online_retail_data(self):
        """Loads online retail dataset"""
        try:
            customers_df = pd.read_csv('data/kaggle/raw/kaggle_online_retail_customers.csv')
            transactions_df = pd.read_csv('data/kaggle/raw/kaggle_online_retail_transactions.csv')
            
            # Convert date columns
            transactions_df['InvoiceDate'] = pd.to_datetime(transactions_df['InvoiceDate'])
            
            # Filter out cancelled transactions (Invoice starting with 'C')
            transactions_df = transactions_df[~transactions_df['Invoice'].str.startswith('C')]
            
            # Create copies with renamed columns
            customers_renamed = customers_df.copy()
            customers_renamed = customers_renamed.rename(columns={'CustomerID': 'customer_id'})
            
            transactions_renamed = transactions_df.copy()
            transactions_renamed = transactions_renamed.rename(columns={
                'CustomerID': 'customer_id',
                'InvoiceDate': 'transaction_date',
                'TotalAmount': 'total_amount'
            })
            
            self.customers_df = customers_renamed
            self.transactions_df = transactions_renamed
            
            print(f"✅ Loaded online retail data: {len(customers_df)} customers, {len(transactions_df)} transactions")
            return True
            
        except FileNotFoundError:
            print("❌ Kaggle online retail data not found. Please run kaggle_data_processor.py first.")
            return False
    
    def calculate_rfm_scores(self):
        """Calculates RFM scores for the loaded dataset"""
        
        if self.customers_df is None or self.transactions_df is None:
            print("❌ No data loaded. Please load data first.")
            return None
        
        print("🔄 Calculating RFM scores...")
        
        # Get current date (use max transaction date as reference)
        current_date = self.transactions_df['transaction_date'].max()
        
        # Calculate RFM metrics
        rfm_data = self.transactions_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'total_amount': 'sum'  # Monetary
        }).reset_index()
        
        # Add frequency count
        frequency_counts = self.transactions_df.groupby('customer_id').size().reset_index(name='frequency')
        rfm_data = rfm_data.merge(frequency_counts, on='customer_id')
        
        # Rename columns for clarity
        rfm_data = rfm_data.rename(columns={'transaction_date': 'recency', 'total_amount': 'monetary'})
        
        # Calculate RFM scores (1-5 scale)
        try:
            rfm_data['R_score'] = pd.qcut(rfm_data['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            rfm_data['F_score'] = pd.qcut(rfm_data['frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rfm_data['M_score'] = pd.qcut(rfm_data['monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            # If qcut fails, use cut instead
            rfm_data['R_score'] = pd.cut(rfm_data['recency'], bins=5, labels=[5, 4, 3, 2, 1])
            rfm_data['F_score'] = pd.cut(rfm_data['frequency'], bins=5, labels=[1, 2, 3, 4, 5])
            rfm_data['M_score'] = pd.cut(rfm_data['monetary'], bins=5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        rfm_data['R_score'] = rfm_data['R_score'].astype(int)
        rfm_data['F_score'] = rfm_data['F_score'].astype(int)
        rfm_data['M_score'] = rfm_data['M_score'].astype(int)
        
        # Calculate combined RFM score
        rfm_data['RFM_score'] = rfm_data['R_score'] + rfm_data['F_score'] + rfm_data['M_score']
        
        # Create customer segments
        rfm_data['segment'] = rfm_data['RFM_score'].apply(self._assign_segment)
        
        # Add customer information
        if self.dataset_type == 'retail_customer':
            rfm_data = rfm_data.merge(self.customers_df[['customer_id', 'Gender', 'Age', 'Annual_Income_k', 'Spending_Score_1_100']], 
                                     on='customer_id', how='left')
        elif self.dataset_type == 'ecommerce_customer':
            rfm_data = rfm_data.merge(self.customers_df[['customer_id', 'customer_city', 'customer_state']], 
                                     on='customer_id', how='left')
        elif self.dataset_type == 'online_retail':
            rfm_data = rfm_data.merge(self.customers_df[['customer_id', 'Country']], 
                                     on='customer_id', how='left')
        
        return rfm_data
    
    def _assign_segment(self, rfm_score):
        """Assigns customer segment based on RFM score"""
        if rfm_score >= 13:
            return 'High-Value'
        elif rfm_score >= 11:
            return 'Loyal'
        elif rfm_score >= 9:
            return 'At-Risk'
        elif rfm_score >= 7:
            return 'New'
        else:
            return 'Lost'
    
    def create_customer_segments(self, rfm_data):
        """Creates detailed customer segments with additional metrics"""
        
        # Add segment priority for marketing
        segment_priority = {
            'High-Value': 1,
            'Loyal': 2,
            'At-Risk': 3,
            'New': 4,
            'Lost': 5
        }
        
        rfm_data['segment_priority'] = rfm_data['segment'].map(segment_priority)
        
        # Calculate weighted RFM score (R*0.5 + F*0.3 + M*0.2)
        rfm_data['RFM_weighted_score'] = (
            rfm_data['R_score'] * 0.5 + 
            rfm_data['F_score'] * 0.3 + 
            rfm_data['M_score'] * 0.2
        )
        
        return rfm_data
    
    def save_results(self, rfm_data, customer_segments):
        """Saves RFM analysis results to Kaggle processed folder"""
        
        # Create processed directory if it doesn't exist
        os.makedirs('data/kaggle/processed', exist_ok=True)
        
        # Save RFM scores
        rfm_data.to_csv('data/kaggle/processed/kaggle_rfm_scores.csv', index=False)
        print("✅ Saved kaggle_rfm_scores.csv")
        
        # Save customer segments
        segments_df = rfm_data[['customer_id', 'segment', 'segment_priority', 'RFM_score', 'RFM_weighted_score']]
        segments_df.to_csv('data/kaggle/processed/kaggle_customer_segments.csv', index=False)
        print("✅ Saved kaggle_customer_segments.csv")
        
        # Save segment statistics
        segment_stats = rfm_data.groupby('segment').agg({
            'recency': ['mean', 'std'],
            'frequency': ['mean', 'std'],
            'monetary': ['mean', 'sum'],
            'RFM_score': 'mean'
        }).round(2)
        
        segment_stats.to_csv('data/kaggle/processed/kaggle_segment_statistics.csv')
        print("✅ Saved kaggle_segment_statistics.csv")
        
        return True
    
    def print_summary(self, rfm_data):
        """Prints summary of RFM analysis"""
        
        print("\n📊 Kaggle RFM Analysis Summary")
        print("=" * 40)
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Total Customers: {len(rfm_data):,}")
        print(f"Total Transactions: {len(self.transactions_df):,}")
        print(f"Date Range: {self.transactions_df['transaction_date'].min()} to {self.transactions_df['transaction_date'].max()}")
        
        print("\n🎯 Customer Segments:")
        segment_counts = rfm_data['segment'].value_counts()
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_data)) * 100
            print(f"  • {segment}: {count:,} customers ({percentage:.1f}%)")
        
        print("\n💰 Revenue by Segment:")
        segment_revenue = rfm_data.groupby('segment')['monetary'].sum().sort_values(ascending=False)
        for segment, revenue in segment_revenue.items():
            print(f"  • {segment}: ${revenue:,.2f}")
        
        print("\n📈 RFM Score Distribution:")
        rfm_score_counts = rfm_data['RFM_score'].value_counts().sort_index()
        for score, count in rfm_score_counts.items():
            percentage = (count / len(rfm_data)) * 100
            print(f"  • Score {score}: {count:,} customers ({percentage:.1f}%)")

def main():
    """Main function to run Kaggle RFM analysis"""
    
    print("🎯 Kaggle RFM Analysis for Customer Segmentation")
    print("=" * 50)
    
    # Available dataset types
    dataset_types = ['retail_customer', 'ecommerce_customer', 'online_retail']
    
    for dataset_type in dataset_types:
        print(f"\n🔄 Processing {dataset_type} dataset...")
        
        # Initialize RFM analysis
        rfm_analyzer = KaggleRFMAnalysis()
        
        # Load data
        if rfm_analyzer.load_kaggle_data(dataset_type):
            # Calculate RFM scores
            rfm_data = rfm_analyzer.calculate_rfm_scores()
            
            if rfm_data is not None:
                # Create customer segments
                customer_segments = rfm_analyzer.create_customer_segments(rfm_data)
                
                # Save results
                rfm_analyzer.save_results(rfm_data, customer_segments)
                
                # Print summary
                rfm_analyzer.print_summary(rfm_data)
                
                print(f"✅ {dataset_type} RFM analysis completed successfully!")
            else:
                print(f"❌ Failed to calculate RFM scores for {dataset_type}")
        else:
            print(f"❌ Failed to load {dataset_type} data")
    
    print("\n🎉 All Kaggle RFM analysis completed!")
    print("📁 Results saved in: data/kaggle/processed/")

if __name__ == "__main__":
    main()
