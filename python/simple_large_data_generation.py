"""
Simple Large-Scale Customer Segmentation & LTV Analysis - Data Generation Script

This script generates a large, realistic customer transaction dataset (50k customers)
with sophisticated patterns that mimic real e-commerce behavior.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_large_customer_data(num_customers=50000):
    """Generate large-scale customer data with realistic patterns"""
    
    print(f"Generating data for {num_customers:,} customers...")
    
    # Customer demographics with realistic distributions
    np.random.seed(42)
    
    # Age distribution (realistic e-commerce demographics)
    ages = np.random.normal(35, 12, num_customers)
    ages = np.clip(ages, 18, 80).astype(int)
    
    # Gender distribution
    genders = np.random.choice(['Male', 'Female', 'Other'], 
                              size=num_customers, 
                              p=[0.48, 0.50, 0.02])
    
    # Geographic distribution (major cities)
    cities = np.random.choice([
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston'
    ], size=num_customers)
    
    # Customer types with realistic proportions
    customer_types = np.random.choice([
        'New', 'Regular', 'Premium', 'VIP', 'Wholesale'
    ], size=num_customers, p=[0.30, 0.45, 0.15, 0.08, 0.02])
    
    # Registration dates (last 3 years with realistic distribution)
    end_date = datetime.now()
    
    # More recent registrations are more common
    registration_days = np.random.exponential(365, num_customers)
    registration_days = np.clip(registration_days, 0, 3*365)
    registration_dates = [end_date - timedelta(days=int(days)) for days in registration_days]
    
    # Create customer dataframe
    customers_df = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'age': ages,
        'gender': genders,
        'city': cities,
        'customer_type': customer_types,
        'registration_date': registration_dates
    })
    
    return customers_df

def generate_large_transaction_data(customers_df, avg_transactions_per_customer=8):
    """Generate large-scale transaction data with realistic patterns"""
    
    print("Generating transaction data...")
    
    # Calculate total transactions
    total_transactions = int(len(customers_df) * avg_transactions_per_customer)
    
    # Product categories with realistic popularity
    categories = np.random.choice([
        'Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports',
        'Beauty', 'Toys', 'Automotive', 'Health', 'Food & Beverage'
    ], size=total_transactions, p=[
        0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.05, 0.04, 0.03, 0.02
    ])
    
    # Price ranges by category (realistic pricing)
    price_ranges = {
        'Electronics': (50, 2000),
        'Clothing': (20, 300),
        'Home & Garden': (30, 800),
        'Books': (10, 100),
        'Sports': (25, 500),
        'Beauty': (15, 200),
        'Toys': (20, 150),
        'Automotive': (50, 1000),
        'Health': (20, 300),
        'Food & Beverage': (5, 100)
    }
    
    # Generate prices based on category
    prices = []
    for category in categories:
        min_price, max_price = price_ranges[category]
        # Use log-normal distribution for realistic price distribution
        price = np.random.lognormal(np.log(min_price), 0.5)
        price = np.clip(price, min_price, max_price)
        prices.append(round(price, 2))
    
    # Quantities with realistic distribution
    quantities = np.random.choice([1, 2, 3, 4, 5], 
                                 size=total_transactions, 
                                 p=[0.70, 0.15, 0.08, 0.05, 0.02])
    
    # Payment methods with realistic preferences
    payment_methods = np.random.choice([
        'Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay'
    ], size=total_transactions, p=[0.40, 0.25, 0.20, 0.10, 0.05])
    
    # Transaction status (mostly completed)
    statuses = np.random.choice(['Completed', 'Refunded', 'Cancelled'], 
                               size=total_transactions, 
                               p=[0.92, 0.06, 0.02])
    
    # Generate transaction dates with realistic patterns
    # More recent transactions are more common
    transaction_days = np.random.exponential(180, total_transactions)
    transaction_days = np.clip(transaction_days, 0, 365)
    transaction_dates = [datetime.now() - timedelta(days=int(days)) for days in transaction_days]
    
    # Assign customers to transactions with realistic frequency patterns
    customer_ids = []
    for customer_id in customers_df['customer_id']:
        # VIP customers have more transactions
        customer_type = customers_df[customers_df['customer_id'] == customer_id]['customer_type'].iloc[0]
        if customer_type == 'VIP':
            num_transactions = np.random.poisson(15)
        elif customer_type == 'Premium':
            num_transactions = np.random.poisson(10)
        elif customer_type == 'Regular':
            num_transactions = np.random.poisson(6)
        else:
            num_transactions = np.random.poisson(3)
        
        customer_ids.extend([customer_id] * num_transactions)
    
    # Ensure we have enough transactions
    while len(customer_ids) < total_transactions:
        customer_ids.append(np.random.choice(customers_df['customer_id']))
    
    # Trim to exact size
    customer_ids = customer_ids[:total_transactions]
    
    # Create transaction dataframe
    transactions_df = pd.DataFrame({
        'transaction_id': range(1, total_transactions + 1),
        'customer_id': customer_ids,
        'transaction_date': transaction_dates,
        'product_category': categories,
        'price': prices,
        'quantity': quantities,
        'payment_method': payment_methods,
        'status': statuses
    })
    
    # Calculate total_amount
    transactions_df['total_amount'] = transactions_df['price'] * transactions_df['quantity']
    
    return transactions_df

def add_seasonal_patterns(transactions_df):
    """Add realistic seasonal patterns to transactions"""
    
    print("Adding seasonal patterns...")
    
    # Holiday season boost (November-December)
    holiday_mask = transactions_df['transaction_date'].dt.month.isin([11, 12])
    holiday_boost = np.random.normal(1.5, 0.3, sum(holiday_mask))
    holiday_boost = np.clip(holiday_boost, 1.0, 3.0)
    
    # Apply holiday boost to quantities and prices
    transactions_df.loc[holiday_mask, 'quantity'] = np.ceil(
        transactions_df.loc[holiday_mask, 'quantity'] * holiday_boost
    ).astype(int)
    
    transactions_df.loc[holiday_mask, 'price'] = (
        transactions_df.loc[holiday_mask, 'price'] * holiday_boost
    ).round(2)
    
    # Recalculate total_amount
    transactions_df['total_amount'] = transactions_df['price'] * transactions_df['quantity']
    
    return transactions_df

def main():
    """Main function to generate large dataset"""
    
    print("=== Large-Scale Customer Segmentation Data Generation ===")
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate customer data
    customers_df = generate_large_customer_data(50000)
    
    # Generate transaction data
    transactions_df = generate_large_transaction_data(customers_df, avg_transactions_per_customer=8)
    
    # Add seasonal patterns
    transactions_df = add_seasonal_patterns(transactions_df)
    
    # Save to raw data folder
    customers_df.to_csv('data/raw/customers_large.csv', index=False)
    transactions_df.to_csv('data/raw/transactions_large.csv', index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📊 Customers: {len(customers_df):,}")
    print(f"🛒 Transactions: {len(transactions_df):,}")
    print(f"💰 Total Revenue: ${transactions_df['total_amount'].sum():,.2f}")
    print(f"📁 Files saved to data/raw/")
    
    # Display sample statistics
    print(f"\n📈 Sample Statistics:")
    print(f"   Average transaction value: ${transactions_df['total_amount'].mean():.2f}")
    print(f"   Most popular category: {transactions_df['product_category'].mode().iloc[0]}")
    print(f"   Customer types: {customers_df['customer_type'].value_counts().to_dict()}")
    
    return customers_df, transactions_df

if __name__ == "__main__":
    customers_df, transactions_df = main()
