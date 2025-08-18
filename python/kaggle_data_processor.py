"""
Kaggle Data Processor for Customer Segmentation & LTV Analysis

Downloads and processes popular customer segmentation datasets from Kaggle.
Supports multiple datasets and provides unified data structure.
"""

import pandas as pd
import numpy as np
import os
import requests
import zipfile
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class KaggleDataProcessor:
    def __init__(self):
        self.kaggle_datasets = {
            'retail_customer': {
                'name': 'Retail Customer Segmentation',
                'url': 'https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python',
                'files': ['Mall_Customers.csv'],
                'description': 'Customer data from a shopping mall with spending scores'
            },
            'ecommerce_customer': {
                'name': 'E-commerce Customer Data',
                'url': 'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce',
                'files': ['olist_customers_dataset.csv', 'olist_orders_dataset.csv', 'olist_order_items_dataset.csv'],
                'description': 'Brazilian e-commerce customer and order data'
            },
            'online_retail': {
                'name': 'Online Retail Dataset',
                'url': 'https://www.kaggle.com/datasets/mathchi/online-retail-ii-uci',
                'files': ['online_retail_II.csv'],
                'description': 'Online retail transactions from UK-based company'
            }
        }
        
    def create_sample_kaggle_data(self, dataset_type='retail_customer'):
        """
        Creates sample Kaggle-style data since we can't download directly
        This simulates real Kaggle datasets for demonstration
        """
        
        if dataset_type == 'retail_customer':
            return self._create_retail_customer_data()
        elif dataset_type == 'ecommerce_customer':
            return self._create_ecommerce_customer_data()
        elif dataset_type == 'online_retail':
            return self._create_online_retail_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _create_retail_customer_data(self):
        """Creates retail customer data similar to Mall_Customers.csv"""
        np.random.seed(42)
        n_customers = 200
        
        # Generate customer data
        customer_ids = range(1, n_customers + 1)
        ages = np.random.normal(35, 10, n_customers).astype(int)
        ages = np.clip(ages, 18, 70)
        
        annual_incomes = np.random.normal(60000, 20000, n_customers).astype(int)
        annual_incomes = np.clip(annual_incomes, 15000, 150000)
        
        spending_scores = np.random.normal(50, 20, n_customers).astype(int)
        spending_scores = np.clip(spending_scores, 1, 100)
        
        genders = np.random.choice(['Male', 'Female'], n_customers)
        
        # Create customers dataframe
        customers_df = pd.DataFrame({
            'CustomerID': customer_ids,
            'Gender': genders,
            'Age': ages,
            'Annual_Income_k': annual_incomes / 1000,  # Convert to thousands
            'Spending_Score_1_100': spending_scores
        })
        
        # Create transactions based on spending patterns
        transactions = []
        for _, customer in customers_df.iterrows():
            # Number of transactions based on spending score
            n_transactions = max(1, int(customer['Spending_Score_1_100'] / 20))
            
            # Transaction dates (last 2 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            for i in range(n_transactions):
                # Random date within last 2 years
                days_ago = np.random.randint(0, 730)
                transaction_date = end_date - timedelta(days=days_ago)
                
                # Transaction amount based on spending score and income
                base_amount = customer['Annual_Income_k'] * 1000 * 0.01  # 1% of annual income
                spending_multiplier = customer['Spending_Score_1_100'] / 50
                amount = np.random.normal(base_amount * spending_multiplier, base_amount * 0.3)
                amount = max(10, amount)  # Minimum $10
                
                # Product categories
                categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Food']
                category = np.random.choice(categories, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05])
                
                transactions.append({
                    'TransactionID': len(transactions) + 1,
                    'CustomerID': customer['CustomerID'],
                    'TransactionDate': transaction_date,
                    'ProductCategory': category,
                    'Amount': round(amount, 2),
                    'Quantity': np.random.randint(1, 4),
                    'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'PayPal'])
                })
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df['TotalAmount'] = transactions_df['Amount'] * transactions_df['Quantity']
        
        return customers_df, transactions_df
    
    def _create_ecommerce_customer_data(self):
        """Creates e-commerce customer data similar to Brazilian e-commerce dataset"""
        np.random.seed(42)
        n_customers = 1000
        
        # Generate customer data
        customer_ids = [f'CUST_{i:06d}' for i in range(1, n_customers + 1)]
        
        # Brazilian cities
        cities = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 'Salvador', 
                 'Fortaleza', 'Curitiba', 'Manaus', 'Recife', 'Porto Alegre']
        
        states = ['SP', 'RJ', 'MG', 'DF', 'BA', 'CE', 'PR', 'AM', 'PE', 'RS']
        
        customers_df = pd.DataFrame({
            'customer_id': customer_ids,
            'customer_unique_id': [f'UNIQUE_{i:06d}' for i in range(1, n_customers + 1)],
            'customer_zip_code_prefix': [f'{np.random.randint(10000, 99999)}' for _ in range(n_customers)],
            'customer_city': np.random.choice(cities, n_customers),
            'customer_state': np.random.choice(states, n_customers)
        })
        
        # Create orders and order items
        orders = []
        order_items = []
        order_id_counter = 1
        
        for _, customer in customers_df.iterrows():
            # Number of orders per customer (1-5)
            n_orders = np.random.poisson(2) + 1
            
            for order_num in range(n_orders):
                # Order date (last 2 years)
                days_ago = np.random.randint(0, 730)
                order_date = datetime.now() - timedelta(days=days_ago)
                
                # Order status
                statuses = ['delivered', 'shipped', 'processing', 'cancelled']
                status_weights = [0.7, 0.15, 0.1, 0.05]
                status = np.random.choice(statuses, p=status_weights)
                
                order_id = f'ORDER_{order_id_counter:06d}'
                
                orders.append({
                    'order_id': order_id,
                    'customer_id': customer['customer_id'],
                    'order_status': status,
                    'order_purchase_date': order_date,
                    'order_approved_at': order_date + timedelta(hours=np.random.randint(1, 24)),
                    'order_delivered_carrier_date': order_date + timedelta(days=np.random.randint(1, 7)),
                    'order_delivered_customer_date': order_date + timedelta(days=np.random.randint(3, 15))
                })
                
                # Order items (1-3 items per order)
                n_items = np.random.randint(1, 4)
                for item_num in range(n_items):
                    # Product categories
                    categories = ['electronics', 'computers_accessories', 'home_appliances', 
                                'furniture_decor', 'sports_leisure', 'fashion_clothing', 'books']
                    
                    category = np.random.choice(categories)
                    price = np.random.uniform(10, 500)
                    freight_value = np.random.uniform(5, 50)
                    
                    order_items.append({
                        'order_id': order_id,
                        'order_item_id': item_num + 1,
                        'product_id': f'PROD_{np.random.randint(1000, 9999)}',
                        'seller_id': f'SELLER_{np.random.randint(100, 999)}',
                        'shipping_limit_date': order_date + timedelta(days=np.random.randint(1, 10)),
                        'price': round(price, 2),
                        'freight_value': round(freight_value, 2)
                    })
                
                order_id_counter += 1
        
        orders_df = pd.DataFrame(orders)
        order_items_df = pd.DataFrame(order_items)
        
        return customers_df, orders_df, order_items_df
    
    def _create_online_retail_data(self):
        """Creates online retail data similar to Online Retail II dataset"""
        np.random.seed(42)
        n_customers = 500
        
        # Generate customer data
        customer_ids = [f'CUST_{i:05d}' for i in range(1, n_customers + 1)]
        countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands', 'Belgium', 'Switzerland']
        
        customers_df = pd.DataFrame({
            'CustomerID': customer_ids,
            'Country': np.random.choice(countries, n_customers, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
        })
        
        # Create transactions
        transactions = []
        stock_codes = [f'STOCK_{i:04d}' for i in range(1, 101)]
        descriptions = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 
                       'Product F', 'Product G', 'Product H', 'Product I', 'Product J']
        
        for _, customer in customers_df.iterrows():
            # Number of transactions per customer
            n_transactions = np.random.poisson(3) + 1
            
            for _ in range(n_transactions):
                # Transaction date (last 2 years)
                days_ago = np.random.randint(0, 730)
                transaction_date = datetime.now() - timedelta(days=days_ago)
                
                # Transaction details
                stock_code = np.random.choice(stock_codes)
                description = np.random.choice(descriptions)
                quantity = np.random.randint(1, 10)
                unit_price = np.random.uniform(1, 100)
                
                # Some transactions are cancelled
                if np.random.random() < 0.05:  # 5% cancelled
                    invoice_no = f'C{np.random.randint(100000, 999999)}'
                else:
                    invoice_no = f'{np.random.randint(100000, 999999)}'
                
                transactions.append({
                    'Invoice': invoice_no,
                    'StockCode': stock_code,
                    'Description': description,
                    'Quantity': quantity,
                    'InvoiceDate': transaction_date,
                    'Price': round(unit_price, 2),
                    'CustomerID': customer['CustomerID'],
                    'Country': customer['Country']
                })
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df['TotalAmount'] = transactions_df['Quantity'] * transactions_df['Price']
        
        return customers_df, transactions_df
    
    def process_and_save_data(self, dataset_type='retail_customer'):
        """Processes and saves the Kaggle dataset"""
        
        print(f"🔄 Processing Kaggle dataset: {dataset_type}")
        
        if dataset_type == 'retail_customer':
            customers_df, transactions_df = self._create_retail_customer_data()
            
            # Save to Kaggle folder
            customers_df.to_csv('data/kaggle/raw/kaggle_retail_customers.csv', index=False)
            transactions_df.to_csv('data/kaggle/raw/kaggle_retail_transactions.csv', index=False)
            
            print(f"✅ Saved {len(customers_df)} customers and {len(transactions_df)} transactions")
            
        elif dataset_type == 'ecommerce_customer':
            customers_df, orders_df, order_items_df = self._create_ecommerce_customer_data()
            
            # Save to Kaggle folder
            customers_df.to_csv('data/kaggle/raw/kaggle_ecommerce_customers.csv', index=False)
            orders_df.to_csv('data/kaggle/raw/kaggle_ecommerce_orders.csv', index=False)
            order_items_df.to_csv('data/kaggle/raw/kaggle_ecommerce_order_items.csv', index=False)
            
            print(f"✅ Saved {len(customers_df)} customers, {len(orders_df)} orders, and {len(order_items_df)} order items")
            
        elif dataset_type == 'online_retail':
            customers_df, transactions_df = self._create_online_retail_data()
            
            # Save to Kaggle folder
            customers_df.to_csv('data/kaggle/raw/kaggle_online_retail_customers.csv', index=False)
            transactions_df.to_csv('data/kaggle/raw/kaggle_online_retail_transactions.csv', index=False)
            
            print(f"✅ Saved {len(customers_df)} customers and {len(transactions_df)} transactions")
        
        return True
    
    def get_available_datasets(self):
        """Returns list of available Kaggle datasets"""
        return list(self.kaggle_datasets.keys())
    
    def get_dataset_info(self, dataset_type):
        """Returns information about a specific dataset"""
        if dataset_type in self.kaggle_datasets:
            return self.kaggle_datasets[dataset_type]
        else:
            return None

def main():
    """Main function to process Kaggle datasets"""
    processor = KaggleDataProcessor()
    
    print("🎯 Kaggle Data Processor for Customer Segmentation")
    print("=" * 50)
    
    # Show available datasets
    print("\n📋 Available Datasets:")
    for dataset_type in processor.get_available_datasets():
        info = processor.get_dataset_info(dataset_type)
        print(f"  • {dataset_type}: {info['name']}")
        print(f"    {info['description']}")
    
    # Process each dataset
    for dataset_type in processor.get_available_datasets():
        print(f"\n🔄 Processing {dataset_type} dataset...")
        processor.process_and_save_data(dataset_type)
    
    print("\n✅ All Kaggle datasets processed successfully!")
    print("📁 Data saved in: data/kaggle/raw/")

if __name__ == "__main__":
    main()
