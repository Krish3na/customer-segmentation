"""
Simple Customer Segmentation & LTV Analysis - K-Means Clustering Script

This script performs K-Means clustering on customer data using RFM scores.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

def load_data():
    """Load customer data and RFM scores"""
    print("Loading customer data and RFM scores...")
    
    try:
        customers_df = pd.read_csv('../data/raw/customers.csv')
        transactions_df = pd.read_csv('../data/raw/transactions.csv')
        rfm_scores_df = pd.read_csv('data/processed/rfm_scores.csv')
        print(f"Loaded {len(customers_df)} customers, {len(transactions_df)} transactions, and RFM scores")
        return customers_df, transactions_df, rfm_scores_df
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please run data generation and RFM analysis first. {e}")
        return None, None, None

def prepare_features(customers_df, transactions_df, rfm_scores_df):
    """Prepare features for clustering"""
    
    print("Preparing features for clustering...")
    
    # Start with RFM scores
    features_df = rfm_scores_df.copy()
    
    # Add customer demographics
    features_df = features_df.merge(customers_df, on='customer_id', how='left')
    
    # Add transaction-based features
    transaction_features = transactions_df.groupby('customer_id').agg({
        'product_category': 'nunique',
        'payment_method': 'nunique',
        'total_amount': ['mean', 'std'],
        'quantity': 'mean'
    }).reset_index()
    
    # Flatten column names
    transaction_features.columns = ['customer_id', 'product_categories', 'payment_methods', 'avg_amount', 'amount_std', 'avg_quantity']
    
    # Merge transaction features
    features_df = features_df.merge(transaction_features, on='customer_id', how='left')
    
    # Add time-based features
    features_df['customer_lifetime_days'] = (
        pd.to_datetime(features_df['last_purchase_date']) - 
        pd.to_datetime(features_df['first_purchase_date'])
    ).dt.days
    
    features_df['avg_days_between_purchases'] = features_df['customer_lifetime_days'] / features_df['frequency']
    
    # Select core features for clustering
    feature_columns = [
        'recency', 'frequency', 'monetary',
        'R_score', 'F_score', 'M_score',
        'RFM_score', 'RFM_weighted_score',
        'avg_order_value', 'customer_lifetime_days',
        'avg_days_between_purchases',
        'product_categories', 'payment_methods',
        'avg_amount', 'amount_std', 'avg_quantity'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    print(f"Selected {len(available_features)} features for clustering")
    
    return features_df, available_features

def determine_optimal_clusters(features_df, feature_columns, max_clusters=8):
    """Determine optimal number of clusters"""
    
    print("Determining optimal number of clusters...")
    
    # Prepare data
    X = features_df[feature_columns].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate inertia and silhouette scores
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Find optimal k using silhouette score
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    
    print(f"Silhouette analysis suggests {optimal_k} clusters")
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    ax1.plot(cluster_range, inertias, 'bo-')
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(cluster_range, silhouette_scores, 'go-')
    ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/processed/cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_k

def perform_clustering(features_df, feature_columns, n_clusters=5):
    """Perform K-Means clustering"""
    
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    
    # Prepare data
    X = features_df[feature_columns].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to features DataFrame
    features_df['cluster'] = cluster_labels
    
    # Calculate cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                  columns=feature_columns)
    cluster_centers['cluster'] = range(n_clusters)
    
    print("Clustering completed successfully")
    
    return features_df, cluster_centers, scaler

def analyze_clusters(features_df, cluster_centers):
    """Analyze cluster characteristics"""
    
    print("Analyzing cluster characteristics...")
    
    # Basic cluster statistics
    cluster_stats = features_df.groupby('cluster').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'RFM_score': 'mean',
        'avg_order_value': 'mean',
        'customer_lifetime_days': 'mean'
    }).round(2)
    
    cluster_stats.columns = ['customer_count', 'avg_recency', 'avg_frequency', 
                           'avg_monetary', 'avg_rfm_score', 'avg_order_value', 'avg_lifetime_days']
    
    # Calculate percentage of customers in each cluster
    total_customers = len(features_df)
    cluster_stats['percentage'] = (cluster_stats['customer_count'] / total_customers * 100).round(1)
    
    # Assign cluster names based on characteristics
    cluster_names = []
    for cluster_id in range(len(cluster_stats)):
        row = cluster_stats.iloc[cluster_id]
        
        if row['avg_rfm_score'] >= 12 and row['avg_monetary'] >= 2000:
            cluster_name = 'High-Value Champions'
        elif row['avg_frequency'] >= 10 and row['avg_monetary'] >= 1500:
            cluster_name = 'Loyal Customers'
        elif row['avg_recency'] <= 30 and row['avg_frequency'] >= 5:
            cluster_name = 'Recent Active'
        elif row['avg_recency'] >= 90 and row['avg_monetary'] >= 1000:
            cluster_name = 'At-Risk High-Value'
        elif row['avg_frequency'] <= 3 and row['avg_monetary'] <= 500:
            cluster_name = 'Occasional Buyers'
        else:
            cluster_name = f'Cluster {cluster_id}'
        
        cluster_names.append(cluster_name)
    
    cluster_stats['cluster_name'] = cluster_names
    
    print("\n=== CLUSTER ANALYSIS SUMMARY ===")
    for cluster_id in range(len(cluster_stats)):
        row = cluster_stats.iloc[cluster_id]
        print(f"\n{row['cluster_name']} (Cluster {cluster_id}):")
        print(f"  Customers: {row['customer_count']} ({row['percentage']}%)")
        print(f"  Avg Recency: {row['avg_recency']:.0f} days")
        print(f"  Avg Frequency: {row['avg_frequency']:.1f} purchases")
        print(f"  Avg Monetary: ${row['avg_monetary']:.2f}")
        print(f"  Avg RFM Score: {row['avg_rfm_score']:.1f}")
        print(f"  Avg Order Value: ${row['avg_order_value']:.2f}")
    
    return cluster_stats

def visualize_clusters(features_df, cluster_centers):
    """Create visualizations for cluster analysis"""
    
    print("Creating cluster visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RFM Score vs Monetary Value
    scatter1 = axes[0, 0].scatter(features_df['monetary'], features_df['RFM_score'], 
                                 c=features_df['cluster'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('Monetary Value ($)')
    axes[0, 0].set_ylabel('RFM Score')
    axes[0, 0].set_title('RFM Score vs Monetary Value by Cluster')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. Recency vs Frequency
    scatter2 = axes[0, 1].scatter(features_df['recency'], features_df['frequency'], 
                                 c=features_df['cluster'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('Recency (days)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Recency vs Frequency by Cluster')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. Cluster size distribution
    cluster_counts = features_df['cluster'].value_counts().sort_index()
    axes[1, 0].bar(range(len(cluster_counts)), cluster_counts.values, color='skyblue')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].set_title('Cluster Size Distribution')
    axes[1, 0].set_xticks(range(len(cluster_counts)))
    axes[1, 0].set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
    
    # 4. Average monetary value by cluster
    cluster_monetary = features_df.groupby('cluster')['monetary'].mean()
    axes[1, 1].bar(range(len(cluster_monetary)), cluster_monetary.values, color='lightcoral')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Average Monetary Value ($)')
    axes[1, 1].set_title('Average Monetary Value by Cluster')
    axes[1, 1].set_xticks(range(len(cluster_monetary)))
    axes[1, 1].set_xticklabels([f'Cluster {i}' for i in cluster_monetary.index])
    
    plt.tight_layout()
    plt.savefig('data/processed/cluster_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Cluster visualizations saved")

def save_clustering_results(features_df, cluster_stats, cluster_centers):
    """Save clustering results"""
    
    # Create processed data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save customer clusters
    customer_clusters = features_df[['customer_id', 'cluster', 'recency', 'frequency', 'monetary', 'RFM_score']]
    customer_clusters.to_csv('data/processed/customer_clusters.csv', index=False)
    print("Saved customer_clusters.csv")
    
    # Save cluster statistics
    cluster_stats.to_csv('data/processed/cluster_statistics.csv', index=False)
    print("Saved cluster_statistics.csv")
    
    # Save cluster centers
    cluster_centers.to_csv('data/processed/cluster_centers.csv', index=False)
    print("Saved cluster_centers.csv")

def main():
    """Main function to perform customer clustering"""
    
    print("Simple Customer Segmentation & LTV Analysis - K-Means Clustering")
    print("=" * 60)
    
    # Load data
    customers_df, transactions_df, rfm_scores_df = load_data()
    if customers_df is None or transactions_df is None or rfm_scores_df is None:
        return
    
    # Prepare features
    features_df, feature_columns = prepare_features(customers_df, transactions_df, rfm_scores_df)
    
    # Determine optimal number of clusters
    optimal_clusters = determine_optimal_clusters(features_df, feature_columns)
    
    # Perform clustering
    features_df, cluster_centers, scaler = perform_clustering(features_df, feature_columns, optimal_clusters)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(features_df, cluster_centers)
    
    # Create visualizations
    visualize_clusters(features_df, cluster_centers)
    
    # Save results
    save_clustering_results(features_df, cluster_stats, cluster_centers)
    
    print("\n" + "=" * 60)
    print("Customer clustering completed successfully!")
    print("Results saved in ../data/processed/ directory")
    print("Next steps:")
    print("1. Run ltv_prediction.py for LTV modeling")
    print("2. Create Power BI dashboard")

if __name__ == "__main__":
    main()
