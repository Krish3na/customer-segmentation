"""
Kaggle Clustering Analysis for Customer Segmentation & LTV Analysis

Performs K-Means clustering on Kaggle datasets with different data structures.
Handles multiple Kaggle dataset formats and provides unified clustering analysis.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class KaggleClustering:
    def __init__(self):
        self.dataset_type = None
        self.rfm_data = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        
    def load_kaggle_rfm_data(self, dataset_type='retail_customer'):
        """Loads Kaggle RFM data for clustering"""
        self.dataset_type = dataset_type
        
        try:
            rfm_file = 'data/kaggle/processed/kaggle_rfm_scores.csv'
            if os.path.exists(rfm_file):
                self.rfm_data = pd.read_csv(rfm_file)
                print(f"✅ Loaded Kaggle RFM data: {len(self.rfm_data)} customers")
                return True
            else:
                print(f"❌ Kaggle RFM data not found. Please run kaggle_rfm_analysis.py first.")
                return False
        except Exception as e:
            print(f"❌ Error loading Kaggle RFM data: {e}")
            return False
    
    def prepare_features(self):
        """Prepares features for clustering based on dataset type"""
        
        if self.rfm_data is None:
            print("❌ No RFM data loaded. Please load data first.")
            return None
        
        print("🔄 Preparing features for clustering...")
        
        # Base RFM features
        base_features = ['recency', 'frequency', 'monetary']
        
        # Additional features based on dataset type
        if self.dataset_type == 'retail_customer':
            # Retail customer specific features
            additional_features = ['Age', 'Annual_Income_k', 'Spending_Score_1_100']
            available_features = [col for col in additional_features if col in self.rfm_data.columns]
            
        elif self.dataset_type == 'ecommerce_customer':
            # E-commerce specific features (if available)
            additional_features = []
            available_features = []
            
        elif self.dataset_type == 'online_retail':
            # Online retail specific features (if available)
            additional_features = []
            available_features = []
        
        # Combine all available features
        all_features = base_features + available_features
        available_features = [col for col in all_features if col in self.rfm_data.columns]
        
        print(f"📊 Using features: {available_features}")
        
        # Prepare feature matrix
        X = self.rfm_data[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return X, available_features
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Finds optimal number of clusters using elbow method and silhouette analysis"""
        
        print("🔍 Finding optimal number of clusters...")
        
        # Calculate inertia for different numbers of clusters
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:
                labels = kmeans.labels_
                silhouette_avg = silhouette_score(X, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point
        elbow_k = self._find_elbow_point(K_range, inertias)
        
        # Find best silhouette score
        best_silhouette_k = K_range[np.argmax(silhouette_scores)]
        
        print(f"📈 Elbow method suggests: {elbow_k} clusters")
        print(f"📈 Silhouette analysis suggests: {best_silhouette_k} clusters")
        
        # Choose optimal k (prefer silhouette if close, otherwise elbow)
        if abs(elbow_k - best_silhouette_k) <= 1:
            optimal_k = best_silhouette_k
            print(f"🎯 Using silhouette-based optimal k: {optimal_k}")
        else:
            optimal_k = elbow_k
            print(f"🎯 Using elbow-based optimal k: {optimal_k}")
        
        return optimal_k, K_range, inertias, silhouette_scores
    
    def _find_elbow_point(self, K_range, inertias):
        """Finds the elbow point in the inertia curve"""
        
        # Calculate the rate of change
        changes = np.diff(inertias)
        change_rate = np.abs(changes)
        
        # Find the point with maximum change rate
        elbow_idx = np.argmax(change_rate) + 1
        return K_range[elbow_idx]
    
    def perform_clustering(self, X, n_clusters):
        """Performs K-Means clustering"""
        
        print(f"🎯 Performing K-Means clustering with {n_clusters} clusters...")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels to RFM data
        self.rfm_data['cluster'] = cluster_labels
        
        print(f"✅ Clustering completed. Cluster distribution:")
        cluster_counts = self.rfm_data['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(self.rfm_data)) * 100
            print(f"  • Cluster {cluster}: {count:,} customers ({percentage:.1f}%)")
        
        return cluster_labels
    
    def analyze_clusters(self, X, feature_names):
        """Analyzes cluster characteristics"""
        
        print("📊 Analyzing cluster characteristics...")
        
        # Calculate cluster statistics
        cluster_stats = self.rfm_data.groupby('cluster').agg({
            'recency': ['mean', 'std'],
            'frequency': ['mean', 'std'],
            'monetary': ['mean', 'sum'],
            'RFM_score': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        
        # Add cluster centers
        cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans_model.cluster_centers_),
            columns=feature_names
        )
        cluster_centers.index = [f'Cluster_{i}' for i in range(len(cluster_centers))]
        
        return cluster_stats, cluster_centers
    
    def create_visualizations(self, X, feature_names, K_range, inertias, silhouette_scores):
        """Creates clustering visualizations"""
        
        print("📈 Creating clustering visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Kaggle Clustering Analysis - {self.dataset_type.replace("_", " ").title()}', fontsize=16)
        
        # 1. Elbow plot
        axes[0, 0].plot(K_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # 2. Silhouette plot
        axes[0, 1].plot(K_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True)
        
        # 3. Cluster distribution
        cluster_counts = self.rfm_data['cluster'].value_counts().sort_index()
        axes[1, 0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], autopct='%1.1f%%')
        axes[1, 0].set_title('Cluster Distribution')
        
        # 4. Feature importance by cluster (box plot for first 3 features)
        if len(feature_names) >= 3:
            feature_data = []
            cluster_labels = []
            feature_names_plot = []
            
            for i, feature in enumerate(feature_names[:3]):
                for cluster in sorted(self.rfm_data['cluster'].unique()):
                    cluster_data = self.rfm_data[self.rfm_data['cluster'] == cluster][feature]
                    feature_data.extend(cluster_data)
                    cluster_labels.extend([f'Cluster {cluster}'] * len(cluster_data))
                    feature_names_plot.extend([feature] * len(cluster_data))
            
            plot_df = pd.DataFrame({
                'Feature': feature_names_plot,
                'Value': feature_data,
                'Cluster': cluster_labels
            })
            
            sns.boxplot(data=plot_df, x='Feature', y='Value', hue='Cluster', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Distribution by Cluster')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, cluster_stats, cluster_centers, fig):
        """Saves clustering results"""
        
        # Create processed directory if it doesn't exist
        os.makedirs('data/kaggle/processed', exist_ok=True)
        
        # Save customer clusters
        customer_clusters = self.rfm_data[['customer_id', 'cluster', 'recency', 'frequency', 'monetary', 'RFM_score']]
        customer_clusters.to_csv('data/kaggle/processed/kaggle_customer_clusters.csv', index=False)
        print("✅ Saved kaggle_customer_clusters.csv")
        
        # Save cluster statistics
        cluster_stats.to_csv('data/kaggle/processed/kaggle_cluster_statistics.csv')
        print("✅ Saved kaggle_cluster_statistics.csv")
        
        # Save cluster centers
        cluster_centers.to_csv('data/kaggle/processed/kaggle_cluster_centers.csv')
        print("✅ Saved kaggle_cluster_centers.csv")
        
        # Save visualization
        if fig:
            fig.savefig('data/kaggle/processed/kaggle_cluster_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved kaggle_cluster_analysis.png")
        
        return True
    
    def print_summary(self, cluster_stats, cluster_centers):
        """Prints clustering summary"""
        
        print("\n📊 Kaggle Clustering Analysis Summary")
        print("=" * 40)
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Total Customers: {len(self.rfm_data):,}")
        print(f"Number of Clusters: {len(cluster_stats)}")
        
        print("\n🎯 Cluster Statistics:")
        print(cluster_stats)
        
        print("\n📍 Cluster Centers:")
        print(cluster_centers)
        
        print("\n💰 Revenue by Cluster:")
        cluster_revenue = self.rfm_data.groupby('cluster')['monetary'].sum().sort_values(ascending=False)
        for cluster, revenue in cluster_revenue.items():
            print(f"  • Cluster {cluster}: ${revenue:,.2f}")

def main():
    """Main function to run Kaggle clustering analysis"""
    
    print("🎯 Kaggle Clustering Analysis for Customer Segmentation")
    print("=" * 50)
    
    # Available dataset types
    dataset_types = ['retail_customer', 'ecommerce_customer', 'online_retail']
    
    for dataset_type in dataset_types:
        print(f"\n🔄 Processing {dataset_type} dataset...")
        
        # Initialize clustering
        clustering = KaggleClustering()
        
        # Load RFM data
        if clustering.load_kaggle_rfm_data(dataset_type):
            # Prepare features
            X, feature_names = clustering.prepare_features()
            
            if X is not None:
                # Find optimal number of clusters
                optimal_k, K_range, inertias, silhouette_scores = clustering.find_optimal_clusters(X)
                
                # Perform clustering
                cluster_labels = clustering.perform_clustering(X, optimal_k)
                
                # Analyze clusters
                cluster_stats, cluster_centers = clustering.analyze_clusters(X, feature_names)
                
                # Create visualizations
                fig = clustering.create_visualizations(X, feature_names, K_range, inertias, silhouette_scores)
                
                # Save results
                clustering.save_results(cluster_stats, cluster_centers, fig)
                
                # Print summary
                clustering.print_summary(cluster_stats, cluster_centers)
                
                print(f"✅ {dataset_type} clustering analysis completed successfully!")
            else:
                print(f"❌ Failed to prepare features for {dataset_type}")
        else:
            print(f"❌ Failed to load {dataset_type} RFM data")
    
    print("\n🎉 All Kaggle clustering analysis completed!")
    print("📁 Results saved in: data/kaggle/processed/")

if __name__ == "__main__":
    main()
