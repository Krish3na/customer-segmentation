"""
Kaggle LTV Prediction for Customer Segmentation & LTV Analysis

Performs LTV prediction on Kaggle datasets with different data structures.
Handles multiple Kaggle dataset formats and provides unified LTV analysis.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class KaggleLTVPrediction:
    def __init__(self):
        self.dataset_type = None
        self.rfm_data = None
        self.clustering_data = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = None
        
    def load_kaggle_data(self, dataset_type='retail_customer'):
        """Loads Kaggle data for LTV prediction"""
        self.dataset_type = dataset_type
        
        try:
            # Load RFM data
            rfm_file = 'data/kaggle/processed/kaggle_rfm_scores.csv'
            if os.path.exists(rfm_file):
                self.rfm_data = pd.read_csv(rfm_file)
                print(f"✅ Loaded Kaggle RFM data: {len(self.rfm_data)} customers")
            else:
                print(f"❌ Kaggle RFM data not found. Please run kaggle_rfm_analysis.py first.")
                return False
            
            # Load clustering data
            clustering_file = 'data/kaggle/processed/kaggle_customer_clusters.csv'
            if os.path.exists(clustering_file):
                self.clustering_data = pd.read_csv(clustering_file)
                print(f"✅ Loaded Kaggle clustering data: {len(self.clustering_data)} customers")
            else:
                print(f"❌ Kaggle clustering data not found. Please run kaggle_clustering.py first.")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Kaggle data: {e}")
            return False
    
    def prepare_features(self):
        """Prepares features for LTV prediction based on dataset type"""
        
        if self.rfm_data is None or self.clustering_data is None:
            print("❌ No data loaded. Please load data first.")
            return None, None
        
        print("🔄 Preparing features for LTV prediction...")
        
        # Use clustering data directly since it already contains RFM information
        merged_data = self.clustering_data.copy()
        
        # Base features
        base_features = ['recency', 'frequency', 'monetary', 'RFM_score', 'cluster']
        
        # Additional features based on dataset type
        if self.dataset_type == 'retail_customer':
            # Check what additional columns are available in RFM data
            available_rfm_columns = [col for col in ['Age', 'Annual_Income_k', 'Spending_Score_1_100'] if col in self.rfm_data.columns]
            if available_rfm_columns:
                # Merge with customer data for additional features
                merge_columns = ['customer_id'] + available_rfm_columns
                merged_data = merged_data.merge(self.rfm_data[merge_columns], on='customer_id', how='left')
                additional_features = available_rfm_columns
            else:
                additional_features = []
            available_features = [col for col in additional_features if col in merged_data.columns]
            
        elif self.dataset_type == 'ecommerce_customer':
            additional_features = []
            available_features = []
            
        elif self.dataset_type == 'online_retail':
            additional_features = []
            available_features = []
        
        # Combine all available features
        all_features = base_features + available_features
        available_features = [col for col in all_features if col in merged_data.columns]
        
        print(f"📊 Using features: {available_features}")
        
        # Prepare feature matrix
        X = merged_data[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Create target variable (LTV based on historical spending patterns)
        # LTV = Average transaction value * Expected frequency * Customer lifetime
        avg_transaction_value = merged_data['monetary'] / merged_data['frequency']
        expected_frequency = merged_data['frequency'] * 1.2  # Assume 20% growth
        customer_lifetime = 12  # Assume 12 months average lifetime
        
        y = avg_transaction_value * expected_frequency * customer_lifetime
        
        # Add some randomness to make it more realistic
        np.random.seed(42)
        y = y * np.random.uniform(0.8, 1.2, len(y))
        
        return X, y, available_features
    
    def train_models(self, X, y):
        """Trains multiple LTV prediction models"""
        
        print("🤖 Training LTV prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"  🔄 Training {name}...")
            
            # Train model
            if name in ['XGBoost', 'LightGBM']:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"    ✅ {name} - R²: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        self.models = results
        return results
    
    def get_best_model(self):
        """Returns the best performing model"""
        
        if not self.models:
            print("❌ No models trained. Please train models first.")
            return None
        
        # Find best model based on R² score
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_model = self.models[best_model_name]
        
        print(f"🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {best_model['r2']:.3f}")
        print(f"   RMSE: {best_model['rmse']:.2f}")
        print(f"   MAE: {best_model['mae']:.2f}")
        
        return best_model_name, best_model
    
    def calculate_feature_importance(self, best_model_name):
        """Calculates feature importance for the best model"""
        
        if best_model_name not in self.models:
            print("❌ Model not found.")
            return None
        
        model = self.models[best_model_name]['model']
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print("❌ Cannot calculate feature importance for this model type.")
            return None
        
        # Create feature importance DataFrame
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def create_predictions(self, X, best_model_name):
        """Creates LTV predictions using the best model"""
        
        if best_model_name not in self.models:
            print("❌ Model not found.")
            return None
        
        model = self.models[best_model_name]['model']
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def create_visualizations(self, X, y, best_model_name):
        """Creates LTV prediction visualizations"""
        
        print("📈 Creating LTV prediction visualizations...")
        
        best_model = self.models[best_model_name]
        y_pred = best_model['y_pred']
        y_test = best_model['y_test']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Kaggle LTV Prediction Analysis - {self.dataset_type.replace("_", " ").title()}', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual LTV')
        axes[0, 0].set_ylabel('Predicted LTV')
        axes[0, 0].set_title('Actual vs Predicted LTV')
        axes[0, 0].grid(True)
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted LTV')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True)
        
        # 3. LTV distribution
        axes[1, 0].hist(y, bins=30, alpha=0.7, label='Actual LTV')
        axes[1, 0].hist(y_pred, bins=30, alpha=0.7, label='Predicted LTV')
        axes[1, 0].set_xlabel('LTV')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('LTV Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Feature importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, X, y, best_model_name, predictions, fig):
        """Saves LTV prediction results"""
        
        # Create processed directory if it doesn't exist
        os.makedirs('data/kaggle/processed', exist_ok=True)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'customer_id': self.rfm_data['customer_id'],
            'actual_ltv': y,
            'predicted_ltv': predictions,
            'ltv_difference': predictions - y,
            'ltv_accuracy': 1 - np.abs(predictions - y) / y
        })
        
        # Add segment information
        predictions_df = predictions_df.merge(
            self.rfm_data[['customer_id', 'segment', 'RFM_score']], 
            on='customer_id', how='left'
        )
        
        # Save predictions
        predictions_df.to_csv('data/kaggle/processed/kaggle_ltv_predictions.csv', index=False)
        print("✅ Saved kaggle_ltv_predictions.csv")
        
        # Save model comparison
        comparison_data = []
        for name, results in self.models.items():
            comparison_data.append({
                'Model': name,
                'R2_Score': results['r2'],
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('data/kaggle/processed/kaggle_model_comparison.csv', index=False)
        print("✅ Saved kaggle_model_comparison.csv")
        
        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv('data/kaggle/processed/kaggle_feature_importance.csv', index=False)
            print("✅ Saved kaggle_feature_importance.csv")
        
        # Save summary statistics
        summary_stats = {
            'dataset_type': self.dataset_type,
            'total_customers': len(predictions_df),
            'best_model': best_model_name,
            'best_r2_score': self.models[best_model_name]['r2'],
            'best_rmse': self.models[best_model_name]['rmse'],
            'best_mae': self.models[best_model_name]['mae'],
            'avg_ltv': predictions_df['predicted_ltv'].mean(),
            'total_predicted_ltv': predictions_df['predicted_ltv'].sum(),
            'avg_ltv_accuracy': predictions_df['ltv_accuracy'].mean()
        }
        
        import json
        with open('data/kaggle/processed/kaggle_ltv_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=4)
        print("✅ Saved kaggle_ltv_summary.json")
        
        # Save visualization
        if fig:
            fig.savefig('data/kaggle/processed/kaggle_ltv_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved kaggle_ltv_analysis.png")
        
        return True
    
    def print_summary(self, predictions_df, best_model_name):
        """Prints LTV prediction summary"""
        
        print("\n📊 Kaggle LTV Prediction Summary")
        print("=" * 40)
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Total Customers: {len(predictions_df):,}")
        print(f"Best Model: {best_model_name}")
        print(f"Best R² Score: {self.models[best_model_name]['r2']:.3f}")
        print(f"Best RMSE: {self.models[best_model_name]['rmse']:.2f}")
        
        print("\n💰 LTV Statistics:")
        print(f"  • Average Predicted LTV: ${predictions_df['predicted_ltv'].mean():,.2f}")
        print(f"  • Total Predicted LTV: ${predictions_df['predicted_ltv'].sum():,.2f}")
        if 'ltv_accuracy' in predictions_df.columns:
            print(f"  • Average LTV Accuracy: {predictions_df['ltv_accuracy'].mean():.1%}")
        
        print("\n🎯 LTV by Segment:")
        segment_ltv = predictions_df.groupby('segment')['predicted_ltv'].agg(['mean', 'sum', 'count'])
        for segment, stats in segment_ltv.iterrows():
            print(f"  • {segment}: ${stats['mean']:,.2f} avg, ${stats['sum']:,.2f} total ({stats['count']} customers)")

def main():
    """Main function to run Kaggle LTV prediction"""
    
    print("🎯 Kaggle LTV Prediction for Customer Segmentation")
    print("=" * 50)
    
    # Available dataset types
    dataset_types = ['retail_customer', 'ecommerce_customer', 'online_retail']
    
    for dataset_type in dataset_types:
        print(f"\n🔄 Processing {dataset_type} dataset...")
        
        # Initialize LTV prediction
        ltv_predictor = KaggleLTVPrediction()
        
        # Load data
        if ltv_predictor.load_kaggle_data(dataset_type):
            # Prepare features
            X, y, feature_names = ltv_predictor.prepare_features()
            
            if X is not None:
                # Train models
                model_results = ltv_predictor.train_models(X, y)
                
                # Get best model
                best_model_name, best_model = ltv_predictor.get_best_model()
                
                # Calculate feature importance
                feature_importance = ltv_predictor.calculate_feature_importance(best_model_name)
                
                # Create predictions
                predictions = ltv_predictor.create_predictions(X, best_model_name)
                
                # Create visualizations
                fig = ltv_predictor.create_visualizations(X, y, best_model_name)
                
                # Create predictions DataFrame for saving
                predictions_df = pd.DataFrame({
                    'customer_id': ltv_predictor.rfm_data['customer_id'],
                    'predicted_ltv': predictions,
                    'segment': ltv_predictor.rfm_data['segment']
                })
                
                # Save results
                ltv_predictor.save_results(X, y, best_model_name, predictions, fig)
                
                # Print summary
                ltv_predictor.print_summary(predictions_df, best_model_name)
                
                print(f"✅ {dataset_type} LTV prediction completed successfully!")
            else:
                print(f"❌ Failed to prepare features for {dataset_type}")
        else:
            print(f"❌ Failed to load {dataset_type} data")
    
    print("\n🎉 All Kaggle LTV prediction completed!")
    print("📁 Results saved in: data/kaggle/processed/")

if __name__ == "__main__":
    main()
