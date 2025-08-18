"""
Simple Customer Segmentation & LTV Analysis - LTV Prediction Script

This script builds machine learning models to predict customer lifetime value (LTV).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all necessary data for LTV prediction"""
    print("Loading data for LTV prediction...")
    
    try:
        customers_df = pd.read_csv('data/raw/customers_large.csv')
        transactions_df = pd.read_csv('data/raw/transactions_large.csv')
        rfm_scores_df = pd.read_csv('data/processed/rfm_scores.csv')
        customer_clusters_df = pd.read_csv('data/processed/customer_clusters.csv')
        print(f"Loaded {len(customers_df)} customers with RFM scores and cluster assignments")
        return customers_df, transactions_df, rfm_scores_df, customer_clusters_df
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please run simple_large_data_generation.py, rfm_analysis.py, and simple_clustering.py first. {e}")
        return None, None, None, None

def calculate_ltv_target(transactions_df, customers_df, prediction_horizon_days=365):
    """Calculate LTV target for each customer"""
    
    print("Calculating LTV targets...")
    
    # Convert dates
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    # Calculate historical LTV (last 12 months)
    analysis_date = transactions_df['transaction_date'].max()
    start_date = analysis_date - pd.Timedelta(days=365)
    
    # Filter transactions for last 12 months
    recent_transactions = transactions_df[
        transactions_df['transaction_date'] >= start_date
    ]
    
    # Calculate LTV for each customer
    ltv_data = []
    
    for customer_id in customers_df['customer_id'].unique():
        customer_transactions = recent_transactions[
            recent_transactions['customer_id'] == customer_id
        ]
        
        if len(customer_transactions) > 0:
            # Calculate average monthly value
            total_value = customer_transactions['total_amount'].sum()
            months_active = 12  # Assuming 12 months of data
            
            avg_monthly_value = total_value / months_active
            
            # Project LTV for prediction horizon
            months_to_predict = prediction_horizon_days / 30
            projected_ltv = avg_monthly_value * months_to_predict
            
            # Add some variability based on customer behavior
            frequency = len(customer_transactions)
            
            # Adjust LTV based on customer characteristics
            if frequency >= 10:  # High frequency customers
                ltv_multiplier = 1.2
            elif frequency >= 5:  # Medium frequency customers
                ltv_multiplier = 1.0
            else:  # Low frequency customers
                ltv_multiplier = 0.8
            
            final_ltv = projected_ltv * ltv_multiplier
            
        else:
            # No recent transactions
            final_ltv = 0
        
        ltv_data.append({
            'customer_id': customer_id,
            'ltv_target': max(0, final_ltv)
        })
    
    ltv_df = pd.DataFrame(ltv_data)
    
    print(f"Calculated LTV targets for {len(ltv_df)} customers")
    print(f"Average LTV: ${ltv_df['ltv_target'].mean():.2f}")
    print(f"LTV Range: ${ltv_df['ltv_target'].min():.2f} - ${ltv_df['ltv_target'].max():.2f}")
    
    return ltv_df

def prepare_ltv_features(customers_df, transactions_df, rfm_scores_df, customer_clusters_df, ltv_df):
    """Prepare features for LTV prediction"""
    
    print("Preparing features for LTV prediction...")
    
    # Start with RFM scores
    features_df = rfm_scores_df.copy()
    
    # Add cluster information
    features_df = features_df.merge(customer_clusters_df[['customer_id', 'cluster']], 
                                   on='customer_id', how='left')
    
    # Add customer demographics
    features_df = features_df.merge(customers_df, on='customer_id', how='left')
    
    # Add transaction-based features
    transaction_features = transactions_df.groupby('customer_id').agg({
        'product_category': 'nunique',
        'payment_method': 'nunique',
        'total_amount': ['mean', 'std', 'sum'],
        'quantity': ['mean', 'sum']
    }).reset_index()
    
    # Flatten column names
    transaction_features.columns = ['customer_id', 'product_categories', 'payment_methods', 
                                  'avg_amount', 'amount_std', 'total_spent', 'avg_quantity', 'total_quantity']
    
    features_df = features_df.merge(transaction_features, on='customer_id', how='left')
    
    # Add time-based features
    features_df['customer_lifetime_days'] = (
        pd.to_datetime(features_df['last_purchase_date']) - 
        pd.to_datetime(features_df['first_purchase_date'])
    ).dt.days
    
    features_df['avg_days_between_purchases'] = features_df['customer_lifetime_days'] / features_df['frequency']
    features_df['purchase_velocity'] = features_df['frequency'] / (features_df['customer_lifetime_days'] / 30)
    
    # Add derived features
    features_df['total_spent_per_day'] = features_df['monetary'] / features_df['customer_lifetime_days']
    features_df['frequency_per_month'] = features_df['frequency'] / 12
    features_df['monetary_per_purchase'] = features_df['monetary'] / features_df['frequency']
    
    # Add LTV target
    features_df = features_df.merge(ltv_df, on='customer_id', how='left')
    
    # Select numerical features for modeling
    feature_columns = [
        'recency', 'frequency', 'monetary',
        'R_score', 'F_score', 'M_score',
        'RFM_score', 'RFM_weighted_score',
        'avg_order_value', 'customer_lifetime_days',
        'avg_days_between_purchases', 'purchase_velocity',
        'total_spent_per_day', 'frequency_per_month', 'monetary_per_purchase',
        'product_categories', 'payment_methods',
        'avg_amount', 'amount_std', 'total_spent',
        'avg_quantity', 'total_quantity', 'cluster'
    ]
    
    # Filter available features and ensure they are numerical
    available_features = []
    for col in feature_columns:
        if col in features_df.columns:
            if features_df[col].dtype in ['int64', 'float64']:
                available_features.append(col)
    
    print(f"Selected {len(available_features)} features for LTV prediction")
    
    return features_df, available_features

def train_ltv_models(features_df, feature_columns, target_column='ltv_target'):
    """Train multiple models for LTV prediction"""
    
    print("Training LTV prediction models...")
    
    # Prepare data
    X = features_df[feature_columns].fillna(0)
    y = features_df[target_column]
    
    # Clean infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Remove any remaining infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        if name in ['Linear Regression', 'Ridge Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        if name in ['Linear Regression', 'Ridge Regression']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'scaler': scaler if name in ['Linear Regression', 'Ridge Regression'] else None,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def select_best_model(results):
    """Select the best performing model"""
    
    print("\n=== MODEL COMPARISON ===")
    
    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'R² Score': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'CV R²': result['cv_mean'],
            'CV Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Select best model based on R² score
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"R² Score: {best_model['r2']:.4f}")
    print(f"RMSE: ${best_model['rmse']:.2f}")
    
    return best_model_name, best_model, comparison_df

def analyze_feature_importance(best_model, feature_columns, model_name):
    """Analyze feature importance for the best model"""
    
    print(f"\nAnalyzing feature importance for {model_name}...")
    
    if hasattr(best_model['model'], 'feature_importances_'):
        # Tree-based models
        importances = best_model['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('data/processed/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif hasattr(best_model['model'], 'coef_'):
        # Linear models
        coefficients = best_model['model'].coef_
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("Top 10 Most Important Features (by coefficient magnitude):")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    return feature_importance_df

def create_ltv_predictions(features_df, best_model, feature_columns, model_name):
    """Create LTV predictions for all customers"""
    
    print(f"Creating LTV predictions using {model_name}...")
    
    # Prepare features
    X = features_df[feature_columns].fillna(0)
    
    # Clean infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Remove any remaining infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Make predictions
    if best_model['scaler'] is not None:
        X_scaled = best_model['scaler'].transform(X)
        predictions = best_model['model'].predict(X_scaled)
    else:
        predictions = best_model['model'].predict(X)
    
    # Create predictions DataFrame
    predictions_df = features_df[['customer_id']].copy()
    predictions_df['actual_ltv'] = features_df['ltv_target']
    predictions_df['predicted_ltv'] = predictions
    predictions_df['prediction_error'] = predictions_df['actual_ltv'] - predictions_df['predicted_ltv']
    predictions_df['prediction_error_pct'] = (predictions_df['prediction_error'] / predictions_df['actual_ltv'] * 100).fillna(0)
    
    # Add segment information
    if 'segment' in features_df.columns:
        predictions_df['segment'] = features_df['segment']
    if 'cluster' in features_df.columns:
        predictions_df['cluster'] = features_df['cluster']
    
    print(f"LTV Predictions Summary:")
    print(f"  Average Predicted LTV: ${predictions_df['predicted_ltv'].mean():.2f}")
    print(f"  Average Actual LTV: ${predictions_df['actual_ltv'].mean():.2f}")
    print(f"  Average Prediction Error: ${predictions_df['prediction_error'].abs().mean():.2f}")
    print(f"  Average Prediction Error %: {predictions_df['prediction_error_pct'].abs().mean():.1f}%")
    
    return predictions_df

def visualize_ltv_results(results, predictions_df, best_model_name):
    """Create visualizations for LTV prediction results"""
    
    print("Creating LTV prediction visualizations...")
    
    best_model = results[best_model_name]
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted LTV
    axes[0, 0].scatter(best_model['y_test'], best_model['y_pred'], alpha=0.6)
    axes[0, 0].plot([best_model['y_test'].min(), best_model['y_test'].max()], 
                    [best_model['y_test'].min(), best_model['y_test'].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual LTV ($)')
    axes[0, 0].set_ylabel('Predicted LTV ($)')
    axes[0, 0].set_title(f'Actual vs Predicted LTV - {best_model_name}')
    
    # 2. Prediction Error Distribution
    axes[0, 1].hist(predictions_df['prediction_error'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Prediction Error ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('LTV Prediction Error Distribution')
    
    # 3. LTV Distribution by Cluster
    if 'cluster' in predictions_df.columns:
        cluster_ltv = predictions_df.groupby('cluster')['predicted_ltv'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(cluster_ltv)), cluster_ltv.values, color='lightgreen')
        axes[1, 0].set_xlabel('Customer Cluster')
        axes[1, 0].set_ylabel('Average Predicted LTV ($)')
        axes[1, 0].set_title('Average Predicted LTV by Cluster')
        axes[1, 0].set_xticks(range(len(cluster_ltv)))
        axes[1, 0].set_xticklabels([f'Cluster {i}' for i in cluster_ltv.index])
    
    # 4. LTV Distribution
    axes[1, 1].hist(predictions_df['predicted_ltv'], bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].set_xlabel('Predicted LTV ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('LTV Distribution')
    
    plt.tight_layout()
    plt.savefig('data/processed/ltv_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("LTV prediction visualizations saved")

def save_ltv_results(predictions_df, comparison_df, feature_importance_df, best_model_name):
    """Save LTV prediction results"""
    
    # Create processed data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save LTV predictions
    predictions_df.to_csv('data/processed/ltv_predictions.csv', index=False)
    print("Saved ltv_predictions.csv")
    
    # Save model comparison
    comparison_df.to_csv('data/processed/model_comparison.csv', index=False)
    print("Saved model_comparison.csv")
    
    # Save feature importance
    if feature_importance_df is not None:
        feature_importance_df.to_csv('data/processed/feature_importance.csv', index=False)
        print("Saved feature_importance.csv")
    
    # Create summary report
    summary_stats = {
        'best_model': best_model_name,
        'total_customers': len(predictions_df),
        'avg_predicted_ltv': predictions_df['predicted_ltv'].mean(),
        'avg_actual_ltv': predictions_df['actual_ltv'].mean(),
        'avg_prediction_error': predictions_df['prediction_error'].abs().mean(),
        'avg_prediction_error_pct': predictions_df['prediction_error_pct'].abs().mean(),
        'total_predicted_value': predictions_df['predicted_ltv'].sum(),
        'total_actual_value': predictions_df['actual_ltv'].sum()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('data/processed/ltv_summary.csv', index=False)
    print("Saved ltv_summary.csv")

def main():
    """Main function to perform LTV prediction"""
    
    print("Simple Customer Segmentation & LTV Analysis - LTV Prediction")
    print("=" * 60)
    
    # Load data
    customers_df, transactions_df, rfm_scores_df, customer_clusters_df = load_data()
    if customers_df is None or transactions_df is None or rfm_scores_df is None:
        return
    
    # Calculate LTV targets
    ltv_df = calculate_ltv_target(transactions_df, customers_df)
    
    # Prepare features
    features_df, feature_columns = prepare_ltv_features(
        customers_df, transactions_df, rfm_scores_df, customer_clusters_df, ltv_df
    )
    
    # Train models
    results = train_ltv_models(features_df, feature_columns)
    
    # Select best model
    best_model_name, best_model, comparison_df = select_best_model(results)
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(best_model, feature_columns, best_model_name)
    
    # Create predictions
    predictions_df = create_ltv_predictions(features_df, best_model, feature_columns, best_model_name)
    
    # Create visualizations
    visualize_ltv_results(results, predictions_df, best_model_name)
    
    # Save results
    save_ltv_results(predictions_df, comparison_df, feature_importance_df, best_model_name)
    
    print("\n" + "=" * 60)
    print("LTV prediction completed successfully!")
    print("Results saved in ../data/processed/ directory")
    print("Next steps:")
    print("1. Create Power BI dashboard")
    print("2. Generate marketing recommendations")

if __name__ == "__main__":
    main()
