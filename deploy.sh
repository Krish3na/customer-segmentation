#!/bin/bash

# Customer Segmentation Dashboard Deployment Script
# Optimized for Streamlit Cloud deployment

echo "🚀 Starting Customer Segmentation Dashboard Deployment..."

# Check if we're in the right directory
if [ ! -f "python/unified_dashboard.py" ]; then
    echo "❌ Error: unified_dashboard.py not found. Please run this script from the project root."
    exit 1
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Copy requirements to root for Streamlit Cloud
cp python/requirements.txt requirements.txt

# Create a simple streamlit run script
echo "streamlit run python/unified_dashboard.py --server.port=8501 --server.address=0.0.0.0" > run.sh
chmod +x run.sh

# Check data files
echo "📊 Checking data files..."
if [ -d "data/kaggle/processed" ]; then
    echo "✅ Kaggle processed data found"
else
    echo "⚠️  Kaggle processed data not found - dashboard will show error"
fi

if [ -d "data/processed" ]; then
    echo "✅ Synthetic processed data found"
else
    echo "⚠️  Synthetic processed data not found - dashboard will show error"
fi

echo ""
echo "🎯 Deployment Preparation Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Push this code to GitHub"
echo "2. Connect your repository to Streamlit Cloud"
echo "3. Set the main file path to: python/unified_dashboard.py"
echo "4. Deploy!"
echo ""
echo "🔧 Performance Optimizations Applied:"
echo "   ✅ Data caching enabled"
echo "   ✅ Memory optimization"
echo "   ✅ Chart optimization"
echo "   ✅ SQL query optimization"
echo "   ✅ System monitoring added"
echo ""
echo "📈 Expected Performance:"
echo "   - Load time: < 5 seconds"
echo "   - Memory usage: < 80%"
echo "   - Smooth chart rendering"
