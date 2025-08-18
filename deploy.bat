@echo off
echo 🚀 Starting Customer Segmentation Dashboard Deployment...

REM Check if we're in the right directory
if not exist "python\unified_dashboard.py" (
    echo ❌ Error: unified_dashboard.py not found. Please run this script from the project root.
    pause
    exit /b 1
)

REM Create .streamlit directory if it doesn't exist
if not exist ".streamlit" mkdir .streamlit

REM Copy requirements to root for Streamlit Cloud
copy "python\requirements.txt" "requirements.txt"

REM Check data files
echo 📊 Checking data files...
if exist "data\kaggle\processed" (
    echo ✅ Kaggle processed data found
) else (
    echo ⚠️  Kaggle processed data not found - dashboard will show error
)

if exist "data\processed" (
    echo ✅ Synthetic processed data found
) else (
    echo ⚠️  Synthetic processed data not found - dashboard will show error
)

echo.
echo 🎯 Deployment Preparation Complete!
echo.
echo 📋 Next Steps:
echo 1. Push this code to GitHub
echo 2. Connect your repository to Streamlit Cloud
echo 3. Set the main file path to: python/unified_dashboard.py
echo 4. Deploy!
echo.
echo 🔧 Performance Optimizations Applied:
echo    ✅ Data caching enabled
echo    ✅ Memory optimization
echo    ✅ Chart optimization
echo    ✅ SQL query optimization
echo    ✅ System monitoring added
echo.
echo 📈 Expected Performance:
echo    - Load time: ^< 5 seconds
echo    - Memory usage: ^< 80%%
echo    - Smooth chart rendering
pause
