# 🚀 Deployment Guide

This guide will help you deploy the Customer Segmentation & LTV Analysis project to various platforms.

## 📋 Prerequisites

Before deployment, ensure you have:
- Python 3.8+ installed
- Git installed
- Required Python packages (see `python/requirements.txt`)
- PostgreSQL (for SQL analysis features)

## 🐳 Local Deployment

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd CustomerSegmentation

# Install dependencies
pip install -r python/requirements.txt

# Generate data (if not already present)
python python/simple_large_data_generation.py
python python/kaggle_data_processor.py

# Run analysis scripts
python python/rfm_analysis.py
python python/simple_clustering.py
python python/simple_ltv_prediction.py
python python/kaggle_rfm_analysis.py
python python/kaggle_clustering.py
python python/kaggle_ltv_prediction.py
```

### 2. Start the Dashboard
```bash
# Run the unified dashboard
streamlit run python/unified_dashboard.py

# Access at http://localhost:8501
```

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set the main file path: `python/unified_dashboard.py`
   - Deploy

3. **Configuration**
   - Add `streamlit run python/unified_dashboard.py` to your deployment command
   - Set Python version to 3.8 or higher

### Option 2: Heroku

1. **Create Procfile**
   ```bash
   echo "web: streamlit run python/unified_dashboard.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

2. **Create runtime.txt**
   ```bash
   echo "python-3.9.16" > runtime.txt
   ```

3. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "python/unified_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t customer-segmentation .
   docker run -p 8501:8501 customer-segmentation
   ```

## 🔧 Environment Variables

Create a `.env` file for configuration:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_segmentation
DB_USER=your_username
DB_PASSWORD=your_password

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 📊 Database Setup

### PostgreSQL Setup
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb customer_segmentation
sudo -u postgres createdb customer_segmentation_kaggle

# Import data
psql -d customer_segmentation -f sql/setup_database.sql
psql -d customer_segmentation_kaggle -f sql/setup_kaggle_database.sql
```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit sensitive data
2. **Database Security**: Use strong passwords and limit access
3. **HTTPS**: Enable SSL for production deployments
4. **Authentication**: Consider adding user authentication for sensitive data

## 📈 Performance Optimization

1. **Data Caching**: Use Streamlit's caching for expensive computations
2. **Database Indexing**: Ensure proper indexes on frequently queried columns
3. **Memory Management**: Monitor memory usage for large datasets
4. **CDN**: Use CDN for static assets in production

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r python/requirements.txt --force-reinstall
   ```

3. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Restart PostgreSQL
   sudo systemctl restart postgresql
   ```

4. **Memory Issues**
   ```bash
   # Increase memory limit for Streamlit
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

## 📞 Support

For deployment issues:
1. Check the [Streamlit documentation](https://docs.streamlit.io/)
2. Review the [GitHub issues](https://github.com/your-repo/issues)
3. Contact the project maintainers

## 🔄 Continuous Deployment

### GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      env:
        STREAMLIT_SHARING_TOKEN: ${{ secrets.STREAMLIT_SHARING_TOKEN }}
      run: |
        pip install streamlit
        streamlit deploy python/unified_dashboard.py
```

---

**Note**: Always test your deployment in a staging environment before going to production.
