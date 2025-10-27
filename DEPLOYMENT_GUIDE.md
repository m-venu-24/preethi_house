# House Price Predictor - Deployment Guide

## Option 1: Deploy to Heroku (Free)

### Steps:
1. **Create Heroku account** at https://heroku.com
2. **Install Heroku CLI** from https://devcenter.heroku.com/articles/heroku-cli
3. **Create these files:**

### Procfile (create this file):
```
web: python app.py
```

### runtime.txt (create this file):
```
python-3.9.18
```

### requirements.txt (already exists):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
flask
gunicorn
```

### Deploy commands:
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-house-price-predictor

# Deploy
git init
git add .
git commit -m "Initial commit"
git push heroku main

# Open your app
heroku open
```

## Option 2: Deploy to PythonAnywhere (Free)

### Steps:
1. **Sign up** at https://pythonanywhere.com
2. **Upload your files** via web interface
3. **Create web app** in dashboard
4. **Configure WSGI** file
5. **Reload** your web app

## Option 3: Deploy to Railway (Free)

### Steps:
1. **Sign up** at https://railway.app
2. **Connect GitHub** repository
3. **Deploy** automatically

## Option 4: Share as Standalone App

### Create executable file:
```bash
pip install pyinstaller
pyinstaller --onefile app.py
```

This creates a .exe file you can share!

## Quick Local Sharing

### For immediate sharing on your network:
1. **Find your IP address:**
   ```bash
   ipconfig
   ```
2. **Share this URL:** `http://YOUR_IP:5000`
3. **Make sure Windows Firewall allows** port 5000

### Example:
- If your IP is 192.168.1.100
- Share: `http://192.168.1.100:5000`
