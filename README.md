# Facial Demographic Analysis - AI-Powered

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

## 🧠 Multi-Task Deep Learning System for Facial Demographic Analysis

Developed an end-to-end multi-task deep learning system leveraging Convolutional Neural Networks (CNN) to simultaneously predict age, gender, and ethnicity from facial images.

### 🎯 Key Features
- **Multi-Output CNN Architecture**: 8.98M parameters with 4 convolutional blocks
- **High Accuracy**: 90.7% gender accuracy, 80.8% ethnicity accuracy, 5.79-year age MAE
- **Real-Time Analysis**: Live camera support + image upload functionality
- **Production-Ready**: Deployed on Railway.app with Flask backend

### 📊 Model Performance
- **Dataset**: UTKFace (23,705 facial images)
- **Training Split**: 51.2% train, 12.8% validation, 20% test
- **Gender Classification**: 90.7% accuracy
- **Ethnicity Classification**: 80.8% accuracy
- **Age Prediction**: 5.79 years Mean Absolute Error

### 🛠️ Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Backend**: Flask, Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Railway.app

### 🚀 Quick Start

#### Local Development
```bash
# Clone repository
git clone https://github.com/MuhammadTaimoorHaider/FYP-FeminaFortress.git
cd FYP-FeminaFortress

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Visit `http://localhost:5000`

#### Deploy to Railway
1. Fork this repository
2. Click the "Deploy on Railway" button above
3. Connect your GitHub repository
4. Railway will automatically deploy your app
5. Your app will be live in 2-3 minutes!

### 📁 Project Structure
```
FYP-FeminaFortress/
├── app.py                      # Flask application
├── best_model.h5              # Trained CNN model
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway configuration
├── runtime.txt               # Python version
├── templates/
│   └── index.html            # Web interface
├── static/
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── camera.js        # Camera functionality
└── README.md
```

### 🎓 Developer Information
- **Developer**: Muhammad Taimoor Haider
- **Supervisor**: Dr. Rao Muhammad Adeel Nawab
- **LinkedIn**: [Muhammad Taimoor](https://www.linkedin.com/in/muhammad-taimoor-43a769121)
- **Email**: dev.taimoor148@gmail.com
- **Date**: November 2025

### 📝 License
This project is developed as part of academic research.

### 🙏 Acknowledgments
- UTKFace Dataset
- TensorFlow Team
- Railway.app for hosting
