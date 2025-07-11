# Movie Sentiment Analysis Project

A comprehensive sentiment analysis application that classifies movie reviews as positive or negative using both traditional machine learning and deep learning approaches.

## 🎯 Project Overview

This project implements multiple sentiment analysis models to classify movie reviews:

- **Traditional ML Models**: Logistic Regression, Random Forest, XGBoost, Passive Aggressive
- **Deep Learning Models**: CNN and Transformer architectures
- **Interactive Web App**: Streamlit-based interface for real-time predictions

## 🚀 Features

- **Multi-Model Comparison**: Compare performance across different algorithms
- **Optimized Training**: Memory-efficient training with system resource adaptation
- **Real-time Predictions**: Web interface for instant sentiment classification
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Model Persistence**: Save/load trained models for deployment

## 📋 Requirements

### System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB+ recommended for deep learning models)
- 2GB free disk space

### Python Dependencies

Install all required packages using:

```bash
pip install -r Requirements.txt
```

The main dependencies include:

- `tensorflow>=2.12.0`
- `pandas>=1.5.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`
- `streamlit`
- `joblib`
- `nltk`
- `contractions`
- `xgboost`

## 🛠️ Setup Instructions

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd "cs20 project V2"

# Or download and extract the ZIP file
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r Requirements.txt
```

### 3. Download NLTK Data

Run Python and execute:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Prepare Dataset

Ensure you have the required dataset files:

- `train.csv` - Training data
- `test.csv` - Test data
- `IMDB Dataset.csv` - IMDB dataset

Or use the provided `final_combined_dataset_clean.csv` if already processed.

## 🏃‍♂️ Quick Start

### Option 1: Use Pre-trained Models (Recommended)

If you have pre-trained model files:

```bash
# Run the web application
streamlit run app.py
```

### Option 2: Train Models from Scratch

#### Step 1: Data Preparation

```bash
# Run dataset concatenation (if needed)
jupyter notebook "dataset concatenation.ipynb"
```

#### Step 2: Train Traditional ML Models

```bash
# Train traditional models
jupyter notebook "model_creation.ipynb"
```

#### Step 3: Train Deep Learning Models

```bash
# Train CNN and Transformer models
jupyter notebook "deep_learning_model_creation.ipynb"
```

#### Step 4: Run the Application

```bash
streamlit run app.py
```

## 📊 Model Performance

Based on the evaluation metrics:

- **Best Traditional Model**: Varies by dataset (typically XGBoost or Logistic Regression)
- **Deep Learning Models**: CNN and Transformer architectures with 95%+ accuracy
- **Evaluation Metrics**: Accuracy, AUC Score, Precision, Recall, F1-Score

## 🎮 Using the Web Application

1. **Start the Application**:

   ```bash
   streamlit run app.py
   ```

2. **Access the Interface**:

   - Open your browser to `http://localhost:8501`

3. **Make Predictions**:

   - Select a model from the dropdown
   - Enter a movie review in the text area
   - Click "Predict" to get sentiment classification

4. **Interpret Results**:
   - Green background = Positive sentiment
   - Red background = Negative sentiment

## 📁 Project Structure

```text
cs20 project V2/
├── README.md                              # This file
├── Requirements.txt                       # Python dependencies
├── app.py                                # Streamlit web application
├── config.py                             # Configuration settings
├── model_evaluation_metrics.json         # Model performance metrics
├──
├── Notebooks/
│   ├── dataset concatenation.ipynb       # Data preparation
│   ├── model_creation.ipynb             # Traditional ML models
│   └── deep_learning_model_creation.ipynb # Deep learning models
├──
├── Models/ (generated after training)
│   ├── tfidf_vectorizer.pkl             # Text vectorizer
│   ├── logistic_regression_model.pkl    # Logistic Regression
│   ├── random_forest_model.pkl          # Random Forest
│   ├── xgboost_model.pkl                # XGBoost
│   ├── passive_aggressive_model.pkl     # Passive Aggressive
│   ├── sentiment_cnn_model.keras        # CNN model
│   ├── sentiment_transformer_model.keras # Transformer model
│   └── best_sentiment_model.keras       # Best performing model
├──
└── Data/
    ├── train.csv                         # Training data
    ├── test.csv                          # Test data
    ├── IMDB Dataset.csv                  # IMDB dataset
    └── final_combined_dataset_clean.csv  # Combined dataset
```

## 🔧 Troubleshooting

### Common Issues

1. **ImportError: No module named 'tensorflow'**

   ```bash
   pip install tensorflow>=2.12.0
   ```

2. **NLTK Data Not Found**

   ```python
   import nltk
   nltk.download('all')
   ```

3. **Memory Issues with Deep Learning**

   - Reduce batch size in the notebooks
   - Use the conservative configuration for systems with <4GB RAM

4. **Model Files Not Found**

   - Ensure you've run the training notebooks first
   - Check that model files are in the correct directory

5. **Streamlit App Won't Start**

   ```bash
   # Check if streamlit is installed
   pip install streamlit

   # Try running with full path
   python -m streamlit run app.py
   ```

### Performance Optimization

- **For Low-Memory Systems**: Use traditional ML models only
- **For High-Performance**: Use deep learning models with larger batch sizes
- **For Production**: Use the best-performing model based on your evaluation

## 🔬 Model Training Details

### Traditional ML Models

- **Preprocessing**: TF-IDF vectorization, stopword removal, lemmatization
- **Models**: Logistic Regression, Random Forest, XGBoost, Passive Aggressive
- **Evaluation**: Cross-validation, confusion matrices, ROC curves

### Deep Learning Models

- **Architectures**: CNN with multiple conv layers, Transformer with multi-head attention
- **Optimization**: Adam optimizer, learning rate scheduling, early stopping
- **Memory Management**: Adaptive batch sizes based on system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect the original dataset licenses.

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review the notebook outputs for error messages
3. Ensure all dependencies are correctly installed

## 🎓 Educational Notes

This project demonstrates:

- Text preprocessing techniques
- Feature extraction methods
- Multiple machine learning approaches
- Deep learning for NLP
- Model evaluation and comparison
- Web application deployment

Perfect for learning sentiment analysis, NLP, and machine learning model deployment!
