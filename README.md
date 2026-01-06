# Roman Urdu Sentiment Analysis System

A professional Flask web application for analyzing sentiments in Roman Urdu text using multiple machine learning models.

## 🌟 Features

- **Multiple ML Models**: Choose from 5 different trained models (SVM, Logistic Regression, Naive Bayes, Decision Tree, Random Forest)
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Professional UI**: Modern, responsive, and attractive user interface
- **Comprehensive Documentation**: Complete system documentation with visualizations
- **Model Comparison**: Detailed accuracy comparisons and performance metrics
- **Interactive Charts**: Visual representation of model performances

## 📊 Sentiment Categories

- **Positive**: Favorable opinions, satisfaction, happiness
- **Negative**: Dissatisfaction, criticism, unfavorable opinions
- **Neutral**: Objective statements without strong emotions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "C:\Users\Muhammad Hamza Nawaz\Desktop\Roman Urdu Sentiment"
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Verify that all model files are present:**
- `saved_ml_models/best_model.joblib`
- `saved_ml_models/tfidf_vectorizer.joblib`
- `saved_ml_models/label_encoder.joblib`
- `saved_ml_models/metadata.json`
- All files in `all_models/` directory
- All files in `saved_ml_models/plots/` directory
- All files in `saved_ml_models/reports/` directory

### Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your web browser and navigate to:**
```
http://localhost:5000
```

3. **The application will be running with three main pages:**
   - Home: `http://localhost:5000/`
   - Prediction: `http://localhost:5000/prediction`
   - Documentation: `http://localhost:5000/documentation`

## 📁 Project Structure

```
Roman Urdu Sentiment/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── prediction.html            # Prediction interface
│   └── documentation.html         # Documentation page
│
├── static/                        # Static files
│   ├── css/
│   │   └── style.css             # Main stylesheet
│   └── js/
│       └── script.js             # JavaScript functionality
│
├── saved_ml_models/               # Trained models and metadata
│   ├── best_model.joblib         # Best performing model (SVM)
│   ├── tfidf_vectorizer.joblib   # TF-IDF vectorizer
│   ├── label_encoder.joblib      # Label encoder
│   ├── metadata.json             # Model metadata and accuracies
│   ├── plots/                    # Confusion matrices and charts
│   │   ├── model_accuracy_comparison.png
│   │   ├── SVM_confusion_matrix.png
│   │   ├── Logistic Regression_confusion_matrix.png
│   │   ├── Naive Bayes_confusion_matrix.png
│   │   ├── Decision Tree_confusion_matrix.png
│   │   ├── KNN_confusion_matrix.png
│   │   └── Random Forest_confusion_matrix.png
│   └── reports/                  # Classification reports
│       ├── SVM_classification_report.csv
│       ├── Logistic Regression_classification_report.csv
│       └── ... (other model reports)
│
└── all_models/                    # Individual model files
    ├── svm.joblib
    ├── logistic_regression.joblib
    ├── naive_bayes.joblib
    ├── decision_tree.joblib
    └── random_forest.joblib
```

## 🎯 Usage

### Making Predictions

1. Navigate to the **Prediction** page
2. Select a model from the dropdown (SVM is recommended as the best model)
3. Enter Roman Urdu text in the text area
4. Click **"Analyze Sentiment"** button
5. View the prediction results with:
   - Sentiment classification (Positive/Negative/Neutral)
   - Confidence score (if available)
   - Cleaned text used for analysis
   - Model used for prediction

### Example Texts

Try these example texts:

**Positive:**
```
yeh mobile bohat acha hai, mujhe bohat pasand aaya
```

**Neutral:**
```
mausam theek hai, na zyada garmi na sardi
```

**Negative:**
```
yeh service bohat buri hai, bilkul kharab experience
```

## 📖 Documentation

Visit the **Documentation** page to explore:

- **Introduction**: Overview of the system
- **Problem Statement**: Challenges in Roman Urdu sentiment analysis
- **Solution Approach**: Technical methodology
- **Dataset Information**: Details about training data
- **Data Preprocessing**: Text cleaning pipeline
- **Models Trained**: All 5 ML models with descriptions
- **Accuracy Comparison**: Interactive chart showing model performances
- **Confusion Matrices**: Visual representation of model predictions
- **Classification Reports**: Detailed metrics (precision, recall, F1-score)
- **Best Model**: Why SVM performs best
- **Conclusions**: Key achievements and future improvements

## 🏆 Model Performance

| Model | Accuracy |
|-------|----------|
| SVM (Best) | 74.40% |
| Logistic Regression | 73.82% |
| Naive Bayes | 72.58% |
| Random Forest | 71.81% |
| Decision Tree | 64.98% |

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Text Preprocessing**
   - Lowercase conversion
   - URL removal
   - Special character elimination
   - Whitespace normalization

2. **Feature Extraction**
   - TF-IDF Vectorization
   - N-gram range: 1-3
   - Max features: 40,000
   - Min document frequency: 3
   - Max document frequency: 0.9

3. **Model Training**
   - GridSearchCV for hyperparameter tuning
   - Class weight balancing for imbalanced data
   - 75-25 train-test split with stratification
   - 3-fold cross-validation

4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Classification Report

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js
- **Icons**: Font Awesome

## 📝 API Endpoints

### GET Routes
- `/` - Home page
- `/prediction` - Prediction interface
- `/documentation` - Documentation page
- `/get_model_accuracy` - Returns JSON with model accuracies

### POST Routes
- `/predict` - Make sentiment predictions
  - **Request Body**: 
    ```json
    {
      "text": "your roman urdu text here",
      "model": "SVM"
    }
    ```
  - **Response**:
    ```json
    {
      "sentiment": "Positive",
      "confidence": 85.5,
      "model_used": "SVM",
      "cleaned_text": "your roman urdu text here"
    }
    ```

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern Gradient Backgrounds**: Attractive color schemes
- **Smooth Animations**: Fade-in effects and transitions
- **Interactive Elements**: Hover effects and dynamic feedback
- **Professional Typography**: Clear and readable text
- **Color-coded Sentiments**: Visual indicators for each sentiment type

## 🚧 Future Enhancements

- Deep learning models (LSTM, BERT)
- Aspect-based sentiment analysis
- Multi-language support
- User authentication and history
- API rate limiting
- Model retraining interface
- Batch prediction support
- Export results to CSV/PDF

## 🐛 Troubleshooting

**Issue**: Flask app not starting
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Model files not found
- **Solution**: Verify that `saved_ml_models/` and `all_models/` directories contain all required files

**Issue**: Predictions not working
- **Solution**: Check that TF-IDF vectorizer and label encoder are loaded correctly

**Issue**: Images not displaying in documentation
- **Solution**: Ensure all PNG files are present in `saved_ml_models/plots/` directory

## 📄 License

This project is created for educational and research purposes.

## 👤 Author

Muhammad Hamza Nawaz

## 🙏 Acknowledgments

- Roman Urdu Tagged Dataset (Kaggle)
- Scikit-learn documentation and community
- Flask framework
- Font Awesome for icons
- Chart.js for visualizations

## 📞 Support

For issues or questions, please check the documentation page within the application for comprehensive information about the system.
