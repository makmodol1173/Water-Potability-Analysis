# 💧 Water Potability Analysis System

This project provides a comprehensive system for analyzing and predicting water potability using machine learning. It features data loading, preprocessing, statistical analysis, various machine learning models, interactive visualizations, and a user-friendly Streamlit interface for real-time predictions.

## ✨ Features

*   **Data Loading & Preprocessing**: Handles data loading from a remote URL, missing value imputation, and duplicate removal.
*   **Statistical Analysis**: Provides detailed descriptive statistics, correlation analysis, and data quality assessment.
*   **Machine Learning Models**: Implements and compares multiple classification models (Random Forest, Logistic Regression, SVM, Gradient Boosting) to predict water potability.
*   **Interactive Visualizations**: Generates various plots (pie charts, bar charts, histograms, scatter plots, heatmaps) to explore data distributions and relationships.
*   **Real-time Prediction**: Allows users to input water parameters and get instant predictions on water potability with confidence scores.
*   **Prediction History**: Keeps a log of past predictions and provides summary statistics.
*   **Modular & OOP Design**: Built with a robust Object-Oriented Programming (OOP) architecture for maintainability and extensibility.

## 🚀 Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository**:
    \`\`\`bash
    git clone https://github.com/your-username/water-potability-analysis.git
    cd water-potability-analysis
    \`\`\`

2.  **Create a virtual environment (recommended)**:
    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    \`\`\`

3.  **Install dependencies**:
    \`\`\`bash
    pip install pandas numpy scikit-learn streamlit plotly
    \`\`\`
    (Note: A `requirements.txt` file can be generated from these dependencies for easier management.)

## 🏃 Usage

To run the Streamlit application, execute the following command from the project's root directory:

\`\`\`bash
streamlit run app.py
\`\`\`

This will open the application in your web browser, where you can navigate through different sections for data overview, analysis, visualizations, model training, and predictions.

## 📊 Data Source

The dataset used in this project is sourced from a Vercel Blob URL:
\`\`\`
https://hebbkx1anhila5yf.public.blob.vercel-storage.com/water_potability_preprocessed-aP2VS7drsoWULn1qmITGHQDpRcDEhe.csv
