import streamlit as st
import random

# Define topics
topics = {
    "Beginner": {
        "Machine Learning": [
            "What is Machine Learning?",
            "Supervised vs Unsupervised Learning",
            "Overfitting & Underfitting",
            "Bias-Variance Tradeoff",
            "Linear Regression",
            "Logistic Regression",
            "Decision Trees",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes Classifier",
            "Train-Test Split",
            "Evaluation Metrics (Accuracy, Precision, Recall, F1)",
            "Confusion Matrix",
            "Feature Engineering Basics",
            "Handling Missing Values",
            "One-Hot Encoding vs Label Encoding"
        ],
        "Statistics": [
            "Types of Data: Qualitative vs Quantitative",
            "Levels of Measurement: Nominal, Ordinal, Interval, Ratio",
            "Measures of Central Tendency (Mean, Median, Mode)",
            "Measures of Dispersion (Variance, Std Dev, Range, IQR)",
            "Basics of Probability",
            "Sampling Techniques",
            "Histograms, Boxplots, Bar Charts",
            "Introduction to Hypothesis Testing",
            "P-values and Confidence Intervals",
            "Basic Distributions (Uniform, Normal, Binomial)",
            "Descriptive vs Inferential Statistics"
        ],
        "AI": [
            "What is Artificial Intelligence?",
            "Difference between AI, ML, and DL",
            "Rule-Based Systems",
            "Turing Test",
            "Applications of AI",
            "Search Algorithms: DFS, BFS",
            "Introduction to NLP",
            "Chatbots and Voice Assistants Basics",
            "Basic Logic & Reasoning in AI"
        ]
    },
    "Intermediate": {
        "Machine Learning": [
            "Gradient Descent",
            "Cost Function in Regression",
            "Clustering: K-Means, Hierarchical",
            "Random Forests",
            "Support Vector Machines (SVM)",
            "Principal Component Analysis (PCA)",
            "Feature Selection Techniques",
            "Cross-Validation (K-Fold, Stratified)",
            "Grid Search & Random Search for Hyperparameter Tuning",
            "Ensemble Methods: Bagging vs Boosting",
            "XGBoost & LightGBM",
            "ROC, AUC, Precision-Recall Curves",
            "Outlier Detection",
            "Normalization vs Standardization"
        ],
        "Statistics": [
            "Central Limit Theorem",
            "t-test (one-sample, two-sample)",
            "z-test",
            "Chi-Square Test",
            "ANOVA (One-way & Two-way)",
            "Regression Analysis (Simple & Multiple)",
            "Correlation vs Causation",
            "Skewness & Kurtosis",
            "Probability Distributions (Poisson, Geometric)",
            "Confidence Level vs Significance Level",
            "Type I and Type II Errors",
            "Effect Size",
            "Bootstrapping"
        ],
        "AI": [
            "Neural Networks (ANN Basics)",
            "CNN for Image Recognition",
            "RNN for Time Series and Text",
            "Transformers Basics",
            "Text Preprocessing (Tokenization, Lemmatization)",
            "Embeddings (Word2Vec, GloVe)",
            "Reinforcement Learning Basics",
            "Q-Learning Concept",
            "Recommendation Systems (Collaborative Filtering)",
            "Sentiment Analysis",
            "Ethics in AI"
        ]
    },
    "Advance": {
        "Machine Learning": [
            "SHAP and LIME for Model Explainability",
            "Model Interpretability Techniques",
            "Handling Imbalanced Data (SMOTE, ROC Analysis)",
            "Transfer Learning",
            "Time Series Forecasting (ARIMA, SARIMA, Prophet)",
            "AutoML Platforms",
            "Model Deployment (Flask, FastAPI, Streamlit)",
            "ML Pipelines and Workflow Automation",
            "Federated Learning",
            "Concept Drift and Data Drift",
            "Model Monitoring and A/B Testing",
            "Online Learning Algorithms"
        ],
        "Statistics": [
            "Bayesian Inference",
            "Maximum Likelihood Estimation (MLE)",
            "Markov Chains",
            "Hidden Markov Models",
            "Multivariate Regression",
            "Survival Analysis",
            "Monte Carlo Simulations",
            "Expectation-Maximization Algorithm",
            "Gibbs Sampling",
            "MCMC (Markov Chain Monte Carlo)",
            "Statistical Power and Sample Size Estimation",
            "Advanced Probability Theory"
        ],
        "AI": [
            "Transformers Deep Dive (BERT, GPT)",
            "Generative Adversarial Networks (GANs)",
            "Attention Mechanism in NLP",
            "Deep Reinforcement Learning",
            "AI in Real-Time Systems (Edge AI, IoT)",
            "Explainable AI (XAI)",
            "Neuro-symbolic AI",
            "Multimodal AI",
            "AI for Healthcare and Finance",
            "Ethical AI and Bias Detection",
            "Meta Learning",
            "AI Governance and Regulation"
        ]
    }
}

# App title
st.title("🎯 Random Topic Generator")

# Dropdowns for level and category
selected_level = st.selectbox("Select Difficulty Level", list(topics.keys()))
selected_category = st.selectbox("Select Topic Category", list(topics[selected_level].keys()))

# Key to store progress
session_key = f"{selected_level}_{selected_category}_shown"

# Initialize session state for shown topics
if session_key not in st.session_state:
    st.session_state[session_key] = []

# Button click logic
# Button to generate a random topic
if st.button("Generate Topic"):
    available = list(set(topics[selected_level][selected_category]) - set(st.session_state[session_key]))
    
    if available:
        topic = random.choice(available)
        st.session_state[session_key].append(topic)
        st.success(f"📌 Random Topic: **{topic}**")
    else:
        st.warning("✅ All topics in this category are already shown!")

# Reset button (always visible)
st.markdown("---")
if st.button("🔄 Reset Topics"):
    st.session_state[session_key] = []
    st.info("Topics have been reset. You can now generate them again.")

