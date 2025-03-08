import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Define the BERTModel class
class BERTModel(nn.Module):
    def __init__(self, n_classes):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# Function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    output_dir = './model_save/'
    try:
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        model = BERTModel(n_classes=2)
        model.load_state_dict(torch.load(f"{output_dir}/model_state_dict.pt", map_location=torch.device('cpu')))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Function to make predictions
def predict(text, model, tokenizer, max_len=160):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)
    
    return prediction.item(), probabilities[0].cpu().numpy()

# Function to display model performance metrics
def display_metrics():
    # Placeholder for metrics (replace with your actual metrics)
    metrics = {
        'Accuracy': 0.92,
        'Precision (Class 0)': 0.90,
        'Recall (Class 0)': 0.93,
        'F1-Score (Class 0)': 0.91,
        'Precision (Class 1)': 0.94,
        'Recall (Class 1)': 0.91,
        'F1-Score (Class 1)': 0.92,
    }
    st.subheader("Model Performance Metrics")
    st.write("Below are the performance metrics of the BERT model on the validation set:")
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    st.dataframe(metrics_df.style.format({"Value": "{:.2%}"}))

    # Placeholder for confusion matrix (replace with your actual confusion matrix)
    confusion_matrix = np.array([[90, 10], [8, 92]])
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# Function to generate a downloadable report
def generate_report(text, prediction, probabilities):
    report = f"Text Classification Report\n\n"
    report += f"Input Text: {text}\n"
    report += f"Predicted Class: {'Class 0' if prediction == 0 else 'Class 1'}\n"
    report += f"Class Probabilities:\n  Class 0: {probabilities[0]:.2%}\n  Class 1: {probabilities[1]:.2%}\n"
    
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
    return buffer

# Streamlit app
def main():
    # Set page configuration for a professional look
    st.set_page_config(page_title="Text Classification with BERT", page_icon="📊", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and introduction
    st.markdown('<div class="main-title">Text Classification with BERT</div>', unsafe_allow_html=True)
    st.write("""
        Welcome to this interactive text classification app powered by a fine-tuned BERT model. 
        This app demonstrates advanced natural language processing capabilities, leveraging state-of-the-art transformer models.
        Enter your text below to classify it into one of two categories (e.g., positive/negative sentiment, spam/not spam, etc.).
    """)

    # Sidebar for navigation and additional information
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Model Insights", "About"])

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please ensure the model files are available.")
        return

    # Page: Prediction
    if page == "Prediction":
        st.subheader("Make a Prediction")
        st.write("Enter your text in the box below and click 'Predict' to see the classification result.")
        
        # Text input
        user_input = st.text_area("Enter text here:", "", height=200)

        if st.button("Predict"):
            if user_input:
                # Make prediction
                prediction, probabilities = predict(user_input, model, tokenizer)
                
                # Display result
                st.write("### Prediction Result:")
                st.success(f"**Predicted Class:** {'Class 0' if prediction == 0 else 'Class 1'}")
                st.write(f"**Class Probabilities:**")
                st.write(f"- Class 0: {probabilities[0]:.2%}")
                st.write(f"- Class 1: {probabilities[1]:.2%}")

                # Offer downloadable report
                report_buffer = generate_report(user_input, prediction, probabilities)
                st.download_button(
                    label="Download Prediction Report",
                    data=report_buffer,
                    file_name="prediction_report.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Please enter some text to classify.")

    # Page: Model Insights
    elif page == "Model Insights":
        st.subheader("Model Insights")
        st.write("""
            This section provides insights into the model's performance and architecture.
            Understanding these metrics is crucial for evaluating the effectiveness of the model.
        """)
        display_metrics()

    # Page: About
    elif page == "About":
        st.subheader("About This Project")
        st.write("""
            This project demonstrates the power of transformer-based models like BERT for text classification tasks.
            The app is built using Streamlit, a Python framework for creating interactive web applications, and leverages
            PyTorch and Hugging Face's Transformers library for model implementation.

            **Key Features:**
            - Fine-tuned BERT model for binary text classification.
            - Interactive user interface for real-time predictions.
            - Detailed model performance metrics and visualizations.
            - Downloadable prediction reports.

            **Technologies Used:**
            - Python, PyTorch, Hugging Face Transformers, Streamlit, Pandas, NumPy, Matplotlib, Seaborn.

            **Developer:** Sai Ruthvik  
            This project is part of my portfolio to showcase my expertise in machine learning, natural language processing, 
            and software engineering. For more details, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sai-ruthvik) 
            or check out my [GitHub](https://github.com/hawkh).
        """)

if __name__ == "__main__":
    main()
