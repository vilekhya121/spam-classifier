#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import email
from bs4 import BeautifulSoup
import joblib
from sklearn.preprocessing import StandardScaler
import torch 
from torch import nn


# In[2]:


scaler = joblib.load('scaler.joblib')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# In[3]:


# Function to preprocess email content
def preprocess_email(email_content):
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(email_content, "html.parser")
    text_content = soup.get_text()
    return text_content


# In[4]:


class EmailClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmailClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# In[5]:


input_size=1000
hidden_size = 64
output_size=2


# In[6]:


model = EmailClassifier(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('email_classifier_model.pth'))
model.eval()


# In[7]:


st.title("Email/SMS Spam Classifier")
input_content=st.text_area("Enter the Emai Content")

if st.button('Predict'):
    email_content = preprocess_email(input_content)
    email_features = vectorizer.transform([email_content]).toarray()
    new_email_features = scaler.transform(email_features)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(new_email_features, dtype=torch.float32)
     
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
    if predicted_class == 1:
        st.header("Spam")
    else:
        st.header("Ham")


# In[ ]:




