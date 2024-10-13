import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set OpenAI API Key (replace with your actual key)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Generate Simulated Claims Data
def generate_claims_data(n=100):
    claims_data = {
        'ClaimID': [f"C{str(i).zfill(4)}" for i in range(1, n+1)],  # Simple ClaimID like C0001
        'MemberID': np.random.randint(1000, 1100, n),
        'ClaimAmount': np.round(np.random.uniform(100, 5000, n), 2),
        'DateOfClaim': pd.date_range(start='2023-01-01', periods=n).strftime('%Y-%m-%d'),
        'ClaimStatus': np.random.choice(['Approved', 'Pending', 'Rejected'], n)
    }
    return pd.DataFrame(claims_data)

# Generate Simulated Enrollment Data
def generate_enrollment_data(n=100):
    enrollment_data = {
        'MemberID': np.random.randint(1000, 1100, n),
        'MemberName': [f"Member_{str(i).zfill(4)}" for i in range(1, n+1)],  # Simple MemberName like Member_0001
        'DateOfEnrollment': pd.date_range(start='2015-01-01', periods=n).strftime('%Y-%m-%d'),
        'PlanType': np.random.choice(['HMO', 'PPO', 'EPO', 'POS'], n)
    }
    return pd.DataFrame(enrollment_data)

# Load the simulated claims and enrollment datasets
@st.cache_data
def load_data():
    claims_df = generate_claims_data()
    enrollment_df = generate_enrollment_data()
    return claims_df, enrollment_df

claims_df, enrollment_df = load_data()

# Function to process the user's question using GPT-4 (with the new chat completion API)
def ask_question(question, claims_df, enrollment_df):
    # Combine the two datasets into a prompt
    combined_data = (
        f"Claims Data:\n{claims_df.head(10).to_string(index=False)}\n\n"
        f"Enrollment Data:\n{enrollment_df.head(10).to_string(index=False)}\n"
    )
    prompt = f"{combined_data}\n\nUser Question: {question}\nAnswer based on the data:"

    # Call the OpenAI Chat API using the correct method
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Using GPT-4 model
        messages=[
            {"role": "system", "content": "You are an expert in healthcare claims and enrollment data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=150
    )

    # Extract and return the response text
    return response.choices[0].message['content'].strip()

# Streamlit UI
st.title("AI Agent for Simulated Claims and Enrollment Data")

st.write("This app allows you to ask questions about claims and enrollment data and get answers.")

# Display the datasets
if st.checkbox("Show Claims Data"):
    st.dataframe(claims_df)

if st.checkbox("Show Enrollment Data"):
    st.dataframe(enrollment_df)

# Input box for the user to ask questions
question = st.text_input("Ask any question about the claims or enrollment data:")

if question:
    answer = ask_question(question, claims_df, enrollment_df)
    st.write("Answer:")
    st.write(answer)
