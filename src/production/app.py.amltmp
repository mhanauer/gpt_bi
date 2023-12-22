import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure the OpenAI client with your API key
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def ask_gpt(prompt):
    """
    This function takes a prompt and returns the response from OpenAI's GPT model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Replace with your desired GPT model
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def process_csv_file(uploaded_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Convert the DataFrame to a string or extract specific information for the prompt
    prompt = df.to_string()

    return prompt

# Streamlit app
def main():
    st.title('Chat with GPT and Upload CSV')

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    user_input = st.text_input("Enter your prompt:", "")

    if uploaded_file:
        # Process the CSV file to create a prompt
        prompt = process_csv_file(uploaded_file)
        gpt_response = ask_gpt(prompt)
        st.text_area("GPT's response from CSV:", gpt_response, height=200)
    
    elif user_input:
        # Direct user prompt
        gpt_response = ask_gpt(user_input)
        st.text_area("GPT's direct response:", gpt_response, height=200)

if __name__ == '__main__':
    main()
