import streamlit as st
import pandas as pd
import numpy as np
import os
import openai
import plotly.express as px
import plotly.graph_objects as go
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def ask_gpt(prompt):
    try:
        response = client.chat_completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def generate_python_code_prompt(df, question):
    num_rows, num_columns = df.shape
    df_head = df.head().to_string()
    prompt = f"""
    Here is a pandas dataframe (df) with {num_rows} rows and {num_columns} columns:
    {df_head}

    Please provide the Python code to answer the following question about the dataframe:
    {question}
    """
    return prompt

def extract_python_code(output):
    if output is None:
        return ""
    match = re.search(r'```python\n(.*?)(```|$)', output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        cleaned_code = '\n'.join([line for line in code.split('\n') if 'read_csv' not in line])
        return cleaned_code
    else:
        st.error("No valid Python code block found in the GPT response.")
        return ""

def execute_code(code, df):
    try:
        exec_locals = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np}
        exec(code, {}, exec_locals)  # Execute the code

        fig = exec_locals.get('fig', None)
        if fig:
            st.plotly_chart(fig)  # Display the Plotly figure

    except Exception as e:
        st.error(f"An error occurred while executing the code: {e}")

def main():
    st.title("MedeGPT")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, 'mede.png')

    st.write("Upload your dataset and enter your question about the data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("DataFrame Preview (just the first few rows):")
        st.write(df.head())

        question = st.text_input("Enter your question about the DataFrame:")
        
        if question and st.button('Generate Plot'):
            formatted_prompt = generate_python_code_prompt(df, question)
            output = ask_gpt(formatted_prompt)
            extracted_code = extract_python_code(output)
            
            if extracted_code:
                st.code(extracted_code, language='python')
                execute_code(extracted_code, df)

if __name__ == "__main__":
    main()
