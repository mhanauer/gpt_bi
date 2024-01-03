import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import openai
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Set up OpenAI client
client = openai.OpenAI(api_key=api_key)

# Existing functions (ask_gpt, generate_python_code_prompt, extract_python_code, execute_code) go here

def analyze_data_with_gpt(df, question):
    """
    This function takes a question about a DataFrame and uses GPT to generate and execute Python code for the analysis.
    """
    # Generate a Python code prompt for GPT based on the question
    formatted_prompt = generate_python_code_prompt(df, question)

    try:
        # Get Python code from GPT
        output = ask_gpt(formatted_prompt)
        generated_code = extract_python_code(output)

        # Execute the generated code
        exec_locals = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np}
        exec(generated_code, {}, exec_locals)  # Execute the code

        # Retrieve the result
        result = exec_locals.get('result', None)
        if result is None:
            raise ValueError("The executed code did not produce a result variable.")

        return result

    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.title("MedeGPT")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, 'mede.png')
    st.image(logo_path, width=300)  # Adjust the path and width as needed
    st.write("Upload your dataset and enter your question about the data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("DataFrame Preview (just the first few rows):")
        st.write(df.head())

        question = st.text_input("Enter your question about the DataFrame:")
        
        if question:
            formatted_prompt = generate_python_code_prompt(df, question)
            output = ask_gpt(formatted_prompt)
            
            try:
                extracted_code = extract_python_code(output)
                st.write("Generated Python Code (Inspect for syntax errors):")
                st.code(extracted_code, language='python')

                if st.button('Execute Code'):
                    # Use the analyze_data_with_gpt function to execute the code and get results
                    results = analyze_data_with_gpt(df, question)

                    if isinstance(results, pd.DataFrame) or isinstance(results, pd.Series):
                        st.write("Analysis Results:")
                        st.write(results)
                    else:
                        st.write("Results:")
                        st.write(results)
            
            except ValueError as e:
                st.write(f"Error: {e}")
    else:
        st.write("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
