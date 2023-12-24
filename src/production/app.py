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
load_dotenv()

# Set up OpenAI client
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

def generate_python_code_prompt(df, question):
    START_CODE_TAG = "```"
    END_CODE_TAG = "```"
    num_rows, num_columns = df.shape
    df_head = df.head().to_string()
    prompt = f"""
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

When asked about the data, your response should include the python code describing the
dataframe `df`. If the question requires data visualization, use Plotly for plotting. Do not include sample data. Using the provided dataframe, df, return python code and prefix
the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG}
exactly to answer the following question:
{question}

When the prompt includes words like plot or graph use only Plotly for any plotting requirements.
"""
    return prompt

def extract_python_code(output):
    match = re.search(r'```python\n(.*?)(```|$)', output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        cleaned_code = '\n'.join([line for line in code.split('\n') if 'read_csv' not in line])
        return cleaned_code
    else:
        raise ValueError("No valid Python code found in the output")

def execute_code(code, df, question, max_retries=5):
    error_message = None
    retries = 0
    
    while retries <= max_retries:
        try:
            exec_locals = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np}
            exec(code, {}, exec_locals)  # Execute the code

            # Check if the figure has been created
            fig = exec_locals.get('fig', None)
            if fig:
                st.plotly_chart(fig)  # Display the Plotly figure
                return None, None  # Return None as there's no result variable in plot cases

            # Check for DataFrame or similar output
            result = exec_locals.get('result', None)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)  # Display the DataFrame
                return None, None

        except Exception as e:
            error_message = str(e)
            result = None
            df_head = df.head().to_string()
            new_formatted_prompt = f"With this pandas dataframe (df): {df_head}\nAfter asking this question\n'{question}' \nI ran this code '{code}' \nAnd received this error message \n'{error_message}'. \nPlease provide new correct Python code."
            output = ask_gpt(new_formatted_prompt)
            code = extract_python_code(output)  # Update code for the next iteration
            retries += 1  # Increment the retry counter
            
    return None, f"Failed to fix the code after {max_retries} retries. Last error: {error_message}"

def main():
    st.title("MedeGPT")
    st.image("mede.png", width=100)  # Adjust the width as needed
    st.write("Upload your own dataset or use the default demo dataset.  Use the key word Plot at the beginning of any prompt.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the uploaded file into DataFrame
        df = pd.read_csv(uploaded_file)
    else:
        # Use default DataFrame
        df = pd.DataFrame({
            'Var1': [1, 2, 3, 4, 5, 6],
            'Var2': [4, 5, 6, 7, 8, 9],
            'Gender': ['M', 'F', 'M', 'M', 'F', "M"],
            'State': ['IN', "NC", 'IN', 'NC', 'IN', 'NC'],
            'Race': ['W', 'B', 'W', 'B', 'W', "B"]
        })

    st.write("DataFrame Preview (just the first few rows):")
    st.write(df.head())

    question = st.text_input("Enter your question about the DataFrame:")
    
    if question:
        formatted_prompt = generate_python_code_prompt(df, question)
        output = ask_gpt(formatted_prompt)
        
        try:
            extracted_code = extract_python_code(output)
            st.write("Generated Python Code:")
            st.code(extracted_code, language='python')
            
            result, error_message = execute_code(extracted_code, df, question)
            if error_message:
                st.write(f"Error: {error_message}")
            elif result is not None:
                st.write("Result:")
                st.write(result)
        
        except ValueError as e:
            st.write(f"Error: {e}")

if __name__ == "__main__":
    main()
