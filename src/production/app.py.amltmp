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

def summarize_results(data):
    """
    This function takes the data (either a DataFrame or plot details) and returns a written summary.
    """
    plot_description = f"This is a {type(data).__name__} with axes: {list(data.layout.xaxis.keys()) + list(data.layout.yaxis.keys())}."
    prompt = f"Summarize the following plot details:\n{plot_description}"

    summary = ask_gpt(prompt)  # Use the GPT function to get the summary
    return summary

def execute_code(code, df, question, max_retries=5):
    error_message = None
    retries = 0
    
    while retries <= max_retries:
        try:
            exec_locals = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np}
            exec(code, {}, exec_locals)  # Execute the code

            fig = exec_locals.get('fig', None)
            if fig:
                st.plotly_chart(fig)  # Display the Plotly figure
                summary = summarize_results(fig)
                st.write("Summary of the Plot:")
                st.write(summary)
                return None, None

            st.write("No plot was generated.")
            return None, None

        except SyntaxError as e:
            st.write(f"Syntax error in the code: {e}")
            return None, f"Syntax error: {e}"
        except Exception as e:
            error_message = str(e)
            retries += 1  # Increment the retry counter
            if retries <= max_retries:
                st.write(f"Attempting to fix the code. Retry {retries}/{max_retries}.")
                df_head = df.head().to_string()
                new_formatted_prompt = f"With this pandas dataframe (df): {df_head}\nAfter asking this question\n'{question}' \nI ran this code '{code}' \nAnd received this error message \n'{error_message}'. \nPlease provide new correct Python code."
                output = ask_gpt(new_formatted_prompt)
                code = extract_python_code(output)
            else:
                st.write(f"Failed to fix the code after {max_retries} retries. Last error: {error_message}")
                return None, error_message
            
    return None, None

def main():
    st.title("Data Analysis with GPT")
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
    else:
        st.write("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
