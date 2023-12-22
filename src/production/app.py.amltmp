import streamlit as st
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY')

def ask_gpt(prompt):
    """
    This function takes a prompt and returns the response from OpenAI's GPT-4 model.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.0",  # replace with your preferred GPT-4 model version
            messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
def main():
    st.title('Chat with Mede GenBI')

    user_input = st.text_input("Enter your prompt:", "")

    if user_input:
        gpt_response = ask_gpt(user_input)
        st.text_area("Mede GenBI's response:", gpt_response, height=200)

if __name__ == '__main__':
    main()
