The error "'ChatCompletionMessage' object is not subscriptable" suggests that the way we're trying to access the response from the OpenAI API isn't compatible with the returned object structure. Let's correct the code to properly handle the response:

```python
import streamlit as st
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
        # Corrected way to access the response content
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
def main():
    st.title('Chat with GPT')

    user_input = st.text_input("Enter your prompt:", "")

    if user_input:
        gpt_response = ask_gpt(user_input)
        st.text_area("GPT's response:", gpt_response, height=200)

if __name__ == '__main__':
    main()
```

This updated code correctly accesses the response content from the OpenAI API. The key change is in how the response's content is retrieved: `response.choices[0].message.content`. This aligns with the structure of the response object in the newer version of the OpenAI API.