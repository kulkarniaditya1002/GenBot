import pandas as pd
import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Function to load model and tokenizer
def load_model_and_tokenizer():
    model_path = 'GPTM'
    tokenizer_path = 'GPTOK'
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Ensure that the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token as GPT-2 does not use a pad token by default
    
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to generate text
def generate_text(input_text, max_length=500):
    """
    Generate text using the fine-tuned GPT-2 model.

    Args:
    input_text (str): Input text to start the generation.
    max_length (int): Maximum length of the generated text.

    Returns:
    str: The generated text.
    """
    if pd.isna(input_text):
        input_text = " "  # Handle NaN values explicitly if found in input

    # Encode the inputs with attention mask
    encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length, return_attention_mask=True)

    # Adjust generation parameters to reduce repetition and encourage diversity
    outputs = model.generate(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_k=20,
        top_p=0.95,
        no_repeat_ngram_size=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit interface
def main():
    st.title("GPT-2 Text Generator")
    st.write("This app uses a fine-tuned GPT-2 model to generate text. Enter your prompt below:")

    # Text box for user input
    user_input = st.text_area("Enter your prompt here", height=150)
    max_length = st.slider("Select the maximum length of generated text", 50, 1000, 500)

    if st.button("Generate Text"):
        with st.spinner('Generating...'):
            generated_text = generate_text(user_input, max_length)
            st.write(generated_text)

    #st.write("Type 'quit' to exit (or just close this window).")

if __name__ == "__main__":
    main()
