import pandas as pd
import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Function to load model and tokenizer dynamically based on the help type
def load_model_and_tokenizer(help_type):
    if help_type == 'English Grammar':
        model_path = 'GPTM'  # Model for English grammar
        tokenizer_path = 'GPTOK'  # Tokenizer for English grammar
    else:
        model_path = 'GPTMC'  # Model for coding
        tokenizer_path = 'GPTOKC'  # Tokenizer for coding
    
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Ensure that the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token as GPT-2 does not use a pad token by default
    
    return model, tokenizer

# Streamlit interface
def main():
    st.title("EduBot: English Grammar and Basic Coding Help")
    st.write("Welcome! What do you need help with today?")

    # User choice for help type
    help_choice = st.radio("Choose a topic for assistance:", ('Coding', 'English Grammar'))

    # Load the appropriate model and tokenizer based on user selection
    model, tokenizer = load_model_and_tokenizer(help_choice)

    if help_choice == 'English Grammar':
        st.write("How can I help you with English grammar? Enter your query below.")
        input_text = st.text_area("Enter your prompt here:", height=150, placeholder="Enter your English grammar query here...")
    else:
        st.write("Please enter your coding related query below.")
        input_text = st.text_area("Enter your prompt here:", height=150, placeholder="Enter your coding query here...")
    
    max_length = st.slider("Select the maximum length of generated text", 50, 1000, 500)
    
    if st.button("Generate Help"):
        with st.spinner('Generating...'):
            generated_text = generate_text(input_text, max_length, model, tokenizer)
            st.write(generated_text)

# Function to generate text using the specified model and tokenizer
def generate_text(input_text, max_length, model, tokenizer):
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

if __name__ == "__main__":
    main()
