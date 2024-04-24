import os
import streamlit as st
from PyPDF2 import PdfReader
from transformers import MT5ForConditionalGeneration, T5Tokenizer

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    
    Args:
    pdf_file (BytesIO): A file-like object representing the PDF.

    Returns:
    str: The extracted text concatenated from all pages.
    """
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text() or ''
        # Normalize whitespace and remove line breaks within text
        extracted_text = ' '.join(extracted_text.replace('\n', ' ').split())
        text += extracted_text + ' '
    return text.strip()

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    # Initialize model and tokenizer
    model_name = "google/mt5-small"  # Consider using a larger model for production
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # File uploader widget
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            st.error("The PDF appears to be empty or text could not be extracted.")
            return

        # Input field for user's question
        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            # Prepare the input text for the model
            input_text = f"translate Arabic to English: {user_question} context: {text}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Generate the answer from the model
            outputs = model.generate(input_ids)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Display the question and its answer
            st.write("Question:", user_question)
            st.write("Predicted Answer:", answer)

if __name__ == '__main__':
    main()
