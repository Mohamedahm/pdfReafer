import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Needed for some TensorFlow Hub models

def load_model():
    """Load a pre-trained T5 model from TensorFlow Hub."""
    model = hub.load("https://tfhub.dev/google/t5-large-ssm-nq/1")
    return model

def process_text(model, question, context):
    """Generate text using a pre-trained model."""
    # Prepare the input text
    input_text = f"question: {question} context: {context}"
    input_tensor = tf.constant([input_text])  # Model expects a batch of inputs

    # Perform inference
    outputs = model.signatures['serving_default'](input_tensor)
    return outputs['outputs'].numpy()[0].decode('utf-8')

import streamlit as st

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    # Load the model
    model = load_model()

    # File uploader widget
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)  # Reuse your existing function
        if not text:
            st.error("The PDF appears to be empty or text could not be extracted.")
            return

        # Input field for user's question
        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            answer = process_text(model, user_question, text)
            st.write("Question:", user_question)
            st.write("Predicted Answer:", answer)

if __name__ == '__main__':
    main()
