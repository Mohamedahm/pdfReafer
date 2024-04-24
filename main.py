import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for TensorFlow Hub models that use text
import streamlit as st

def load_model():
    """Load a pre-trained T5 model from TensorFlow Hub."""
    try:
        model = hub.load("https://tfhub.dev/google/t5-large-ssm-nq/1")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def process_text(model, question, context):
    """Generate text using a pre-trained model."""
    try:
        input_text = f"question: {question} context: {context}"
        input_tensor = tf.constant([input_text])  # Model expects a batch of inputs
        outputs = model.signatures['serving_default'](input_tensor)
        return outputs['outputs'].numpy()[0].decode('utf-8')
    except Exception as e:
        st.error(f"Failed to process text: {e}")
        return "Error in processing text"

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    model = load_model()
    if not model:
        st.error("Model could not be loaded.")
        return

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)  # Ensure this function is correctly implemented
        if not text:
            st.error("The PDF appears to be empty or text could not be extracted.")
            return

        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            answer = process_text(model, user_question, text)
            st.write("Question:", user_question)
            st.write("Predicted Answer:", answer)

if __name__ == '__main__':
    main()
