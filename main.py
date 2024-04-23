import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ''
    return text

def split_into_chunks(text, max_length=1000):
    """
    Splits the text into chunks of a specified max_length.
    """
    sentences = text.split('.')
    current_chunk = ""
    chunks = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_length:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    pdf_file = st.file_uploader("Upload your pdf", type="pdf")
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        text = ' '.join(text.replace('\n', ' ').split())
        chunks = split_into_chunks(text, max_length=4096)  # Adjust based on your model's token limit

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        knowledge_pdf = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            docs = knowledge_pdf.similarity_search(user_question, top_k=5)  # Adjust top_k as needed
            llm = OpenAI(api_key=openai_api_key, model_name='gpt-3.5-turbo-0613')
            chain = load_qa_chain(llm, chain_type="extractive_qa")
            response = chain.run(input_documents=docs, question=user_question)

            st.write("Question:", user_question)
            st.write("Context:")
            for doc in docs:
                st.write(doc)
            st.write("Predicted Answer:")
            st.write(response)

if __name__ == '__main__':
    main()
