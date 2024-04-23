import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from googletrans import Translator

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    # Set page configuration as the first Streamlit call
    st.set_page_config(page_title='Ask your pdf')

    st.header('Ask your pdf')

    # Retrieve the API key from environment variable or secrets
    openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    pdf_file = st.file_uploader("Upload your pdf", type="pdf")

    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        # Preprocess and split text
        text = text.replace('\n', ' ').join(text.split())
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_pdf = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            docs = knowledge_pdf.similarity_search(user_question)
            llm = OpenAI(openai_api_key=openai_api_key, model_name='text-embedding-3-small')
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Display the context and answer
            st.write("Question:", user_question)
            st.write("Context:")
            for doc in docs:
                st.write(doc)
            st.write("Predicted Answer:")
            st.write(response)

if __name__ == '__main__':
    main()
