import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  # Assuming this is still part of langchain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from googletrans import Translator

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

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
        text = text.replace('\n', ' ').join(text.split())
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_pdf = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            docs = knowledge_pdf.similarity_search(user_question)
            llm = OpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo-0613')
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write("Question:", user_question)
            st.write("Context:")
            for doc in docs:
                st.write(doc)
            st.write("Predicted Answer:")
            st.write(response)

if __name__ == '__main__':
    main()
