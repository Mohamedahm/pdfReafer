import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import json

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Check if text is extracted
            text += page_text
    return text

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    # Load API key from a JSON file
    try:
        with open('api_key.json', 'r') as key_file:
            api_data = json.load(key_file)
            OPENAI_API_KEY = api_data['api_key']
    except FileNotFoundError:
        st.error("API key file not found. Please ensure the 'api_key.json' file is present.")
        return
    except json.JSONDecodeError:
        st.error("Error decoding the API key. Check the 'api_key.json' file format.")
        return

    pdf_file = st.file_uploader("Upload your pdf", type="pdf")
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        text = ' '.join(text.replace('\n', ' ').split())  # Normalize whitespace

        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)

        # Create embeddings from chunks
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        knowledge_pdf = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about the pdf")
        if user_question:
            docs = knowledge_pdf.similarity_search(user_question)

            llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo-0613')
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
