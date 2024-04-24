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
        extracted_text = page.extract_text() or ''
        # Normalize whitespace and remove line breaks within text
        extracted_text = ' '.join(extracted_text.replace('\n', ' ').split())
        text += extracted_text + ' '
    return text.strip()

def split_into_chunks(text, max_length=4096):
    # Break the text into sentences and then build chunks that are smaller than max_length
    sentences = text.split('.')
    current_chunk = ""
    chunks = []
    for sentence in sentences:
        sentence += '.'  # Add the period back to each sentence
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:  # Avoid adding empty strings
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(current_chunk.strip())
    return chunks


def initialize_knowledge_base(text, api_key):
    chunks = split_into_chunks(text)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def main():
    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    openai_api_key = os.getenv('OPENAI_API_KEY', st.secrets.get("OPENAI_API_KEY"))
    if not openai_api_key:
        st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    pdf_file = st.file_uploader("Upload your pdf", type="pdf")
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            st.error("The PDF appears to be empty or text could not be extracted.")
            return

        knowledge_pdf = initialize_knowledge_base(text, openai_api_key)

        user_question = st.text_input("Ask a question about the pdf")
       if user_question:
            docs = knowledge_pdf.similarity_search(user_question, top_k=5)
            llm = OpenAI(api_key=openai_api_key, model_name='gpt-3.5-turbo-0613')
            chain = load_qa_chain(llm, chain_type="map_rerank")
            response = chain.run(input_documents=docs, question=user_question)
        
            st.write("Question:", user_question)
            st.write("Context:")
            for doc in docs:
                st.write(doc)
        
            if response:
                st.write("Predicted Answer:")
                st.write(response)
            else:
                st.error("Unable to generate a valid answer. Please try rephrasing your question or ask about another aspect of the document.")


if __name__ == '__main__':
    main()
