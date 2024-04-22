# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from googletrans import Translator

# def extract_text_from_pdf(pdf_file):
#     text = ""
#     pdf_reader = PdfReader(pdf_file)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def main():
#     st.set_page_config(page_title='Ask your pdf')
#     st.header('Ask your pdf')

#     pdf = st.file_uploader("Upload your pdf", type="pdf")

#     if pdf:
#         text = extract_text_from_pdf(pdf)

#         # Preprocess text
#         text = text.replace('\n', ' ')  # Remove line breaks
#         text = ' '.join(text.split())  # Remove extra whitespace

#         # Split text into chunks
#         text_splitter = CharacterTextSplitter(
#             separator='\n',
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # Create embeddings
#         embeddings = OpenAIEmbeddings(openai_api_key='sk-proj-mUFh9VAwG1abeJiLrvwJT3BlbkFJ5rtniMavQttDMbxW4Qwp')
#         knowledge_pdf = FAISS.from_texts(chunks, embeddings)

#         user_question = st.text_input("Ask question about pdf")

#         if user_question:
#             docs = knowledge_pdf.similarity_search(user_question)

#             llm = OpenAI(openai_api_key='sk-proj-mUFh9VAwG1abeJiLrvwJT3BlbkFJ5rtniMavQttDMbxW4Qwp', model_name='gpt-3.5-turbo-0613')
#             chain = load_qa_chain(llm, chain_type="stuff")
#             response = chain.run(input_documents=docs, question=user_question)

#             # Translate the response
#             translator = Translator()
#             translated_response = translator.translate(response, dest='en').text

#             st.write("Translated Answer:")
#             st.write(translated_response)

# if __name__ == '__main__':
#     main()

# import streamlit as st

# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# import json
# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from googletrans import Translator

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
    # Hardcoded API key
    api_key = "sk-proj-mUFh9VAwG1abeJiLrvwJT3BlbkFJ5rtniMavQttDMbxW4Qwp"

    # Use the API key in your code
    openai.api_key = api_key

    st.set_page_config(page_title='Ask your pdf')
    st.header('Ask your pdf')

    pdf_file = st.file_uploader("Upload your pdf", type="pdf")

    if pdf_file:
        text = extract_text_from_pdf(pdf_file)

        # Preprocess text
        text = text.replace('\n', ' ')  # Remove line breaks
        text = ' '.join(text.split())  # Remove extra whitespace

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        knowledge_pdf = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about the pdf")

        if user_question:
            docs = knowledge_pdf.similarity_search(user_question)

            llm = OpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo-0613')
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(docs)
            # Display the context and answer
            st.write("Question:", user_question)
            st.write("Context:")
            for doc in docs:
                st.write(doc)
            st.write("Predicted Answer:")
            st.write(response)

if __name__ == '__main__':
    main()
