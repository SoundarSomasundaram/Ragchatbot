from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


def main():
    # Load environment variables
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 📄🤖")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Read PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # FREE embeddings (no API key needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # GROQ LLM (LLaMA 3)
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
             model="llama-3.1-8b-instant",
            temperature=0.2
        )

        # User question
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=4)

            # Combine retrieved text
            context = "\n\n".join([doc.page_content for doc in docs])

            # Ask LLM
            response = llm.invoke(
                f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found in the document."

Context:
{context}

Question:
{user_question}
"""
            )

            st.subheader("Answer:")
            st.write(response.content)


if __name__ == '__main__':
    main()
