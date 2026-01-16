from dotenv import load_dotenv
import streamlit as st
import os
from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ---------- CACHE PDF PROCESSING ----------
@st.cache_resource(show_spinner=False)
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base, chunks


# ---------- CACHE RECOMMENDED QUESTIONS ----------
@st.cache_data(show_spinner=False)
def generate_recommended_questions(chunks, llm):
    sample_text = "\n".join(chunks[:3])

    prompt = f"""
You are given a document.

Based on the content below, generate 5 useful questions
that a student might ask to understand the document.

Content:
{sample_text}

Return ONLY the questions as a numbered list.
"""

    response = llm.invoke(prompt)
    return [q.strip() for q in response.content.split("\n") if q.strip()]


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF", layout="wide")
    st.header("📄 Ask your PDF 🤖")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        with st.spinner("📖 Processing PDF..."):
            knowledge_base, chunks = process_pdf(pdf)

        st.success("✅ PDF processed successfully")

        # LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.2
        )

        # ---------- RECOMMENDED QUESTIONS ----------
        st.markdown("### 🤔 Recommended Questions")

        recommended_questions = generate_recommended_questions(chunks, llm)

        if "question" not in st.session_state:
            st.session_state.question = ""

        for q in recommended_questions:
            if st.button(q):
                st.session_state.question = q

        # ---------- USER QUESTION ----------
        st.markdown("### 💬 Ask a Question")
        user_question = st.text_input(
            "Type your question here:",
            value=st.session_state.question
        )

        if user_question:
            with st.spinner("🤖 Generating answer..."):
                docs = knowledge_base.similarity_search(user_question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                response = llm.invoke(
                    f"""
Answer the question using ONLY the context below.
If the answer is not present, say:
"Answer not found in the document."

Context:
{context}

Question:
{user_question}
"""
                )

            st.subheader("✅ Answer")
            st.write(response.content)

            # Optional: show sources
            with st.expander("📚 View source text"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(doc.page_content[:500] + "...")
                    st.markdown("---")


if __name__ == "__main__":
    main()
