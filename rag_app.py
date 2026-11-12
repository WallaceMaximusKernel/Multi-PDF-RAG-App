
import os
import hashlib
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Application")
st.write("Upload multiple PDFs and ask questions across all documents.")


# --------------------------------------------------
# 2. API key validation
# --------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please configure your environment variable.")
    st.stop()


# --------------------------------------------------
# 3. Initialize session state
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# --------------------------------------------------
# 4. Upload multiple PDF files
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Multiple PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Total PDFs uploaded: {len(uploaded_files)}")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")


# --------------------------------------------------
# 5. Check whether uploaded files have changed
# --------------------------------------------------

signature = None

if uploaded_files:
    signature_data = "".join(
        f"{f.name}:{hashlib.sha256(f.getvalue()).hexdigest()}"
        for f in uploaded_files
    )
    signature = hashlib.sha256(signature_data.encode()).hexdigest()

    if signature != st.session_state.indexed_signature:
        st.session_state.store = None
        st.session_state.chunk_count = 0


# --------------------------------------------------
# 6. Load, split and index all PDFs
# --------------------------------------------------

if uploaded_files:
    st.write(f"**Uploaded documents:** {len(uploaded_files)}")

    if st.button("Build Vector Database"):

        all_documents = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                for uploaded_file in uploaded_files:

                    file_path = os.path.join(
                        temp_dir, uploaded_file.name
                    )

                    with open(file_path, "wb") as pdf_file:
                        pdf_file.write(uploaded_file.getvalue())

                    loader = PyPDFLoader(file_path)
                    documents = loader.load()

                    # Preserve original filename and page metadata
                    for document in documents:
                        document.metadata["source"] = uploaded_file.name

                    all_documents.extend(documents)

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(all_documents)

            if not chunks:
                st.error("No readable text found in the uploaded PDFs.")
                st.stop()

            # Create embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            # Create one FAISS vector database
            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.store = vector_store
            st.session_state.indexed_signature = signature
            st.session_state.chunk_count = len(chunks)

            st.success(
                f"Successfully indexed {len(chunks)} chunks "
                f"from {len(uploaded_files)} PDFs!"
            )

        except Exception as e:
            st.error(f"Error while indexing PDFs: {e}")


# --------------------------------------------------
# 7. Display indexing status
# --------------------------------------------------

if st.session_state.store is not None:

    st.divider()

    st.success(
        f"Vector database ready: "
        f"{st.session_state.chunk_count} chunks indexed."
    )


# --------------------------------------------------
# 8. Ask questions across all uploaded PDFs
# --------------------------------------------------

if st.session_state.store is not None:

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Get Answer"):

        if question.strip():

            try:
                # Retrieve relevant chunks
                retriever = st.session_state.store.as_retriever(
                    search_kwargs={"k": 5}
                )

                retrieved_docs = retriever.invoke(question)

                # Prepare context from retrieved documents
                context = ""

                for index, document in enumerate(retrieved_docs):
                    context += (
                        f"\nDocument {index + 1}\n"
                        f"Source: {document.metadata.get('source', 'Unknown')}\n"
                        f"Page: {document.metadata.get('page', 0) + 1}\n"
                        f"Content: {document.page_content}\n"
                    )

                # Initialize language model
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                prompt = f"""
                You are a document-based question answering assistant.

                Answer the user's question using ONLY the
                information provided in the retrieved context.

                Instructions:
                1. Do not invent information.
                2. If the documents disagree, clearly explain
                   the conflicting information and identify
                   the respective sources.
                3. If the answer is not available, say so.
                4. Give a clear and concise answer.

                Retrieved Context:
                {context}

                User Question:
                {question}

                Answer:
                """

                response = llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response.content)

                # Display source documents
                st.subheader("Sources")

                seen_sources = set()

                for document in retrieved_docs:

                    filename = document.metadata.get(
                        "source", "Unknown"
                    )

                    page = document.metadata.get("page", 0) + 1

                    source = (filename, page)

                    if source not in seen_sources:
                        st.caption(
                            f"📄 {filename} - Page {page}"
                        )
                        seen_sources.add(source)

            except Exception as e:
                st.error(f"Error while generating answer: {e}")

        else:
            st.warning("Please enter a question.")
