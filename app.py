import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f5f7ff 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #312e81;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 35px;
    }

    .answer-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border-left: 6px solid #6366f1;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
        color: #1e293b;
        font-size: 17px;
        line-height: 1.7;
    }

    .source-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        color: #334155;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

@st.cache_resource
def connect_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="company_policies"
    )

    return collection


collection = connect_database()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 RAG Assistant")

    st.markdown(
        """
        ### How it works

        **1.** You ask a question

        ↓

        **2.** SentenceTransformer creates an embedding

        ↓

        **3.** ChromaDB searches company policies

        ↓

        **4.** Relevant policies are retrieved

        ↓

        **5.** Qwen generates the final answer
        """
    )

    st.divider()

    st.metric(
        "Knowledge Chunks",
        collection.count()
    )

    st.write("🧠 Embedding Model")
    st.code("all-MiniLM-L6-v2")

    st.write("✨ Generative LLM")
    st.code("qwen3:1.7b")

    st.write("🗄️ Vector Database")
    st.code("ChromaDB")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏢 Company AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about company policies and get
    answers powered by Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "💬 Ask a company policy question",
    placeholder="Example: How many annual leave days do I get?"
)


ask_button = st.button(
    "✨ Ask Assistant",
    type="primary"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🔍 Searching company policies..."
        ):

            # ------------------------------------------------
            # STEP 1: CREATE QUERY EMBEDDING
            # ------------------------------------------------

            query_embedding = embedding_model.encode(
                [question]
            )


            # ------------------------------------------------
            # STEP 2: SEARCH CHROMADB
            # ------------------------------------------------

            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )


            # ------------------------------------------------
            # STEP 3: GET RETRIEVED DOCUMENTS
            # ------------------------------------------------

            documents = results["documents"][0]

            metadatas = results["metadatas"][0]

            distances = results["distances"][0]


            # ------------------------------------------------
            # STEP 4: BUILD CONTEXT
            # ------------------------------------------------

            context = "\n\n".join(documents)


            # ------------------------------------------------
            # STEP 5: BUILD PROMPT
            # ------------------------------------------------

            prompt = f"""
You are a helpful company policy assistant.

Answer the user's question using ONLY the information
provided in the company policy context below.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I don't know based on the provided company policies."

Keep the answer short, clear and professional.

COMPANY POLICY CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


            # ------------------------------------------------
            # STEP 6: SEND TO QWEN THROUGH OLLAMA
            # ------------------------------------------------

            response = ollama.chat(
                model="qwen3:1.7b",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            # ------------------------------------------------
            # STEP 7: GET FINAL ANSWER
            # ------------------------------------------------

            answer = response[
                "message"
            ][
                "content"
            ]


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.markdown("### 🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-card">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        st.markdown("### 📚 Retrieved Sources")

        for i in range(len(documents)):

            metadata = metadatas[i]

            with st.expander(
                f"📄 Source {i + 1}: "
                f"{metadata['title']}"
            ):

                st.write(
                    "**Policy ID:**",
                    metadata["policy_id"]
                )

                st.write(
                    "**Category:**",
                    metadata["category"]
                )

                st.write(
                    "**Owner:**",
                    metadata["owner"]
                )

                st.write(
                    "**Cosine Distance:**",
                    round(distances[i], 4)
                )

                st.write(
                    "**Retrieved Text:**"
                )

                st.write(
                    documents[i]
                )