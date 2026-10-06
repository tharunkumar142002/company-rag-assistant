import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# STEP 2: CONNECT TO EXISTING CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="company_policies"
)

print("Connected to ChromaDB")
print("Stored chunks:", collection.count())


# ==========================================
# STEP 3: GET QUESTION FROM USER
# ==========================================

question = input("\nAsk a company policy question: ")


# ==========================================
# STEP 4: CONVERT QUESTION INTO EMBEDDING
# ==========================================

query_embedding = embedding_model.encode(
    [question]
)

print("\nQuery embedding shape:")
print(query_embedding.shape)


# ==========================================
# STEP 5: SEARCH CHROMADB
# ==========================================

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
    include=["documents", "metadatas", "distances"]
)


# ==========================================
# STEP 6: DISPLAY RETRIEVED CHUNKS
# ==========================================

print("\nRetrieved Chunks:")

for i in range(len(results["documents"][0])):

    print("\n--------------------------------")

    print("Rank:", i + 1)

    print(
        "Title:",
        results["metadatas"][0][i]["title"]
    )

    print(
        "Policy ID:",
        results["metadatas"][0][i]["policy_id"]
    )

    print(
        "Cosine Distance:",
        results["distances"][0][i]
    )

    print("Text:")
    print(results["documents"][0][i])


# ==========================================
# STEP 7: GET RETRIEVED DOCUMENTS
# ==========================================

retrieved_documents = results["documents"][0]


# ==========================================
# STEP 8: COMBINE DOCUMENTS INTO CONTEXT
# ==========================================

context = "\n\n".join(retrieved_documents)


# ==========================================
# STEP 9: CREATE PROMPT
# ==========================================

prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the
information provided in the context below.

If the answer is not available in the context,
say:

"I don't know based on the provided company policies."

Keep the answer short and clear.

CONTEXT:

{context}


QUESTION:

{question}


ANSWER:
"""


# ==========================================
# STEP 10: SEND PROMPT TO QWEN
# ==========================================

print("\nGenerating answer using Qwen...\n")

response = ollama.chat(
    model="qwen3:1.7b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# STEP 11: GET FINAL ANSWER
# ==========================================

final_answer = response["message"]["content"]


# ==========================================
# STEP 12: DISPLAY FINAL ANSWER
# ==========================================

print("\n================================")
print("FINAL ANSWER")
print("================================")

print(final_answer)