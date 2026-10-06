import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


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
# STEP 3: USER QUESTION
# ==========================================

query = "How many annual leave days do I get?"

print("\nUser Question:")
print(query)


# ==========================================
# STEP 4: CONVERT QUESTION INTO EMBEDDING
# ==========================================

query_embedding = model.encode([query])

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
# STEP 6: DISPLAY RESULTS
# ==========================================

print("\nTop 3 matching chunks:")

for i in range(len(results["documents"][0])):

    print("\n--------------------------------")

    print("Rank:", i + 1)

    print("Text:")
    print(results["documents"][0][i])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nCosine Distance:")
    print(results["distances"][0][i])