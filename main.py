import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# STEP 1: LOAD DATASET
# ==================================================

df = pd.read_csv("company_rag_dataset.csv")

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==================================================
# STEP 2: CHECK MISSING VALUES
# ==================================================

print("\nMissing values:")
print(df.isnull().sum())

# Remove rows where content is missing
df = df.dropna(subset=["content"]).copy()


# ==================================================
# STEP 3: CLEAN THE TEXT
# ==================================================

def clean_text(text):

    # Convert value into string
    text = str(text)

    # Replace multiple spaces/newlines/tabs with one space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from beginning and end
    text = text.strip()

    return text


# Apply cleaning to every policy
df["clean_content"] = df["content"].apply(clean_text)

# Remove empty content
df = df[df["clean_content"] != ""].copy()

print("\nRecords after preprocessing:", len(df))


# ==================================================
# STEP 4: CHUNKING FUNCTION
# ==================================================

def create_chunks(text, chunk_size=80, overlap=15):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")

    # Convert text into words
    words = text.split()

    chunks = []

    # 80 - 15 = 65
    step = chunk_size - overlap

    for start in range(0, len(words), step):

        # Take maximum 80 words
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        # Convert words back into text
        chunk = " ".join(chunk_words)

        # Store chunk
        chunks.append(chunk)

        # Stop when we reach the end
        if start + chunk_size >= len(words):
            break

    return chunks


# ==================================================
# STEP 5: CREATE CHUNKS FOR ALL POLICIES
# ==================================================

all_chunks = []

for _, row in df.iterrows():

    policy_chunks = create_chunks(
        row["clean_content"],
        chunk_size=80,
        overlap=15
    )

    for index, chunk in enumerate(policy_chunks):

        all_chunks.append({

            "chunk_id": f"{row['id']}_chunk_{index + 1}",

            "policy_id": row["id"],

            "category": row["category"],

            "title": row["title"],

            "owner": row["owner"],

            "text": chunk
        })


print("\nOriginal policies:", len(df))

print("Total chunks created:", len(all_chunks))


# ==================================================
# STEP 6: DISPLAY FIRST CHUNK
# ==================================================

if all_chunks:

    print("\nFirst chunk:")

    print("Chunk ID:", all_chunks[0]["chunk_id"])

    print("Policy ID:", all_chunks[0]["policy_id"])

    print("Category:", all_chunks[0]["category"])

    print("Title:", all_chunks[0]["title"])

    print("Owner:", all_chunks[0]["owner"])

    print("Text:", all_chunks[0]["text"])


# ==================================================
# STEP 7: LOAD EMBEDDING MODEL
# ==================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==================================================
# STEP 8: GET ONLY CHUNK TEXT
# ==================================================

chunk_texts = []

for chunk in all_chunks:

    chunk_texts.append(chunk["text"])


print("\nNumber of texts to embed:", len(chunk_texts))


# ==================================================
# STEP 9: GENERATE EMBEDDINGS
# ==================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    chunk_texts,
    batch_size=16,
    show_progress_bar=True
)


# ==================================================
# STEP 10: DISPLAY EMBEDDING INFORMATION
# ==================================================

print("\nEmbeddings generated successfully!")

print("Embedding shape:", embeddings.shape)


# ==================================================
# STEP 11: DISPLAY FIRST EMBEDDING
# ==================================================

print("\nFirst chunk text:")

print(chunk_texts[0])

print("\nFirst 10 numbers of its embedding:")

print(embeddings[0][:10])

# ==================================================
# STEP 12: CREATE CHROMADB DATABASE
# ==================================================

print("\nCreating ChromaDB database...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="company_policies",
    metadata={"hnsw:space": "cosine"}
)


# ==================================================
# STEP 13: PREPARE DATA FOR CHROMADB
# ==================================================

ids = []
documents = []
metadatas = []

for chunk in all_chunks:

    ids.append(chunk["chunk_id"])

    documents.append(chunk["text"])

    metadatas.append({
        "policy_id": chunk["policy_id"],
        "category": chunk["category"],
        "title": chunk["title"],
        "owner": chunk["owner"]
    })


# ==================================================
# STEP 14: STORE DATA IN CHROMADB
# ==================================================

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# ==================================================
# STEP 15: VERIFY STORED DATA
# ==================================================

print("\nEmbeddings stored successfully!")

print(
    "Total chunks stored in ChromaDB:",
    collection.count()
)