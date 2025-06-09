# ====== IMPORTS ====== #

import hashlib
import fitz  # PyMuPDF
from typing import List, Dict
from fastapi import File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import shutil
import openai
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer # Metrics
from sklearn.metrics.pairwise import cosine_similarity # Metrics
import numpy as np # Metrics
import spacy # Metrics
from fastapi import FastAPI, HTTPException, Depends
import numpy as np
import os
import json
import datetime
import logging
import pickle
import faiss
from uuid import uuid4
from fastapi import FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPD
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI

# ====== IMPORTS ====== #



# ====== GLOBAL SET-UP ====== #




# ====== Logging Setup ====== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("parse_log.txt"),
        logging.StreamHandler()
    ]
)

# ====== SPACY SET-UP ====== #
nlp = spacy.load("en_core_web_sm")

# ====== BASEMODEL-INPUT-QUERY ====== #
class QueryInput(BaseModel):
    query: str

# ====== OPENAI_API_KEY ====== #
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


# ====== Config & Security Hardening ====== #
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ====== Responsible AI Manifest ====== #
RESPONSIBLE_AI_MANIFEST = {
    "policy": "No document or chunk processed by this pipeline will be used for LLM or embedding training unless explicitly approved by client X.",
    "timestamp": datetime.datetime.now().isoformat(),
    "enforced": True
}
with open("responsible_ai_manifest.json", "w") as f:
    json.dump(RESPONSIBLE_AI_MANIFEST, f, indent=2)

# ====== FastAPI App & Auth ====== #
app = FastAPI()
security = HTTPBasic()

# ===== CORS middleware for frontend integration (if web UI planned) ===== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Embedding Model ====== #
model = SentenceTransformer('all-MiniLM-L6-v2')  # Local model
embedding_dim = model.get_sentence_embedding_dimension() # change this to be dynamic

# ====== Ensure Index Storage Directory Exists ====== #
os.makedirs("faiss_index_store", exist_ok=True)

# ====== Vector DB Setup (FAISS) ====== #
index_path = "faiss_index_store/my_index.faiss"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print("✅ FAISS index loaded.")
else:
    index = faiss.IndexFlatL2(embedding_dim)
    print("⚠️ New FAISS index created.")


# ====== Load Chunk Metadata from Disk (if exists) ====== #

try:
    with open("doc_chunks.pkl", "rb") as f:
        chunk_metadata_store = pickle.load(f)
    index = faiss.read_index("faiss_index_store/my_index.faiss")
    print(f"✅ Loaded {len(chunk_metadata_store)} chunks from disk.")
except Exception as e:
    chunk_metadata_store = []
    index = faiss.IndexFlatL2(embedding_dim)
    print("⚠️ FAISS or Metadata load failed:", str(e))


# ✅ You're not missing anything. In fact, your new code is:
# Cleaner
# Logically ordered
# Avoids duplicate declarations
# Prevents reinitializing FAISS index unnecessarily

# 🔧 Why it's solid:
# ✅ Gracefully handles missing or corrupt files.
# ✅ Loads both doc_chunks.pkl and FAISS index together — avoids mismatch.
# ✅ Falls back cleanly to an empty metadata list and a new FAISS index.
# ✅ Avoids code duplication.
# ✅ Perfectly placed in the setup phase.


# ====== Load Chunk Metadata from Disk (if exists) ====== #




# ====== GLOBAL SET-UP ======#

# ====== OPERATIONS SET-UP ====== #

# ====== Retrieve Chunks ====== #
def retrieve_chunks_for_query(query: str, top_k: int = 5):
    # Step 1: Embed the user query
    query_vector = model.encode([query]).astype('float32')

    # Step 2: Search FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Step 3: Get corresponding chunk texts from stored metadata
    top_chunks = [doc_chunks[i]["chunk"] for i in indices[0]]

    return top_chunks

# ====== Utility Functions ====== #
def clean_text(text):
    return text.replace('\n', ' ').replace('\xa0', ' ').strip()

def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def encrypt_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

# ====== RBAC Layer (Simplified) ====== #
RBAC_USERS = {
    "admin": ["parse", "index", "query"],
    "viewer": ["query"]
}

def check_permission(user_role, action):
    return action in RBAC_USERS.get(user_role, [])

# ====== OPERATIONS SET-UP ====== #



# ====== PDF Parsing Function ====== #

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    raw_text = []
    metadata = {
        "file_name": os.path.basename(file_path),
        "page_count": len(doc),
        "trainable": False,
        "source_type": "pdf"
    }
    
    # Extract and clean text
    for i, page in enumerate(doc):
        text = page.get_text()
        if text:
            raw_text.append((i + 1, clean_text(text)))

    # Chunking and Encryption
    chunks = []
    for page_num, text in raw_text:
        logical_segments = text.split("\n\n") # \n\n - paragraph by paragraph  || space - word by word || . senetence by sentance || \n - line by line
        for segment in logical_segments:
            segment_chunks = chunk_text(segment)
            for chunk in segment_chunks:
                chunk_id = str(uuid4())
                encrypted_chunk = encrypt_text(chunk)
                chunks.append((chunk_id, chunk, page_num, encrypted_chunk))
    
    # Embedding and Indexing
    chunk_texts = [c[1] for c in chunks]
    embeddings = model.encode(chunk_texts)
    index.add(np.array(embeddings).astype('float32'))

    # Store metadata
    for i, (chunk_id, chunk, page_num, encrypted) in enumerate(chunks):
        metadata_record = {
            "id": chunk_id,
            "file_name": metadata['file_name'],
            "page": page_num,
            "chunk": chunk,
            "trainable": False,
            "timestamp": datetime.datetime.now().isoformat(),
            "encrypted_hash": encrypted
        }
        chunk_metadata_store.append(metadata_record)
        logging.info(f"Chunk Stored: {chunk_id} | File: {metadata['file_name']} | Page: {page_num}")

    # 🔍 Compute chunk quality metrics
    metrics = compute_chunk_quality_metrics(chunk_texts)
    logging.info(f"📊 Chunk Quality Metrics: {metrics}")


    # 💾 Save FAISS index to disk & metadata
    faiss.write_index(index, "faiss_index_store/my_index.faiss")
    # Save chunk metadata to disk for persistence
    with open("doc_chunks.pkl", "wb") as f:
        pickle.dump(chunk_metadata_store, f)


    # Return first 3 chunk texts for validation
    # ✅ Final output
    first_3_chunks = [meta["chunk"] for meta in chunk_metadata_store[:3]]
    return {
        "status": "success",
        "total_chunks_indexed": len(chunks),
        "first_3_chunks": first_3_chunks,
        "chunk_quality_metrics": metrics
    }    

# ====== PDF Parsing Function ====== #


# ====== CHUNK QUALITY METRICS ====== #

def compute_chunk_quality_metrics(chunks):

    # Metric 1: Meaningful Chunk Ratio
    meaningful_chunks = [c for c in chunks if len(c.strip()) > 20]
    meaningful_chunk_ratio = len(meaningful_chunks) / len(chunks)

    # Metric 2: Average Chunk Cohesion
    vectorizer = TfidfVectorizer().fit_transform(chunks)
    vectors = vectorizer.toarray()
    similarity_matrix = np.inner(vectors, vectors)
    np.fill_diagonal(similarity_matrix, 0)
    avg_cohesion = np.mean(similarity_matrix)

    # Metric 3: Redundancy Ratio
    redundant_count = sum(
        np.any(similarity_matrix[i] > 0.9) for i in range(len(chunks))
    )
    redundancy_ratio = redundant_count / len(chunks)

    # Metric 4: Entity Density per Chunk
    total_entities = sum(len(nlp(chunk).ents) for chunk in chunks)
    entity_density = total_entities / len(chunks)

    # Metric 5: Token Efficiency
    total_tokens = sum(len(nlp(chunk)) for chunk in chunks)
    avg_chunk_len = sum(len(chunk.split()) for chunk in chunks) / len(chunks)
    token_efficiency = total_tokens / len(chunks) / avg_chunk_len

    # Metric 6: Context Overlap (between adjacent chunks)
    overlap_scores = []
    for i in range(len(chunks) - 1):
        vec_pair = TfidfVectorizer().fit_transform([chunks[i], chunks[i + 1]]).toarray()
        sim = cosine_similarity([vec_pair[0]], [vec_pair[1]])[0][0]
        overlap_scores.append(sim)
    context_overlap = np.mean(overlap_scores) if overlap_scores else 0

    # Metric 7: Semantic Dissimilarity (inverse of average similarity between adjacent chunks)
    semantic_similarities = overlap_scores  # reuse from above
    semantic_dissimilarity = 1 - np.mean(semantic_similarities) if semantic_similarities else 0

    # Metric 8: Average Chunk Length Variance
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    length_variance = np.var(chunk_lengths)

    return {
        "meaningful_chunk_ratio": round(meaningful_chunk_ratio, 3),
        "average_chunk_cohesion": round(avg_cohesion, 3),
        "redundancy_ratio": round(redundancy_ratio, 3),
        "entity_density_per_chunk": round(entity_density, 3),
        "token_efficiency": round(token_efficiency, 3),
        "context_overlap": round(context_overlap, 3),
        "semantic_dissimilarity": round(semantic_dissimilarity, 3),
        "chunk_length_variance": round(length_variance, 3)
    }

# ====== CHUNK QUALITY METRICS ====== #




# ====== GLOBAL SET-UP ====== #




# ====== ENDPOINTS ===== #

# ====== Endpoint for Multiple PDF Upload and Parsing ====== #

@app.post("/parse-multiple-pdfs")
def upload_multiple_pdfs(
    files: List[UploadFile] = File(...),
    credentials: HTTPBasicCredentials = Depends(security)
):
    user_role = credentials.username.lower()
    if not check_permission(user_role, "parse"):
        raise HTTPException(status_code=403, detail="Access Denied.")
    
    total_chunks = 0
    all_top_chunks = []
    all_metrics = []

    for file in files:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = parse_pdf(temp_file_path)
            total_chunks += result["total_chunks_indexed"]
            all_top_chunks.extend(result["first_3_chunks"])
            all_metrics.append(result["chunk_quality_metrics"])  # ✅ Collect per-file metrics
            os.remove(temp_file_path)

        except Exception as e:
            logging.error(f"Parsing Error in {file.filename}: {str(e)}")

    # Limit to first 3 chunks across all files
    return JSONResponse(content={
        "status": "success",
        "total_chunks_indexed": total_chunks,
        "first_3_chunks": [f"***{chunk}***" for chunk in all_top_chunks[:3]],
        "chunk_quality_metrics": all_metrics  # ✅ Now correctly returned
    }, status_code=200)


# ====== Endpoint for Multiple PDF Upload and Parsing ====== #

# ========= Methods ========= #

def grounded_context_match_rate(query, answer, matched_chunks):
    return float(any(chunk.lower() in answer.lower() for chunk in matched_chunks))

def query_drift_distance(query, answer, matched_chunks, query_vector, chunk_vectors):
    if query_vector is None or chunk_vectors is None:
        return 0.0
    avg_chunk_vector = np.mean(chunk_vectors, axis=0, keepdims=True)
    drift = 1 - cosine_similarity(query_vector, avg_chunk_vector)[0][0]
    return float(f"{drift:.3f}")

def redundancy_load_factor(query, answer, matched_chunks):
    unique_chunks = set(matched_chunks)
    return round(1 - (len(unique_chunks) / len(matched_chunks)), 3)

def zero_shot_fallback_indicator(query, answer, matched_chunks):
    return float("I don't know" in answer or "I'm not sure" in answer)

def llm_used_retrieval_percent(query, answer, matched_chunks):
    used = sum(1 for chunk in matched_chunks if any(word in answer for word in chunk.split()[:5]))
    return round(used / len(matched_chunks), 3)

def composite_score(metrics):
    keys = ["grounded_context_match_rate", "query_drift_distance", "redundancy_load_factor", "llm_used_retrieval_percent"]
    weights = [0.3, 0.2, 0.2, 0.3]
    score = sum(float(metrics.get(k, 0.0)) * w for k, w in zip(keys, weights))
    return float(f"{score:.3f}")

def anchor_keyword_coverage(query, matched_chunks):
    vectorizer = CountVectorizer(stop_words='english')
    query_tokens = vectorizer.build_tokenizer()(query.lower())
    chunk_text = " ".join(matched_chunks).lower()
    matched = sum(1 for token in query_tokens if token in chunk_text)
    return round(matched / len(query_tokens), 3) if query_tokens else 0.0

def source_diversity_index(chunk_metadata_store, indices):
    doc_ids = [chunk_metadata_store[i]["file_name"] for i in indices[0]]
    diversity = len(set(doc_ids)) / len(doc_ids)
    return round(diversity, 3)

def noise_to_signal_ratio(matched_chunks):
    if not matched_chunks:
        return 0.0
    noisy = sum(1 for chunk in matched_chunks if len(chunk.split()) < 20)
    return round(noisy / len(matched_chunks), 3)

def topical_drift_index(matched_chunks, embedding_model):
    if embedding_model is None:
        return 0.0
    vectors = embedding_model.encode(matched_chunks)
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = np.mean(sim_matrix)
    drift = 1 - avg_sim
    return float(f"{drift:.3f}")

def chunk_freshness_score(chunk_metadata_store, indices):
    timestamps = []
    for i in indices[0]:
        ts = chunk_metadata_store[i].get("timestamp")
        if ts:
            try:
                timestamps.append(datetime.datetime.fromisoformat(ts))
            except:
                continue
    if not timestamps:
        return 0.0
    now = datetime.datetime.now()
    avg_age = sum((now - ts).days for ts in timestamps) / len(timestamps)
    freshness = max(0.0, 1 - avg_age / 365)
    return round(freshness, 3)

def compression_utility_index(answer, matched_chunks):
    context_tokens = sum(len(chunk.split()) for chunk in matched_chunks)
    answer_tokens = len(answer.split())
    return round(answer_tokens / context_tokens, 3) if context_tokens else 0.0



# ========= Methods ========= #


# ====== Endpoint for Querying ====== #

@app.post("/ask")
def ask_question(input: QueryInput, credentials: HTTPBasicCredentials = Depends(security)):
    # 🔒 Check permission
    user_role = credentials.username.lower()
    if not check_permission(user_role, "query"):
        raise HTTPException(status_code=403, detail="Access Denied.")
    
    # 🔍 Step 1: Embed the query
    query = input.query
    query_vector = model.encode([query]).astype('float32')

    # 🔎 Step 2: Retrieve top-k similar chunks
    distances, indices = index.search(query_vector, k=5)
    matched_chunks = [chunk_metadata_store[i]["chunk"] for i in indices[0]]
    chunk_vectors = model.encode(matched_chunks).astype('float32')
    context = "\n\n".join(matched_chunks)

    # 💬 Step 3: Format GPT-4 request with structured context and question
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the context below to answer accurately."},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Question:\n{query}"}
    ]


    # 🧠 Step 4: Call GPT-4 to generate the answer
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    answer = response.choices[0].message.content
    
    metrics = {
        "grounded_context_match_rate": grounded_context_match_rate(query, answer, matched_chunks),
        "query_drift_distance": query_drift_distance(query, answer, matched_chunks, query_vector, chunk_vectors),
        "redundancy_load_factor": redundancy_load_factor(query, answer, matched_chunks),
        "zero_shot_fallback_indicator": zero_shot_fallback_indicator(query, answer, matched_chunks),
        "llm_used_retrieval_percent": llm_used_retrieval_percent(query, answer, matched_chunks),
        "anchor_keyword_coverage": anchor_keyword_coverage(query, matched_chunks),
        "source_diversity_index": source_diversity_index(chunk_metadata_store, indices),
        "noise_to_signal_ratio": noise_to_signal_ratio(matched_chunks),
        "topical_drift_index": topical_drift_index(matched_chunks, model),
        "chunk_freshness_score": chunk_freshness_score(chunk_metadata_store, indices),
        "compression_utility_index": compression_utility_index(answer, matched_chunks)
    }
    metrics["composite_score"] = composite_score(metrics)

    
    # ✅ 🔧 Fix: convert to float for JSON serialization
    metrics = {k: float(v) for k, v in metrics.items()}    

    # 📤 Step 5: Return full structured response
    return {
        "query": query,
        "matched_chunks": matched_chunks,
        "response": answer,
        "metrics": metrics
    }

# ====== Endpoint for Querying ====== #

# ====== ENDPOINTS ===== #



# ===== in future ===== #

# ✅ 6. Suggestions for production deployment:
# Area	Recommendation
# Security	Implement JWT or OAuth instead of basic auth.
# Error Handling	Add proper try-except blocks in all routes.
# Testing	Add unit tests using pytest for parsing, chunking, querying.
# Rate Limiting	Use FastAPI’s middleware or Nginx for abuse protection.
# Monitoring	Add Prometheus/Grafana or simple uptime health endpoint.
# Docs	Enable FastAPI Swagger UI at /docs. Very useful!
# Persistence	Consider using SQLite/Redis/Postgres for chunk metadata instead of pkl.

# ===== in future ===== #
