import argparse

import faiss
import uvicorn

from fastapi import FastAPI, HTTPException
from transformers import AutoConfig, set_seed

from activeft.sift import Retriever
from models.client_request_doc import ClientRequestDoc
from models.embedder import EMBEDDING_MODEL_PATH
from kb.utils import load_knowledge_base


# Initialize FastAPI application
app = FastAPI()

# Load the embedding model and tokenizer
embedding_model_config = AutoConfig.from_pretrained(EMBEDDING_MODEL_PATH)
dimension = embedding_model_config.hidden_size

# Initialize FAISS index for similarity search
index = faiss.IndexFlatIP(dimension)
knowledge_base_samples = None
retriever = None


# Define the search endpoint
@app.post("/search/")
async def search(client_request: ClientRequestDoc):
    query_embedding = client_request.query_embedding
    num_retrieved_samples = client_request.num_retrieved_samples
    if query_embedding.size(1) != dimension:
        raise HTTPException(status_code=400, detail=f"Embedding dimension must be {dimension}, and Embedding model should be {EMBEDDING_MODEL_PATH}.")
    
    # Perform the search using the retriever
    values, indices, _, _ = retriever.search(query_embedding, N=num_retrieved_samples, K=1000)
    scores = [abs(float(val)) for val in values]
    retrieved_samples = {knowledge_base_samples[int(idx)]: scores[i] for i, idx in enumerate(indices)}
    return {"retrieved_samples": retrieved_samples}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_base_path', type=str, help="Path to the knowledge base.", required=True)
    parser.add_argument(
        "--only_faiss",
        action="store_true",
        help="Flag to indicate whether to only use Faiss for search."
    )
    args = parser.parse_args()
    set_seed(42)

    # Load and concatenate datasets
    knowledge_base_samples, embs = load_knowledge_base(args.knowledge_base_path)
    index.add(embs)
    retriever = Retriever(index, only_faiss=args.only_faiss)

    # Start the TTT application
    uvicorn.run(app, host="0.0.0.0", port=8000)
