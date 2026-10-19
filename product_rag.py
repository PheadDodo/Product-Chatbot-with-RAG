from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from time import perf_counter

# List all markdown files in the specified folder
def load_brochures(folder):
    documents = []  # Initialize the list to store document dictionaries
    for path in sorted(folder.glob("*.md")):
        text=path.read_text(encoding="utf-8")
        documents.append({
            "source": path.name,
            "text": text,
        })
    return documents

def split_into_pages(documents):
    pages = []

    for document in documents:
        parts = document["text"].split("## Page ")

        for part in parts[1:]:
            page_number, page_text = part.split("\n", 1)

            pages.append({
                "source": document["source"],
                "page": int(page_number.strip()),
                "text": page_text.strip()
            })

    return pages

folder = Path("./data/brochures")
documents = load_brochures(folder)
pages = split_into_pages(documents)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device=device,
)

reranker = CrossEncoder(
    "Qwen/Qwen3-Reranker-0.6B",
    device=device,
)

query = (
    "We have Larox pressure filters but performance drops "
    "whenever our feed grade changes"
)

# encode the query into an embedding
query_embedding = model.encode(
    [query],
    prompt_name="query",
    normalize_embeddings=True,
)

page_texts = [page["text"] for page in pages]

# encode the page texts into embeddings
page_embeddings = model.encode(
    page_texts,
    batch_size=4,
    normalize_embeddings=True,
)

scores = model.similarity(query_embedding, page_embeddings)[0]

# find the most relevant page
top_score, top_index = scores.topk(k=15)

# assess which passages answer the question based on similarity scores
candidates = [pages[index.item()] for index in top_index]
candidate_texts = [page["text"] for page in candidates]

start_time = perf_counter()

reranked = reranker.rank(
    query,
    candidate_texts,
    top_k=3,
    batch_size=4,
    prompt=(
        "Given a customer equipment problem, retrieve passages "
        "containing substantive evidence for a product or service "
        "capability that addresses the problem. "
        "Merely naming the equipment is insufficient."
    ),
)

end_time = perf_counter() - start_time
print("Reranking time:", round(end_time, 4), "seconds")

for result in reranked:
    page = candidates[result["corpus_id"]]

    print("Source:", page["source"], "| Page:", page["page"])
    print("Reranker score:", round(result["score"], 4))
    print(page["text"])
    print()

