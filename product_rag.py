import gc
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from embedding_cache import load_or_build, local_model_identity

project_dir = Path(__file__).resolve().parent
MODEL_DIR = project_dir / "models"


EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
CANDIDATE_LIMIT = 60
RESULT_LIMIT = 10


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_processed_records(folder):
    paths = sorted(folder.glob("*-body-passages.json"))

    if not paths:
        raise FileNotFoundError(
            f"No body-passage files found in {folder}. "
            "Run prepare_brochures.py first."
        )

    records = []

    for path in paths:
        file_records = json.loads(
            path.read_text(encoding="utf-8")
        )
        records.extend(file_records)

    return records


def _configure_model_cache():
    """Configure this process only when retrieval is explicitly requested."""
    os.environ["HF_HUB_CACHE"] = str(MODEL_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_DIR)
    os.environ["HF_XET_CACHE"] = str(MODEL_DIR / "xet")
    os.environ["HF_ASSETS_CACHE"] = str(MODEL_DIR / "assets")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


class _ProductRetriever:
    """Own the retrieval models for one call, before GGUF generation starts."""

    def __init__(self, records):
        self.records = records
        self.model = None
        self.reranker = None
        self.record_embeddings = None
        self._torch = None
        self._closed = False

    def load(self):
        # These assignments must precede imports that read HF cache settings.
        load_started = perf_counter()
        _configure_model_cache()
        try:
            import torch

            self._torch = torch
            from sentence_transformers import SentenceTransformer, CrossEncoder
            from transformers import BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this loading configuration.")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            embedding_identity = local_model_identity(MODEL_DIR, EMBEDDING_MODEL_NAME)
            revision_kwargs = {}
            if embedding_identity["revision"]:
                # Tie loading and the vector-cache key to the same local snapshot.
                revision_kwargs["revision"] = embedding_identity["revision"]
            self.model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(MODEL_DIR),
                **revision_kwargs,
                model_kwargs={
                    "quantization_config": quantization_config,
                    "dtype": torch.bfloat16,
                    "device_map": {"": 0},
                },
            )

            self.reranker = CrossEncoder(
                RERANKER_MODEL_NAME,
                cache_folder=str(MODEL_DIR),
                model_kwargs={
                    "quantization_config": quantization_config,
                    "dtype": torch.bfloat16,
                    "device_map": {"": 0},
                },
            )

            record_texts = [record["text"] for record in self.records]
            encoding_settings = {
                "batch_size": 4,
                "normalize_embeddings": True,
                "precision": "float32",
                "convert_to_numpy": True,
            }

            def encode_corpus():
                return self.model.encode(record_texts, **encoding_settings)

            # Prefer the actual loaded model's commit; it also handles the first
            # download, when refs/main did not exist before construction.
            commit = getattr(self.model[0].auto_model.config, "_commit_hash", None)
            if commit:
                embedding_identity = local_model_identity(MODEL_DIR, EMBEDDING_MODEL_NAME, commit)

            corpus_started = perf_counter()
            if embedding_identity["revision"]:
                from importlib.metadata import PackageNotFoundError, version

                package_versions = {}
                for package in ("sentence-transformers", "transformers", "torch", "bitsandbytes"):
                    try:
                        package_versions[package] = version(package)
                    except PackageNotFoundError:
                        package_versions[package] = None
                cache_settings = {
                    **embedding_identity,
                    "encoding": encoding_settings,
                    "prompts": self.model.prompts,
                    "default_prompt_name": self.model.default_prompt_name,
                    "max_seq_length": self.model.max_seq_length,
                    "truncate_dim": self.model.truncate_dim,
                    "quantization": {
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_compute_dtype": "bfloat16",
                        "dtype": "bfloat16",
                    },
                    "package_versions": package_versions,
                }
                self.record_embeddings, cache_status = load_or_build(
                    cache_dir=project_dir / "data" / "processed" / "embedding_cache",
                    records=self.records,
                    settings=cache_settings,
                    expected_dimension=self.model.get_sentence_embedding_dimension(),
                    encode=encode_corpus,
                )
            else:
                # An unidentified revision is never trusted for persistent reuse.
                self.record_embeddings = encode_corpus()
                cache_status = "BYPASS (unresolved model revision)"
            print("Corpus embedding cache:", cache_status)
            print("Corpus embedding/cache time:", round(perf_counter() - corpus_started, 4), "seconds")
            print("Retrieval model/corpus load time:", round(perf_counter() - load_started, 4), "seconds", flush=True)
        except BaseException as error:
            # Failed constructors can otherwise survive through traceback locals.
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise

    def retrieve(self, query, top_k):
        query_embedding = scores = top_values = top_indices = None
        reranked = result = index = None
        try:
            query_embedding = self.model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
            )

            scores = self.model.similarity(
                query_embedding,
                self.record_embeddings,
            )[0]

            candidate_count = min(CANDIDATE_LIMIT, len(self.records))
            top_values, top_indices = scores.topk(k=candidate_count)

            candidates = [
                self.records[candidate_index.item()]
                for candidate_index in top_indices
            ]

            print("\nEmbedding candidates before reranking:")

            for rank, index in enumerate(top_indices, start=1):
                record_index = index.item()
                record = self.records[record_index]

                print()
                print("Embedding rank:", rank)
                print("ID:", record["id"])
                print("Similarity:", float(scores[record_index]))
                print(record["text"])

            candidate_texts = [
                record["text"]
                for record in candidates
            ]

            # reranked = self.reranker.rank(
            #     query,
            #     candidate_texts,
            #     top_k=min(top_k, len(candidates)),
            #     batch_size=4,
            #     prompt=(
            #         "Given a customer equipment problem, retrieve passages "
            #         "containing substantive evidence for a product or service "
            #         "capability that addresses the problem. "
            #         "Merely naming the equipment is insufficient."
            #     ),
            # )

            reranked = self.reranker.rank(
                query,
                candidate_texts,
                top_k=min(top_k, len(candidates)),
                batch_size=4,
            )

            evidence = []

            for result in reranked:
                record = candidates[result["corpus_id"]]
                evidence.append({
                    **record,
                    "score": float(result["score"]),
                })

            return evidence
        finally:
            # Tracebacks must not keep these query-side tensors alive on failure.
            query_embedding = scores = top_values = top_indices = None
            reranked = result = index = None

    def close(self):
        """Drop quantized models rather than moving or recasting them."""
        if self._closed:
            return
        self._closed = True
        self.record_embeddings = None
        self.reranker = None
        self.model = None
        self.records = []
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _print_results(evidence, elapsed):
    print("\nRetrieval time:", round(elapsed, 4), "seconds")

    for rank, record in enumerate(evidence, start=1):
        print()
        print("Rerank position:", rank)
        print("ID:", record["id"])
        print("Source:", record["source"])
        print("Pages:", record["pages"])
        print("Label:", record["label"])
        print("Review status:", record["review_status"])
        print("Reranker score:", round(record["score"], 4))
        print(record["text"])


def retrieve(query, top_k=RESULT_LIMIT, *, verbose=True):
    """Retrieve live passages, save diagnostics, and release models before return.

    Importing this module never loads records/models or changes cache settings.
    ``verbose=False`` retains full file logs while suppressing the stdout dump.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be an integer of at least 1.")

    started_at = datetime.now(timezone.utc)
    output_dir = project_dir / "tmp" / "retrieval_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = started_at.strftime("%Y%m%dT%H%M%S_%fZ")
    run_path = output_dir / f"retrieval_{stamp}.txt"
    latest_path = output_dir / "latest.txt"

    with (
        run_path.open("x", encoding="utf-8") as run_file,
        latest_path.open("w", encoding="utf-8") as latest_file,
    ):
        streams = [run_file, latest_file]
        if verbose:
            streams.insert(0, sys.stdout)
        with redirect_stdout(_Tee(*streams)):
            print("Run status: STARTED")
            print("Started at (UTC):", started_at.isoformat())
            print("Question:", query)
            print("Embedding model:", EMBEDDING_MODEL_NAME)
            print("Reranker model:", RERANKER_MODEL_NAME)
            print("Model cache:", MODEL_DIR)
            print("Candidate limit:", CANDIDATE_LIMIT)
            print("Requested results:", top_k)
            print("Saved run:", run_path)
            print("Latest result:", latest_path, flush=True)

            retriever = None
            try:
                try:
                    folder = project_dir / "data" / "processed" / "records"
                    records = load_processed_records(folder)
                    if not records:
                        raise ValueError("No body passages were loaded.")
                    print("Loaded body passages:", len(records), flush=True)
                    retriever = _ProductRetriever(records)
                    retriever.load()

                    start_time = perf_counter()
                    evidence = retriever.retrieve(query, top_k)
                    elapsed = perf_counter() - start_time
                    _print_results(evidence, elapsed)
                except BaseException as error:
                    traceback.clear_frames(error.__traceback__)
                    raise
                finally:
                    if retriever is not None:
                        retriever.close()
            except Exception:
                print("\nRun status: FAILED", flush=True)
                traceback.print_exc(file=run_file)
                traceback.print_exc(file=latest_file)
                raise

            print("\nRun status: COMPLETED", flush=True)
    return evidence


def run_query(query):
    return retrieve(query, top_k=RESULT_LIMIT)


def main():
    query = input("Enter your question: ")
    run_query(query)


if __name__ == "__main__":
    main()
