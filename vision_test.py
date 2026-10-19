"""Read one classified flowchart using prepared picture records."""

import argparse
import base64
import json
from functools import partial
from pathlib import Path
from time import perf_counter

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(r"D:\ml_projects\models\Qwen3.8-27B")
MODEL_PATH = MODEL_DIR / "Qwen3.8-27B-Q4_K_M.gguf"
PROJECTOR_PATH = MODEL_DIR / "mmproj-F16.gguf"
RECORDS_PATH = PROJECT_DIR / "data/processed/records/ac-filter-picture-candidates.json"
PICTURE_ID = "ac-filter:#/pictures/2"
OUTPUT_PATH = PROJECT_DIR / "data/processed/ac-filter-flowchart-vision-test.json"

GPU_LAYERS = 52
CONTEXT_SIZE = 4096


def load_picture_record(records_path, picture_id, project_dir):
    """Select one flowchart and resolve its image against the project folder."""
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Picture records must be a JSON list.")

    matches = [record for record in records if record["id"] == picture_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one picture record: {picture_id}; found {len(matches)}")

    record = matches[0]
    if record.get("content_type") != "picture_candidate":
        raise ValueError(f"Record is not a picture candidate: {picture_id}")
    if record.get("picture_type") != "flowchart":
        raise ValueError(f"Picture is not classified as a flowchart: {picture_id}")

    for field in ("source", "pages", "provenance", "image_path"):
        if not record.get(field):
            raise ValueError(f"Picture record is missing {field}: {picture_id}")
    provenance_pages = {entry["page_no"] for entry in record["provenance"]}
    if set(record["pages"]) != provenance_pages:
        raise ValueError(f"Picture pages do not match provenance: {picture_id}")

    image_path = (project_dir / record["image_path"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    return record, image_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="Validate the selected record and image without loading the model.",
    )
    args = parser.parse_args(argv)
    record, image_path = load_picture_record(RECORDS_PATH, PICTURE_ID, PROJECT_DIR)
    print("Picture:", record["id"], flush=True)
    print("Source:", record["source"], "| Pages:", record["pages"], flush=True)
    print("Image:", image_path, flush=True)
    if args.check_input:
        print("Input check passed.", flush=True)
        return

    for path in (MODEL_PATH, PROJECTOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MTMDChatHandler

    print("llama-cpp-python version:", llama_cpp.__version__, flush=True)
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("This llama-cpp-python build does not support GPU offload.")

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    handler = MTMDChatHandler(clip_model_path=str(PROJECTOR_PATH), use_gpu=True)

    print(f"Loading main model with {GPU_LAYERS} layers on GPU...", flush=True)
    load_start = perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        # v0.3.35 passes template options through the chat handler.
        chat_handler=partial(handler, enable_thinking=False),
        n_gpu_layers=GPU_LAYERS,
        n_ctx=CONTEXT_SIZE,
        n_batch=128,
        n_ubatch=128,
        flash_attn=True,
        verbose=True,
    )
    load_seconds = perf_counter() - load_start

    try:
        print(f"Main model loaded in {load_seconds:.1f} seconds.", flush=True)
        print("Reading flowchart; first request also loads the vision projector...", flush=True)
        request_start = perf_counter()
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract information only from the supplied image. "
                        "Do not invent equipment, labels, numbers, or connections. "
                        "Mark unreadable labels or ambiguous arrow directions as unclear."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "List the equipment nodes first. Trace each connection independently, "
                                "following its arrowhead to determine direction.\n\n"
                                "Return a table:\n"
                                "Source | Destination | Stream label | Uncertainty\n"
                                "Include returns, drains and backwash connections. Do not assume "
                                "crossing lines are connected. Mark ambiguous endpoints as unclear."
                            ),
                        },
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0.2,
            seed=42,
        )
        request_seconds = perf_counter() - request_start
        choice = response["choices"][0]
        print("\nModel output:\n", flush=True)
        print(choice["message"]["content"], flush=True)
        print("\nFinish reason:", choice["finish_reason"], flush=True)
        print(f"First image request: {request_seconds:.1f} seconds.", flush=True)

        result = {
            "source_record_id": record["id"],
            "source": record["source"],
            "pages": record["pages"],
            "provenance": record["provenance"],
            "review_status": "pending",
            "image": str(image_path),
            "model": str(MODEL_PATH),
            "projector": str(PROJECTOR_PATH),
            "llama_cpp_version": llama_cpp.__version__,
            "n_gpu_layers": GPU_LAYERS,
            "n_ctx": CONTEXT_SIZE,
            "main_model_load_seconds": load_seconds,
            "first_image_request_seconds": request_seconds,
            "response": response,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Saved result:", OUTPUT_PATH, flush=True)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
