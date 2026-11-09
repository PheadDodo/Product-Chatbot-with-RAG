"""Small, lazy NumPy cache for corpus vectors, never passage approval metadata."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile


SCHEMA_VERSION = 1
CONFIG_FILES = (
    "config.json", "config_sentence_transformers.json", "modules.json",
    "sentence_bert_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "1_Pooling/config.json", "2_Normalize/config.json",
)


def local_model_identity(model_dir, model_name, revision=None):
    """Read an existing HF snapshot identity/configs without contacting the Hub."""
    repository = Path(model_dir) / ("models--" + model_name.replace("/", "--"))
    if not revision:
        try:
            revision = (repository / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            revision = None
    configs = {}
    if revision:
        snapshot = repository / "snapshots" / revision
        for filename in CONFIG_FILES:
            path = snapshot / filename
            if path.is_file():
                configs[filename] = sha256(path.read_bytes()).hexdigest()
    return {"model": model_name, "revision": revision, "config_sha256": configs}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _validate_array(array, row_count, dimension):
    import numpy as np

    if array.ndim != 2 or array.shape != (row_count, dimension):
        raise ValueError(f"Corpus embeddings have invalid shape: {array.shape}.")
    if array.dtype != np.dtype("float32"):
        raise ValueError(f"Corpus embeddings require float32 dtype, received {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError("Corpus embeddings contain nonfinite values.")


def _atomic_save(path, write):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_or_build(*, cache_dir, records, settings, expected_dimension, encode):
    """Return (float32 vectors, HIT/MISS/REBUILT); rebuild invalid/stale data.

    The identity tracks ordered IDs/text hashes and encoding settings. Metadata
    comes exclusively from current records. NumPy is imported only on this call;
    loading explicitly disallows pickle. Invalid newly encoded vectors fail.
    """
    import numpy as np

    identity = {
        "schema": SCHEMA_VERSION,
        "settings": settings,
        "dimension": expected_dimension,
        "dtype": "float32",
        "records": [
            {"id": record["id"], "text_sha256": sha256(record["text"].encode("utf-8")).hexdigest()}
            for record in records
        ],
    }
    key = sha256(_json_bytes(identity)).hexdigest()
    directory = Path(cache_dir)
    array_path = directory / f"corpus-{key}.npz"
    manifest_path = directory / f"corpus-{key}.json"
    status = "REBUILT" if array_path.exists() or manifest_path.exists() else "MISS"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Corpus cache identity differs.")
        if manifest["array_sha256"] != sha256(array_path.read_bytes()).hexdigest():
            raise ValueError("Corpus cache checksum differs.")
        with array_path.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as saved:
                vectors = saved["embeddings"]
        _validate_array(vectors, len(records), expected_dimension)
        return vectors, "HIT"
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass

    vectors = np.asarray(encode())
    _validate_array(vectors, len(records), expected_dimension)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_save(array_path, lambda stream: np.savez_compressed(stream, embeddings=vectors))
    manifest = {"identity": identity, "array_sha256": sha256(array_path.read_bytes()).hexdigest()}
    _atomic_save(manifest_path, lambda stream: stream.write(_json_bytes(manifest)))
    return vectors, status
