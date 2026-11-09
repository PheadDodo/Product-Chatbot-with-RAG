import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class ImportSafetyTests(unittest.TestCase):
    def test_cache_import_does_not_import_numeric_or_model_backends(self):
        script = """
import os
import sys
for name in ('numpy', 'torch', 'sentence_transformers', 'transformers'):
    sys.modules[name] = None
environment = dict(os.environ)
import embedding_cache
assert dict(os.environ) == environment
"""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", script],
            cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_NUMPY, "Existing interpreter has no NumPy; use greencv_llm for codec tests.")
class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import embedding_cache

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.cache = embedding_cache
        self.np = np
        self.records = [
            {"id": "a:1", "text": "First passage", "review_status": "pending"},
            {"id": "a:2", "text": "Second passage", "review_status": "pending"},
        ]
        self.settings = {
            "model": "Qwen/test", "revision": "abc123", "normalize_embeddings": True,
            "prompt": None, "quantization": {"load_in_4bit": True},
        }
        self.vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.encode = Mock(side_effect=lambda: self.vectors.copy())

    def load(self, **changes):
        arguments = {
            "cache_dir": self.directory, "records": self.records,
            "settings": self.settings, "expected_dimension": 3, "encode": self.encode,
        }
        arguments.update(changes)
        return self.cache.load_or_build(**arguments)

    def test_second_call_reuses_real_npz_without_reembedding(self):
        original, first_status = self.load()
        restored, second_status = self.load()
        self.np.testing.assert_array_equal(restored, original)
        self.assertEqual((first_status, second_status), ("MISS", "HIT"))
        self.assertEqual(self.encode.call_count, 1)
        manifest = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertNotIn("review_status", json.dumps(manifest))
        self.assertNotIn("First passage", json.dumps(manifest))

    def test_ids_order_text_and_encoding_settings_invalidate(self):
        self.load()
        alternatives = [
            {"records": list(reversed(self.records))},
            {"records": [{**self.records[0], "id": "changed"}, self.records[1]]},
            {"records": [{**self.records[0], "text": "Changed"}, self.records[1]]},
            {"settings": {**self.settings, "revision": "def456"}},
            {"settings": {**self.settings, "normalize_embeddings": False}},
            {"settings": {**self.settings, "prompt": "Document:"}},
            {"settings": {**self.settings, "quantization": {"load_in_4bit": False}}},
        ]
        for changes in alternatives:
            with self.subTest(changes=changes):
                _, status = self.load(**changes)
                self.assertEqual(status, "MISS")
        self.assertEqual(self.encode.call_count, 1 + len(alternatives))

    def test_metadata_changes_do_not_override_current_records_or_rebuild(self):
        self.load()
        self.records[0]["review_status"] = "reviewed"
        self.records[0]["pages"] = [4]
        _, status = self.load()
        self.assertEqual(status, "HIT")
        self.assertEqual(self.records[0]["pages"], [4])
        self.assertEqual(self.encode.call_count, 1)

    def test_invalid_json_or_npz_is_rebuilt(self):
        for suffix, data in ((".json", b"broken"), (".npz", b"broken"), (".npz", b"PK\x03\x04broken")):
            with self.subTest(suffix=suffix, data=data):
                self.load()
                path = next(self.directory.glob("*" + suffix))
                path.write_bytes(data)
                if data.startswith(b"PK"):
                    from hashlib import sha256

                    manifest_path = next(self.directory.glob("*.json"))
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["array_sha256"] = sha256(data).hexdigest()
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                _, status = self.load()
                self.assertEqual(status, "REBUILT")

    def test_wrong_dimensions_dtype_and_nonfinite_cached_vectors_are_rebuilt(self):
        # Write a coherent checksum so validation must inspect actual array data.
        from hashlib import sha256

        alternatives = [
            self.np.zeros((2, 2), dtype=self.np.float32),
            self.vectors.astype(self.np.float64),
            self.np.array([[float("nan"), 0, 0], [0, 1, 0]], dtype=self.np.float32),
        ]
        for invalid in alternatives:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                self.load()
                array_path = next(self.directory.glob("*.npz"))
                with array_path.open("wb") as stream:
                    self.np.savez_compressed(stream, embeddings=invalid)
                manifest_path = next(self.directory.glob("*.json"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["array_sha256"] = sha256(array_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                _, status = self.load()
                self.assertEqual(status, "REBUILT")

    def test_invalid_new_vectors_fail_without_saving_a_cache(self):
        self.encode.side_effect = lambda: self.np.zeros((1, 3), dtype=self.np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_snapshot_revision_and_small_config_changes_are_detected_offline(self):
        repository = self.directory / "models--Qwen--test"
        (repository / "refs").mkdir(parents=True)
        (repository / "refs" / "main").write_text("abc123", encoding="utf-8")
        snapshot = repository / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        config = snapshot / "config.json"
        config.write_text('{"hidden_size":3}', encoding="utf-8")
        first = self.cache.local_model_identity(self.directory, "Qwen/test")
        self.assertEqual(first["revision"], "abc123")
        config.write_text('{"hidden_size":4}', encoding="utf-8")
        second = self.cache.local_model_identity(self.directory, "Qwen/test")
        self.assertNotEqual(first, second)
        self.assertEqual(
            self.cache.local_model_identity(self.directory, "Qwen/missing")["revision"], None,
        )

    def test_retriever_load_reuses_corpus_vectors_but_keeps_current_metadata(self):
        import product_rag

        owner = self

        class EmbeddingStub:
            max_seq_length = 2048
            truncate_dim = None
            default_prompt_name = None
            prompts = {"query": "Search:", "document": ""}

            def __init__(self, *args, **kwargs):
                pass

            def __getitem__(self, index):
                return types.SimpleNamespace(auto_model=types.SimpleNamespace(
                    config=types.SimpleNamespace(_commit_hash="abc123"),
                ))

            def get_sentence_embedding_dimension(self):
                return 3

            def encode(self, texts, **kwargs):
                owner.assertEqual(texts, [record["text"] for record in owner.records])
                owner.encode()
                return owner.vectors.copy()

        torch = types.SimpleNamespace(
            bfloat16="bfloat16",
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
        )
        modules = {
            "torch": torch,
            "sentence_transformers": types.SimpleNamespace(
                SentenceTransformer=EmbeddingStub, CrossEncoder=Mock(),
            ),
            "transformers": types.SimpleNamespace(BitsAndBytesConfig=Mock()),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(product_rag, "_configure_model_cache"),
            patch.object(product_rag, "MODEL_DIR", self.directory / "models"),
            patch.object(product_rag, "project_dir", self.directory),
        ):
            for index in range(2):
                retriever = product_rag._ProductRetriever(self.records)
                try:
                    retriever.load()
                    self.np.testing.assert_array_equal(retriever.record_embeddings, self.vectors)
                    self.assertIs(retriever.records, self.records)
                finally:
                    retriever.close()
                if index == 0:
                    self.records[0]["pages"] = [7]
        self.assertEqual(self.encode.call_count, 1)


if __name__ == "__main__":
    unittest.main()
