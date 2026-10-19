"""Check vision-test data handling using temporary fixtures and a mock backend."""

import base64
import builtins
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import vision_test


class VisionTestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "images" / "flowchart.png"
        self.image.parent.mkdir()
        self.image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAusB9YL7rNwAAAAASUVORK5CYII="
        )
        self.image.write_bytes(self.image_bytes)
        self.records_path = self.root / "records" / "pictures.json"
        self.records_path.parent.mkdir()
        self.record = {
            "id": "example:#/pictures/2",
            "source": "example.pdf",
            "pages": [2],
            "content_type": "picture_candidate",
            "picture_type": "flowchart",
            "image_path": "images/flowchart.png",
            "provenance": [{
                "page_no": 2,
                "bbox": {
                    "l": 10, "t": 100, "r": 200, "b": 20,
                    "coord_origin": "BOTTOMLEFT",
                },
                "charspan": [0, 0],
            }],
            "review_status": "pending",
        }
        self.write_records([self.record])
        self.model = self.root / "model.gguf"
        self.projector = self.root / "projector.gguf"
        self.model.write_bytes(b"mock model")
        self.projector.write_bytes(b"mock projector")
        self.output = self.root / "output" / "result.json"

    def write_records(self, records):
        self.records_path.write_text(json.dumps(records), encoding="utf-8")

    def load_record(self):
        return vision_test.load_picture_record(
            self.records_path, self.record["id"], self.root
        )

    def configured_paths(self):
        return patch.multiple(
            vision_test,
            PROJECT_DIR=self.root,
            RECORDS_PATH=self.records_path,
            PICTURE_ID=self.record["id"],
            MODEL_PATH=self.model,
            PROJECTOR_PATH=self.projector,
            OUTPUT_PATH=self.output,
        )

    @contextlib.contextmanager
    def mock_backend(self, request_error=None):
        response = {
            "choices": [{
                "message": {"role": "assistant", "content": "Mock flowchart description"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 4},
        }
        llm = Mock()
        llm.create_chat_completion.return_value = response
        llm.create_chat_completion.side_effect = request_error
        backend = types.ModuleType("llama_cpp")
        backend.__path__ = []
        backend.__version__ = "mock-version"
        backend.llama_supports_gpu_offload = Mock(return_value=True)
        backend.Llama = Mock(return_value=llm)
        chat_format = types.ModuleType("llama_cpp.llama_chat_format")
        chat_format.MTMDChatHandler = Mock(return_value=Mock())
        with patch.dict(sys.modules, {
            "llama_cpp": backend,
            "llama_cpp.llama_chat_format": chat_format,
        }):
            yield llm, response

    def test_selects_exact_id_and_preserves_provenance(self):
        other = dict(self.record, id="example:#/pictures/0", picture_type="logo")
        self.write_records([other, self.record])
        record, image = self.load_record()
        self.assertEqual(record, self.record)
        self.assertEqual(image, self.image.resolve())

    def test_rejects_missing_or_duplicate_id(self):
        for records in ([], [self.record, self.record]):
            with self.subTest(count=len(records)):
                self.write_records(records)
                with self.assertRaisesRegex(ValueError, "Expected one picture record"):
                    self.load_record()

    def test_rejects_wrong_record_or_picture_type(self):
        for field, value in (
            ("content_type", "body_candidate"),
            ("picture_type", "logo"),
        ):
            with self.subTest(field=field):
                self.write_records([dict(self.record, **{field: value})])
                with self.assertRaises(ValueError):
                    self.load_record()

    def test_rejects_missing_image(self):
        self.image.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load_record()

    def test_rejects_missing_or_inconsistent_provenance(self):
        for change in ({"provenance": []}, {"pages": [3]}):
            with self.subTest(change=change):
                self.write_records([dict(self.record, **change)])
                with self.assertRaises(ValueError):
                    self.load_record()

    def test_check_input_needs_no_backend_or_model_files(self):
        self.model.unlink()
        self.projector.unlink()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "llama_cpp" or name.startswith("llama_cpp."):
                raise AssertionError("Input checking must not import the model backend")
            return real_import(name, *args, **kwargs)

        with self.configured_paths(), patch(
            "builtins.__import__", side_effect=guarded_import
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            vision_test.main(["--check-input"])
        self.assertIn("Input check passed.", stdout.getvalue())
        self.assertFalse(self.output.exists())

    def test_mock_run_saves_source_metadata_and_selected_image(self):
        with self.configured_paths(), self.mock_backend() as (
            llm, response
        ), contextlib.redirect_stdout(io.StringIO()):
            vision_test.main([])

        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["source_record_id"], self.record["id"])
        for field in ("source", "pages", "provenance"):
            self.assertEqual(result[field], self.record[field])
        self.assertEqual(result["review_status"], "pending")
        self.assertEqual(result["image"], str(self.image.resolve()))
        self.assertEqual(result["response"], response)
        request = llm.create_chat_completion.call_args.kwargs
        image_url = request["messages"][1]["content"][0]["image_url"]["url"]
        prefix, encoded = image_url.split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded), self.image_bytes)
        llm.close.assert_called_once()

    def test_closes_model_when_request_fails(self):
        with self.configured_paths(), self.mock_backend(
            RuntimeError("request failed")
        ) as (llm, _), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                vision_test.main([])
        llm.close.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_closes_model_when_output_write_fails(self):
        with self.configured_paths(), self.mock_backend() as (
            llm, _
        ), patch.object(
            Path, "write_text", side_effect=OSError("disk full")
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "disk full"):
                vision_test.main([])
        llm.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
