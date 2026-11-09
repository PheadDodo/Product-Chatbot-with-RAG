"""Check citation-to-member attribution without loading any model backend."""

import copy
import json
from pathlib import Path
import unittest

from load_llm import format_passages


PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_passage():
    return {
        "id": "example:passage:#/texts/1",
        "source": "example.pdf",
        "pages": [1, 2],
        "label": "passage",
        "review_status": "pending",
        "score": 0.75,
        "text": "## Filter optimizer\n\nAdjusts cycle setpoints.",
        "member_ids": ["example:#/texts/1", "example:#/texts/2"],
        "members": [
            {
                "id": "example:#/texts/1", "source": "example.pdf",
                "pages": [1], "label": "section_header",
                "text": "Filter optimizer", "review_status": "pending",
            },
            {
                "id": "example:#/texts/2", "source": "example.pdf",
                "pages": [2], "label": "text",
                "text": "Adjusts cycle setpoints.", "review_status": "pending",
            },
        ],
    }


class PassageCitationFormattingTests(unittest.TestCase):
    def test_title_and_prose_are_attributed_to_their_own_pages(self):
        passage = make_passage()

        context = format_passages([passage])

        self.assertIn("Source: example.pdf\nPages: [1, 2]", context)
        title = (
            "Member ID: example:#/texts/1\nMember pages: [1]\n"
            "Member label: section_header\n## Filter optimizer"
        )
        prose = (
            "Member ID: example:#/texts/2\nMember pages: [2]\n"
            "Member label: text\nAdjusts cycle setpoints."
        )
        self.assertIn(title, context)
        self.assertIn(prose, context)
        self.assertLess(context.index(title), context.index(prose))
        self.assertEqual(context.count("## Filter optimizer"), 1)
        self.assertEqual(context.count("Adjusts cycle setpoints."), 1)

    def test_preserves_rendered_markers_order_and_all_member_pages(self):
        passage = make_passage()
        passage["pages"] = [1, 2, 3]
        passage["members"][1]["pages"] = [2, 3]
        prose = "Adjusts cycle setpoints.\n\nThe same member continues."
        passage["members"][1]["text"] = prose
        passage["text"] = "## Filter optimizer\n\n" + prose
        passage["text"] += "\n\n1. Numbered benefit\n\n\u00b7 Bulleted benefit"
        for index, text in enumerate(("Numbered benefit", "Bulleted benefit"), 3):
            passage["member_ids"].append(f"example:#/texts/{index}")
            passage["members"].append({
                "id": f"example:#/texts/{index}", "source": "example.pdf",
                "pages": [3], "label": "list_item", "text": text,
                "review_status": "pending",
            })

        context = format_passages([passage])

        self.assertIn("Member pages: [2, 3]\nMember label: text\n", context)
        expected_texts = (
            "## Filter optimizer", prose,
            "1. Numbered benefit", "\u00b7 Bulleted benefit",
        )
        positions = [context.index(text) for text in expected_texts]
        self.assertEqual(positions, sorted(positions))
        for text in expected_texts:
            self.assertEqual(context.count(text), 1)

    def test_parent_pending_status_and_source_records_remain_unchanged(self):
        passage = make_passage()
        passage["members"][0]["review_status"] = "reviewed"
        original = copy.deepcopy(passage)

        context = format_passages([passage])

        self.assertIn("Review status: pending\n", context)
        self.assertEqual(passage, original)

    def test_incomplete_members_fall_back_to_complete_aggregate_text(self):
        original = make_passage()
        cases = {}
        for name, members in (("absent", None), ("empty", [])):
            record = copy.deepcopy(original)
            if members is None:
                record.pop("members")
            else:
                record["members"] = members
            cases[name] = record
        record = copy.deepcopy(original)
        del record["members"][1]["text"]
        cases["missing text"] = record
        record = copy.deepcopy(original)
        record["members"][1]["pages"] = []
        cases["missing pages"] = record
        record = copy.deepcopy(original)
        record["members"][1]["text"] = "Does not match the supplied passage."
        cases["text mismatch"] = record
        record = copy.deepcopy(original)
        record["members"][1].update(label="list_item", text="setpoints.")
        cases["omitted prose is not a list marker"] = record
        record = copy.deepcopy(original)
        record["member_ids"] = None
        cases["invalid member ID list"] = record
        record = copy.deepcopy(original)
        record["members"][1]["pages"] = [3]
        cases["member page outside passage"] = record

        for name, passage in cases.items():
            with self.subTest(name=name):
                before = copy.deepcopy(passage)
                context = format_passages([passage])
                self.assertIn("Pages: [1, 2]\n", context)
                self.assertTrue(context.endswith(original["text"]))
                self.assertEqual(context.count(original["text"]), 1)
                self.assertNotIn("Member pages:", context)
                self.assertEqual(passage, before)

    def test_saved_optimizer_intro_separates_cover_and_mechanism(self):
        path = PROJECT_DIR / "data/processed/records/filter-optimizer-body-passages.json"
        if not path.is_file():
            self.skipTest("Private saved brochure fixture is unavailable")
        passage = next(
            record for record in json.loads(path.read_text(encoding="utf-8"))
            if record["id"] == "filter-optimizer:passage:#/texts/1"
        )
        passage["score"] = 0.75
        original = copy.deepcopy(passage)

        context = format_passages([passage])

        self.assertIn(
            "Member ID: filter-optimizer:#/texts/1\nMember pages: [1]\n"
            "Member label: section_header\n## Filter optimizer", context,
        )
        self.assertIn(
            "Member ID: filter-optimizer:#/texts/2\nMember pages: [2]\n"
            "Member label: text\n" + passage["members"][1]["text"], context,
        )
        self.assertEqual(context.count("## Filter optimizer"), 1)
        self.assertEqual(context.count(passage["members"][1]["text"]), 1)
        self.assertEqual(passage, original)


if __name__ == "__main__":
    unittest.main()
