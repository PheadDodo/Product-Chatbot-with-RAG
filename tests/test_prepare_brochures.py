"""Check passage boundaries and source traceability without extraction or models."""

import copy
import json
import unittest
from pathlib import Path

from prepare_brochures import build_passages


PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_item(index, text, label="text", pages=(1,), **fields):
    return {
        "self_ref": f"#/texts/{index}",
        "label": label,
        "text": text,
        "content_layer": "body",
        "prov": [{"page_no": page, "charspan": [0, len(text)]} for page in pages],
        **fields,
    }


def make_record(item, status="pending"):
    return {
        "id": f"example:{item['self_ref']}",
        "source": "example.pdf",
        "pages": sorted({entry["page_no"] for entry in item["prov"]}),
        "content_type": "body_candidate",
        "label": item["label"],
        "text": item["text"],
        "provenance": copy.deepcopy(item["prov"]),
        "review_status": status,
    }


def refs(*items):
    return [{"$ref": item["self_ref"]} for item in items]


def make_document(texts, children=None, **categories):
    return {
        "name": "example",
        "origin": {"filename": "example.pdf"},
        "body": {"self_ref": "#/body", "children": refs(*texts) if children is None else children},
        "texts": texts,
        "groups": [],
        "pictures": [],
        "tables": [],
        **categories,
    }


class BuildPassagesTests(unittest.TestCase):
    def test_heading_keeps_its_paragraphs_until_next_section(self):
        items = [
            make_item(9, "Proved by design", "section_header"),
            make_item(10, "The AC filter is an ideal choice."),
            make_item(11, "AC filter is designed for smaller side streams."),
            make_item(12, "Led by experience", "section_header"),
            make_item(13, "The supplier has a comprehensive filtration portfolio."),
        ]
        records = [make_record(item) for item in items]

        passages = build_passages(make_document(items), records)

        self.assertEqual([passage["section"] for passage in passages], [
            "Proved by design", "Led by experience",
        ])
        self.assertEqual(passages[0]["member_ids"], [record["id"] for record in records[:3]])
        self.assertEqual(passages[1]["member_ids"], [record["id"] for record in records[3:]])
        self.assertEqual(passages[0]["text"], (
            "## Proved by design\n\nThe AC filter is an ideal choice.\n\n"
            "AC filter is designed for smaller side streams."
        ))
        self.assertNotIn("Led by experience", passages[0]["text"])

    def test_list_items_stay_with_heading_and_retain_markers(self):
        items = [
            make_item(0, "Benefits", "section_header"),
            make_item(1, "Low energy consumption", "list_item", marker="·"),
            make_item(2, "Consistent residual cake moisture", "list_item"),
        ]
        passages = build_passages(make_document(items), [make_record(item) for item in items])

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["text"], (
            "## Benefits\n\n· Low energy consumption\n\n- Consistent residual cake moisture"
        ))
        self.assertEqual(len(passages[0]["member_ids"]), 3)

    def test_preamble_survives_but_heading_only_sections_are_omitted(self):
        items = [
            make_item(0, "An introductory product description."),
            make_item(1, "Cover title", "title"),
            make_item(2, "Useful section", "section_header"),
            make_item(3, "The useful explanation."),
            make_item(4, "Trailing heading", "section_header"),
        ]
        passages = build_passages(make_document(items), [make_record(item) for item in items])

        self.assertEqual([passage["section"] for passage in passages], [None, "Useful section"])
        self.assertEqual(passages[0]["text"], "An introductory product description.")
        self.assertEqual(passages[1]["member_ids"], ["example:#/texts/2", "example:#/texts/3"])
        self.assertNotIn("Cover title", passages[1]["text"])

    def test_tree_order_includes_nested_children_and_ignores_unselected_text(self):
        heading = make_item(0, "Technical details", "section_header")
        paragraph = make_item(1, "Selected paragraph.")
        bullet = make_item(2, "Selected bullet.", "list_item", marker="1.")
        footnote = make_item(3, "Selected table footnote.", "footnote")
        caption = make_item(4, "Selected picture caption.", "caption")
        excluded = make_item(5, "Excluded private annotation.")
        figure_ocr = make_item(6, "Unselected diagram OCR.")
        group = {"self_ref": "#/groups/0", "children": refs(bullet)}
        table = {"self_ref": "#/tables/0", "children": refs(footnote)}
        picture = {"self_ref": "#/pictures/0", "children": refs(caption, figure_ocr)}
        children = refs(heading, paragraph, excluded, group, table, picture)
        texts = [caption, excluded, footnote, bullet, paragraph, figure_ocr, heading]
        document = make_document(texts, children, groups=[group], tables=[table], pictures=[picture])
        selected = [make_record(item) for item in [caption, bullet, paragraph, heading, footnote]]

        passages = build_passages(document, selected)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["member_ids"], [
            "example:#/texts/0", "example:#/texts/1", "example:#/texts/2",
            "example:#/texts/3", "example:#/texts/4",
        ])
        self.assertNotIn("Excluded private annotation", passages[0]["text"])
        self.assertNotIn("Unselected diagram OCR", passages[0]["text"])

    def test_cross_page_passage_preserves_members_and_never_inherits_approval(self):
        items = [
            make_item(0, "Operating conditions", "section_header", pages=(2,)),
            make_item(1, "A paragraph spanning two pages.", pages=(2, 3)),
            make_item(2, "The explanation continues.", pages=(4,)),
        ]
        document = make_document(items)
        records = [make_record(item, status="reviewed") for item in items]
        before_document = copy.deepcopy(document)
        before_records = copy.deepcopy(records)

        passages = build_passages(document, records)

        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage["source"], "example.pdf")
        self.assertEqual(passage["pages"], [2, 3, 4])
        self.assertEqual(passage["content_type"], "body_passage")
        self.assertEqual(passage["label"], "passage")
        self.assertEqual(passage["review_status"], "pending")
        self.assertEqual(passage["members"], records)
        self.assertEqual(passage["provenance"], [entry for record in records for entry in record["provenance"]])
        self.assertEqual(records, before_records)
        self.assertEqual(document, before_document)
        self.assertEqual(passage["id"], build_passages(document, records)[0]["id"])

    def test_empty_selection_returns_no_passages(self):
        item = make_item(0, "An excluded paragraph.")
        self.assertEqual(build_passages(make_document([item]), []), [])

    def test_selected_record_missing_from_body_tree_fails_visibly(self):
        item = make_item(0, "Unreachable selected text.")
        with self.assertRaises(ValueError):
            build_passages(make_document([item], children=[]), [make_record(item)])

    def test_cyclic_child_tree_fails_visibly(self):
        item = make_item(0, "Selected text.")
        group = {"self_ref": "#/groups/0", "children": [{"$ref": "#/groups/0"}]}
        document = make_document([item], children=refs(item, group), groups=[group])
        with self.assertRaises(ValueError):
            build_passages(document, [make_record(item)])


class SavedBrochurePassageTests(unittest.TestCase):
    def load_saved(self, name):
        document_path = PROJECT_DIR / "data" / "processed" / "docling" / name / "document.json"
        records_path = PROJECT_DIR / "data" / "processed" / "records" / f"{name}-body-candidates.json"
        if not document_path.is_file() or not records_path.is_file():
            self.skipTest("Private saved brochure fixtures are unavailable")
        return (
            json.loads(document_path.read_text(encoding="utf-8")),
            json.loads(records_path.read_text(encoding="utf-8")),
        )

    def test_ac_filter_design_is_one_complete_section(self):
        document, records = self.load_saved("ac-filter")
        passages = build_passages(document, records)
        passage = next(p for p in passages if p["section"] == "Proved by design")
        self.assertEqual(passage["member_ids"], [
            "ac-filter:#/texts/9", "ac-filter:#/texts/10", "ac-filter:#/texts/11",
        ])
        for record in records:
            if record["id"] in passage["member_ids"]:
                self.assertIn(record["text"], passage["text"])
        self.assertNotIn("Led by experience", passage["text"])

    def test_filter_optimizer_benefits_form_one_passage(self):
        document, records = self.load_saved("filter-optimizer")
        passages = build_passages(document, records)
        passage = next(p for p in passages if p["section"] == "Benefits")
        self.assertEqual(passage["member_ids"], [
            f"filter-optimizer:#/texts/{index}" for index in range(11, 17)
        ])
        self.assertEqual(passage["pages"], [2])
        self.assertEqual(passage["review_status"], "pending")
        self.assertEqual(len(passage["members"]), 6)


if __name__ == "__main__":
    unittest.main()
