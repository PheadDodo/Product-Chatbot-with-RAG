import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data/processed/docling"
OUTPUT_DIR = PROJECT_DIR / "data/processed/records"
OVERRIDES_PATH = PROJECT_DIR / "data/processed/review-overrides.json"


def get_picture_ref(item, items):
    """Check whether an item's parent chain includes a picture."""
    parent = item.get("parent", {}).get("$ref")
    visited = set()

    while parent:
        if parent in visited:
            raise ValueError(f"Cyclic parent reference: {parent}")

        visited.add(parent)

        if parent.startswith("#/pictures/"):
            return parent

        parent = items.get(parent, {}).get("parent", {}).get("$ref")

    return None


def main():
    overrides = json.loads(
        OVERRIDES_PATH.read_text(encoding="utf-8")
    )

    paths = sorted(INPUT_DIR.glob("**/document.json"))
    if not paths:
        raise FileNotFoundError(f"No document.json files found in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_records = 0
    for index, path in enumerate(paths, start=1):
        document = json.loads(path.read_text(encoding="utf-8"))
        # refereces are local to each brochure
        items = {
            item["self_ref"]: item
            for category in ("texts", "groups", "pictures", "tables")
            for item in document.get(category, [])            
        }
        body_candidates = []
        picture_ocr = []
        for item in document["texts"]:
            record_id = f"{document['name']}:{item['self_ref']}"
            action = overrides.get(record_id, {}).get("action")

            if action not in (None, "exclude", "figure_text"):
                raise ValueError(
                    f"Unknown review action {action!r}: {record_id}"
                )

            if action == "exclude":
                continue

            if action == "figure_text" or get_picture_ref(item, items):
                picture_ocr.append(item)
            elif item.get("content_layer") == "body":
                body_candidates.append(item)
        records = []
        for item in body_candidates:
            provenance = item["prov"]
            if not provenance:
                raise ValueError(f"Item has no provenance: {item}")
            records.append({
                    "id": f"{document['name']}:{item['self_ref']}",
                    "source": document["origin"]["filename"],
                    "pages": sorted({
                        entry["page_no"] for entry in provenance
                    }),
                    "content_type": "body_candidate",
                    "label": item["label"],
                    "text": item["text"],
                    "provenance": provenance,
                    "review_status": "pending",            
            })
        output_path = (OUTPUT_DIR / f"{document['name']}-body-candidates.json")
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        figure_records = []

        for item in picture_ocr:
            record_id = f"{document['name']}:{item['self_ref']}"
            picture_ref = (
                overrides.get(record_id, {}).get("picture_ref")
                or get_picture_ref(item, items)
            )

            if (
                picture_ref not in items
                or not picture_ref.startswith("#/pictures/")
            ):
                raise ValueError(
                    f"Missing or invalid picture reference: {record_id}"
                )
            provenance = item["prov"]
            if not provenance:
                raise ValueError(f"Item has no provenance: {item}")

            figure_records.append({
                "id": f"{document['name']}:{item['self_ref']}",
                "source": document["origin"]["filename"],
                "pages": sorted({
                    entry["page_no"] for entry in provenance
                }),
                "content_type": "figure_text_candidate",
                "label": item["label"],
                "text": item["text"],
                "parent_ref": item.get("parent", {}).get("$ref"),
                "picture_ref": picture_ref,
                "provenance": provenance,
                "review_status": "pending",
            })

        figure_path = (
            OUTPUT_DIR / f"{document['name']}-figure-text-candidates.json"
        )
        figure_path.write_text(
            json.dumps(figure_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        picture_records = []

        for picture in document.get("pictures", []):
            picture_ref = picture["self_ref"]
            provenance = picture["prov"]

            if not provenance:
                raise ValueError(f"Picture has no provenance: {picture_ref}")

            image_file = (
                path.parent
                / Path(picture["image"]["uri"].replace("\\", "/"))
            ).resolve()

            if not image_file.is_file():
                raise FileNotFoundError(image_file)

            linked_text = [
                record for record in figure_records
                if record["picture_ref"] == picture_ref
            ]

            picture_records.append({
                "id": f"{document['name']}:{picture_ref}",
                "source": document["origin"]["filename"],
                "pages": sorted({p["page_no"] for p in provenance}),
                "content_type": "picture_candidate",
                "picture_ref": picture_ref,
                "image_path": image_file.relative_to(PROJECT_DIR).as_posix(),
                "figure_text_ids": [r["id"] for r in linked_text],
                "extracted_text": "\n".join(r["text"] for r in linked_text),
                "provenance": provenance,
                "picture_type": overrides.get(
                    f"{document['name']}:{picture_ref}", {}
                ).get("picture_type", "unclassified"),
                "review_status": "pending",
            })

        picture_path = (
            OUTPUT_DIR / f"{document['name']}-picture-candidates.json"
        )
        picture_path.write_text(
            json.dumps(picture_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        other_count = (
            len(document["texts"])
            - len(body_candidates)
            - len(picture_ocr)        
        )
        total_records += len(records)
        print(
            f"[{index}/{len(paths)}] {document['name']}: "
            f"{len(records)} body candidates, "
            f"{len(picture_ocr)} picture OCR, "
            f"{other_count} other"
        )

    print("\nBrochures processed:", len(paths))
    print("Total saved body records:", total_records)
    print("Output folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()