import json
from pathlib import Path

def format_reviewed_record(record):
    if record.get("review_status") != "reviewed":
        raise ValueError("Only reviewed records can become evidence.")

    pages = ", ".join(str(page) for page in record["pages"])

    parts = [
        f"Evidence ID: {record["id"]}",
        f"Source: {record["source"]} | Pages: {pages}",
    ]

    if record.get("review_scope"):
        parts.append(f"Review scope: {record["review_scope"]}")

    parts.append("Reviewed description:\n" + record["text"])

    uncertainties = record.get("uncertainties", [])
    if uncertainties:
        parts.append(
            "Uncertainties:\n"
            + "\n".join(f"- {note}" for note in uncertainties)            
        )

    return "\n\n".join(parts)

def main():
    project_dir = Path(__file__).resolve().parent
    record_path = (
        project_dir
        / "data/processed/reviewed/ac-filter-flowchart.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    print(format_reviewed_record(record))

if __name__ == "__main__":
    main()