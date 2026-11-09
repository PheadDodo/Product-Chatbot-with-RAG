import json
import re

from load_service_reports import PROJECT_DIR, load_service_reports


OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "service-hours.json"
NUMBER = r"(\d+(?:\.\d+)?)"
UNIT = r"\s*(?:hours?|hrs?|h)?"

PATTERNS = {
    "reported_onsite_hours": re.compile(
        r"(?:total\s+)?"
        r"(?:on[- ]site(?:\s+(?:hours|work))?|hours\s+on[- ]site)"
        rf"\s*:\s*{NUMBER}{UNIT}",
        re.IGNORECASE,
    ),
    "reported_travel_hours": re.compile(
        r"(?:total\s+)?travel(?:\s+(?:hours|time))?"
        r"(?:\s*\(round trip\))?"
        rf"\s*:\s*{NUMBER}{UNIT}",
        re.IGNORECASE,
    ),
}


def normalize_line(text):
    text = text.replace("**", "").strip()

    if text.startswith("|"):
        cells = [cell.strip() for cell in text.split("|") if cell.strip()]
        if len(cells) != 2:
            return None
        return f"{cells[0].rstrip(':')}: {cells[1]}"

    return text.removeprefix("- ").strip()


def extract_report_hours(record):
    result = {
        "id": record["id"],
        "source": record["source"],
        "source_path": record["source_path"],
    }

    for field, pattern in PATTERNS.items():
        evidence = []

        for line in record["lines"]:
            normalized = normalize_line(line["text"])
            match = pattern.fullmatch(normalized) if normalized else None

            if match:
                evidence.append({
                    "value": float(match.group(1)),
                    "line_start": line["line_number"],
                    "line_end": line["line_number"],
                    "quote": line["text"],
                })

        values = {item["value"] for item in evidence}
        status = (
            "extracted" if len(values) == 1
            else "conflicting" if values
            else "not_extracted"
        )

        result[field] = {
            "value": next(iter(values)) if len(values) == 1 else None,
            "unit": "hours",
            "status": status,
            "evidence": evidence,
        }

    return result


def main():
    records = [extract_report_hours(record) for record in load_service_reports()]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Reports processed: {len(records)}")
    for field in PATTERNS:
        count = sum(record[field]["status"] == "extracted" for record in records)
        print(f"{field}: {count}/{len(records)} extracted")

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()