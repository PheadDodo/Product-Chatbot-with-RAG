import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "service-records.json"
OUTPUT_PATH = PROJECT_DIR / "tmp" / "service-candidates.json"


def find_service_candidates(
    records,
    equipment_term="PF-DS",
    work_terms=("cloth", "cloths"),
    country_term="Chile",
):
    """Return lexical candidates with original records and matching evidence."""
    def term_pattern(term):
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Search terms must be nonempty strings.")
        return re.compile(
            rf"(?<!\w){re.escape(term.strip())}(?!\w)",
            re.IGNORECASE,
        )

    equipment_pattern = term_pattern(equipment_term)

    if isinstance(work_terms, (str, bytes)):
        raise ValueError("work_terms must be a collection of strings.")
    try:
        work_terms = tuple(work_terms)
    except TypeError as error:
        raise ValueError("work_terms must be a collection of strings.") from error
    if not work_terms:
        raise ValueError("Provide at least one work term.")

    work_patterns = [term_pattern(term) for term in work_terms]
    country_pattern = (
        term_pattern(country_term) if country_term is not None else None
    )
    candidates = []

    for record in records:
        equipment = record["reported_equipment"]
        matched_fields = [
            name
            for name in ("description", "model")
            if equipment[name]["status"] == "extracted"
            and equipment_pattern.search(equipment[name]["value"])
        ]
        if not matched_fields:
            continue

        work = record["reported_work"]
        if work["status"] != "captured":
            continue

        matched_passages = [
            passage
            for passage in work["passages"]
            if any(pattern.search(passage["text"]) for pattern in work_patterns)
        ]
        if not matched_passages:
            continue

        location = record["reported_location"]
        country_mentioned = None
        if country_pattern is not None and location["status"] == "extracted":
            country_mentioned = bool(country_pattern.search(location["value"]))

        candidates.append({
            "record": record,
            "matched_equipment_fields": matched_fields,
            "matched_work_passages": matched_passages,
            "requested_country_mentioned": country_mentioned,
        })

    return sorted(candidates, key=lambda item: item["record"]["id"])


def format_service_context(candidates):
    """Render selected historical records with their original source evidence."""
    lines = [
        "Historical service evidence",
        "Extraction and arithmetic statuses apply to their captured fields only.",
        "Country flags indicate literal mentions; absent tax evidence is unspecified.",
    ]

    def details(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def add_evidence(source, evidence):
        items = evidence if isinstance(evidence, list) else [evidence]
        for item in items:
            if "source_lines" in item:
                lines.append("Evidence match: " + details({
                    key: value for key, value in item.items()
                    if key != "source_lines"
                }))
                spans = item["source_lines"]
            else:
                spans = [item]
            for span in spans:
                lines.extend([
                    f"[{source}, lines {span['line_start']}-{span['line_end']}]",
                    span["quote"],
                ])

    def add_field(source, title, field):
        lines.append(title + ": " + details({
            key: value for key, value in field.items()
            if key not in ("evidence", "tax_evidence")
        }))
        add_evidence(source, field["evidence"])

    if not candidates:
        lines.append("No historical candidates matched.")

    for candidate in candidates:
        record = candidate["record"]
        source = record["source"]
        lines.extend([
            "",
            "=" * 72,
            f"ID: {record['id']}",
            f"Source: {source}",
            "Matched equipment fields: "
            + details(candidate["matched_equipment_fields"]),
            "Requested country term mentioned: "
            + details(candidate["requested_country_mentioned"]),
        ])

        add_field(source, "Location", record["reported_location"])
        for name, field in record["reported_equipment"].items():
            add_field(source, f"Equipment {name}", field)
        add_field(source, "Reported on-site hours", record["reported_onsite_hours"])
        add_field(source, "Reported travel hours", record["reported_travel_hours"])

        work = record["reported_work"]
        lines.append(f"Work capture status: {work['status']}")
        for passage in work["passages"]:
            lines.append(f"Work section: {passage['section']}")
            add_evidence(source, passage["evidence"])

        cost = record["reported_total_cost"]
        add_field(source, "Reported final total", cost)
        lines.append("Tax/subtotal evidence:")
        if cost["tax_evidence"]:
            add_evidence(source, cost["tax_evidence"])
        else:
            lines.append("Not captured; tax basis remains unspecified.")

        tables = record["reported_cost_tables"]
        lines.append(f"Monetary tables captured: {len(tables)}")
        for table in tables:
            lines.append("Separate cost table: " + details({
                key: value for key, value in table.items()
                if key not in ("header_evidence", "rows")
            }))
            add_evidence(source, table["header_evidence"])
            for row in table["rows"]:
                add_evidence(source, row["evidence"])
                lines.append("Row fields and checks: " + details({
                    key: value for key, value in row.items()
                    if key not in ("cells", "raw_cells", "evidence")
                }))

        lines.append("Report-total check: " + details(record["report_total_check"]))

    return "\n".join(lines) + "\n"


def main():
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    search = {
        "equipment_term": "PF-DS",
        "work_terms": ["cloth", "cloths"],
        "country_term": "Chile",
    }
    candidates = find_service_candidates(records, **search)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {"search": search, "candidates": candidates},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"Candidates: {len(candidates)}")
    for candidate in candidates:
        record = candidate["record"]
        location = record["reported_location"]["value"] or "location unknown"
        cost = record["reported_total_cost"]
        total = (
            f"{cost['currency']} {cost['amount']}"
            if cost["status"] == "extracted"
            else f"unknown ({cost['status']})"
        )
        print(f"{record['id']} | {location} | reported total: {total}")

    print(f"Saved candidate evidence: {OUTPUT_PATH}")

    context_path = OUTPUT_PATH.with_name("service-context.txt")
    context_path.write_text(
        format_service_context(candidates),
        encoding="utf-8",
    )
    print("Saved service context:", context_path)


if __name__ == "__main__":
    main()
