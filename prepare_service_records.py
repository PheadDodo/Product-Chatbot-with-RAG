import json
import re
from decimal import Decimal

from extract_service_hours import extract_report_hours
from load_service_reports import PROJECT_DIR, load_service_reports


OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "service-records.json"

CODES = r"USD|EUR|AUD|CAD|SEK|ZAR"
AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"

TOTAL_LABEL = re.compile(
    r"\b(?:grand\s+total|cost\s+total|total\s+(?:due|(?:service\s+)?cost))\b",
    re.IGNORECASE,
)

MONEY = re.compile(
    rf"(?:(?:{CODES})\s*)?[$\u20ac]?\s*(?P<amount>{AMOUNT})"
    rf"(?:\s*(?:{CODES}))?\s*(?:\([^)]*\))?\.?",
    re.IGNORECASE,
)


def source_line(line):
    return {
        "line_start": line["line_number"],
        "line_end": line["line_number"],
        "quote": line["text"],
    }


def currency_codes(text):
    codes = set(re.findall(rf"\b(?:{CODES})\b", text.upper()))

    if "\u20ac" in text:
        codes.add("EUR")

    return codes


def extract_report_cost(record):
    matches = []
    tax_evidence = []
    table_header = None

    for line in record["lines"]:
        text = line["text"].replace("**", "").strip()

        if re.search(r"\b(?:VAT|tax)\b", text, re.IGNORECASE) or re.match(
            r"\|\s*subtotal\s*\|", text, re.IGNORECASE
        ):
            tax_evidence.append(source_line(line))

        if text.startswith("|"):
            cells = [cell.strip() for cell in text.strip("|").split("|")]

            if (
                re.search(
                    r"\b(?:total|subtotal|amount)\b",
                    cells[-1],
                    re.IGNORECASE,
                )
                and currency_codes(cells[-1])
            ):
                table_header = line

            if (
                len(cells) < 2
                or cells[0].lower() not in {"total", "total amount due"}
                or any(cells[1:-1])
            ):
                continue

            payload = cells[-1]
            context = [table_header] if table_header else []

        else:
            table_header = None
            label = TOTAL_LABEL.search(text)

            if not label:
                continue

            tail = text[label.end():]
            separator = re.search(r":|\bis\b", tail, re.IGNORECASE)

            if not separator:
                continue

            payload = tail[separator.end():].strip()
            context = []

        money = MONEY.fullmatch(payload)

        if not money:
            continue

        evidence_lines = context + [line]
        codes = set()

        for item in evidence_lines:
            codes.update(currency_codes(item["text"]))

        matches.append({
            "amount": f"{Decimal(money.group('amount').replace(',', '')):.2f}",
            "currency": next(iter(codes)) if len(codes) == 1 else None,
            "currency_candidates": sorted(codes),
            "source_lines": [
                source_line(item) for item in evidence_lines
            ],
        })

    result = {
        "amount": None,
        "currency": None,
        "status": "not_extracted",
        "evidence": matches,
        "tax_evidence": tax_evidence,
    }

    if not matches:
        return result

    values = {(item["amount"], item["currency"]) for item in matches}

    if len(values) != 1 or any(
        len(item["currency_candidates"]) > 1 for item in matches
    ):
        result["status"] = "conflicting"
        return result

    amount, currency = next(iter(values))

    result.update(
        amount=amount,
        currency=currency,
        status="extracted" if currency else "currency_unknown",
    )

    return result

def normalize_cost_amount(text, currency_candidates=()):
    codes = set(currency_candidates) | currency_codes(text)

    result = {
        "amount": None,
        "currency": next(iter(codes)) if len(codes) == 1 else None,
        "status": "not_extracted",
        "currency_candidates": sorted(codes),
    }

    match = MONEY.fullmatch(text)
    if not match:
        return result

    result["amount"] = (
        f"{Decimal(match.group('amount').replace(',', '')):.2f}"
    )
    result["status"] = (
        "conflicting" if len(codes) > 1
        else "extracted" if codes
        else "currency_unknown"
    )

    return result

QUANTITY_HEADERS = {
    "qty", "quantity", "qty/hours", "quantity/hours", "units/hours"
}

UNIT_ALIASES = {
    "h": "hours", "hr": "hours", "hrs": "hours",
    "hour": "hours", "hours": "hours",
    "l": "litre", "litre": "litre", "litres": "litre",
    "liter": "litre", "liters": "litre", "ml": "ml",
    "each": "each", "ea": "each",
    "pc": "piece", "pcs": "piece", "piece": "piece", "pieces": "piece",
    "unit": "unit", "units": "unit",
    "pair": "pair", "pairs": "pair",
    "tube": "tube", "tubes": "tube",
    "kit": "kit", "kits": "kit",
    "bottle": "bottle", "bottles": "bottle",
    "lot": "lot", "lots": "lot",
}


def normalize_cost_quantity(columns, cells, row_kind):
    result = {"value": None, "unit": None, "status": "not_extracted"}

    if row_kind != "item":
        result["status"] = "not_applicable"
        return result

    if len(columns) != len(cells) or len(set(columns)) != len(columns):
        return result

    fields = {
        re.sub(r"\s+", "", column.casefold()): value.strip()
        for column, value in zip(columns, cells)
    }
    if len(fields) != len(columns):
        return result

    quantities = [
        value for column, value in fields.items()
        if column in QUANTITY_HEADERS
    ]
    if len(quantities) != 1:
        return result

    match = re.fullmatch(
        r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([A-Za-z]+)?",
        quantities[0],
    )
    if not match:
        return result

    result["value"] = str(Decimal(match.group(1).replace(",", "")))
    unit_texts = [
        value for value in (match.group(2), fields.get("unit")) if value
    ]
    units = {UNIT_ALIASES.get(value.casefold()) for value in unit_texts}

    if not units or None in units:
        result["status"] = "unit_unknown"
    elif len(units) > 1:
        result["status"] = "conflicting"
    else:
        result.update(unit=units.pop(), status="extracted")

    return result

def extract_cost_tables(record):
    tables = []
    table = None
    section = None

    for line in record["lines"]:
        text = line["text"].replace("**", "").strip()

        if text.startswith("#"):
            section = text.lstrip("#").strip()
            table = None
            continue

        if text.rstrip(":").strip().casefold() == "cost summary":
            section = text.rstrip(":").strip()
            table = None
            continue

        if not text.startswith("|"):
            table = None
            continue

        cells = [cell.strip() for cell in text.strip("|").split("|")]
        is_header = (
            len(cells) >= 2
            and any(
                re.search(r"\brate\b|\bunit\s+cost\b", cell, re.IGNORECASE)
                for cell in cells
            )
            and re.search(
                r"\b(?:total|subtotal|amount)\b",
                cells[-1],
                re.IGNORECASE,
            )
        )

        if is_header:
            table = {
                "id": f"{record['id']}:cost-table:{line['line_number']}",
                "section": section,
                "columns": cells,
                "currency_candidates": sorted(currency_codes(line["text"])),
                "header_evidence": source_line(line),
                "rows": [],
            }
            tables.append(table)
            continue

        if table is None or all(
            re.fullmatch(r"[-:\s]+", cell) for cell in cells
        ):
            continue

        issues = []
        columns = table["columns"]

        if len(cells) != len(columns):
            issues.append("column_count_mismatch")
        if len(set(columns)) != len(columns):
            issues.append("duplicate_columns")
        if not line["text"].rstrip().endswith("|"):
            issues.append("missing_closing_pipe")
        if MONEY.fullmatch(cells[-1]) is None:
            issues.append("unparsed_amount")

        label = cells[0].lower()
        row_kind = (
            "subtotal" if label == "subtotal"
            else "tax" if re.match(r"(?:vat|tax)\b", label)
            else "total" if label in {
                "total", "grand total", "total amount due"
            }
            else "item"
        )

        mapping_ok = (
            len(cells) == len(columns)
            and len(set(columns)) == len(columns)
        )

        table["rows"].append({
            "kind": row_kind,
            "cells": dict(zip(columns, cells)) if mapping_ok else None,
            "raw_cells": cells,
            "reported_quantity": normalize_cost_quantity(
                columns,
                cells,
                row_kind,
            ),
            "reported_amount": normalize_cost_amount(
                cells[-1],
                table["currency_candidates"],
            ),
            "status": "needs_review" if issues else "captured",
            "issues": issues,
            "evidence": source_line(line),
        })

    return tables


def prepare_service_records():
    records = []

    for report in load_service_reports():
        record = extract_report_hours(report)
        record["reported_total_cost"] = extract_report_cost(report)
        record["reported_cost_tables"] = extract_cost_tables(report)
        records.append(record)

    return records


def main():
    records = prepare_service_records()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    totals = [record["reported_total_cost"] for record in records]
    found = sum(item["amount"] is not None for item in totals)
    known_currency = sum(item["status"] == "extracted" for item in totals)

    print(f"Reports processed: {len(records)}")
    print(f"Reported totals: {found}/{len(records)}")
    print(f"Totals with known currency: {known_currency}/{found}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()