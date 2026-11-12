import json
import re
from decimal import Decimal, Inexact, localcontext

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


def normalize_cost_rate(columns, cells, row_kind, currency_candidates=()):
    """Parse a stated rate, keeping monetary and per-unit certainty separate."""
    result = normalize_cost_amount("", currency_candidates)
    result.update(raw_text=None, per_unit=None, unit_status="not_extracted")

    if row_kind != "item":
        result.update(status="not_applicable", unit_status="not_applicable")
        return result

    headers = [re.sub(r"\s+", " ", name.strip()).casefold() for name in columns]
    if len(headers) != len(cells) or len(set(headers)) != len(headers):
        return result

    rates = [
        value.strip() for header, value in zip(headers, cells)
        if re.fullmatch(
            r"(?:rate|unit\s+rate|unit\s+cost)(?:\s*\([^)]*\))?",
            header,
        )
    ]
    if len(rates) != 1:
        return result

    result["raw_text"] = rates[0]
    amount_text, separator, unit_text = rates[0].partition("/")
    result.update(
        normalize_cost_amount(amount_text.strip(), currency_candidates)
    )
    if result["amount"] is None:
        return result

    if separator:
        result["per_unit"] = UNIT_ALIASES.get(unit_text.strip().casefold())
    result["unit_status"] = (
        "extracted" if result["per_unit"] else "unit_unknown"
    )
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
            "reported_rate": normalize_cost_rate(
                columns,
                cells,
                row_kind,
                table["currency_candidates"],
            ),
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


def check_line_arithmetic(row):
    """Compare explicit compatible line values without approving the evidence."""
    result = {
        "status": "not_checked",
        "reason": None,
        "calculated_amount": None,
        "difference": None,
        "currency": None,
    }

    if row["kind"] != "item":
        result["status"] = "not_applicable"
        return result

    quantity = row["reported_quantity"]
    rate = row["reported_rate"]
    amount = row["reported_amount"]

    checks = [
        (
            row["status"] != "captured" or bool(row["issues"]),
            "row_needs_review",
        ),
        (
            quantity["status"] != "extracted" or not quantity["unit"],
            "quantity_or_unit_unresolved",
        ),
        (rate["status"] != "extracted", "rate_unresolved"),
        (amount["status"] != "extracted", "reported_amount_unresolved"),
        (
            rate["unit_status"] != "extracted" or not rate["per_unit"],
            "rate_basis_unresolved",
        ),
        (quantity["unit"] != rate["per_unit"], "unit_mismatch"),
        (
            not rate["currency"] or rate["currency"] != amount["currency"],
            "currency_mismatch",
        ),
    ]

    for failed, reason in checks:
        if failed:
            result["reason"] = reason
            return result

    try:
        values = [
            Decimal(value)
            for value in (
                quantity["value"],
                rate["amount"],
                amount["amount"],
            )
        ]
        if any(not value.is_finite() or value < 0 for value in values):
            result["reason"] = "invalid_numeric_value"
            return result

        # Never label a rounded-away discrepancy as an exact match.
        with localcontext() as context:
            context.traps[Inexact] = True
            calculated = values[0] * values[1]
            difference = values[2] - calculated

        result.update(
            currency=amount["currency"],
            calculated_amount=f"{calculated:f}",
        )

        if calculated != calculated.quantize(Decimal("0.01")):
            result["reason"] = "rounding_policy_required"
            return result

    except Inexact:
        result["reason"] = "arithmetic_precision_unresolved"
        return result
    except (ArithmeticError, ValueError, TypeError):
        result["reason"] = "invalid_numeric_value"
        return result

    result.update(
        status="matched" if difference == 0 else "mismatch",
        calculated_amount=f"{calculated:.2f}",
        difference=f"{difference:.2f}",
    )
    return result


def check_table_totals(table):
    """Check reported table amounts without reconstructing missing values."""
    result = {
        "status": "not_checked",
        "reason": None,
        "currency": None,
        "item_sum": None,
        "tax_sum": None,
        "calculated_total": None,
        "subtotal_difference": None,
        "total_difference": None,
    }
    rows = table["rows"]
    items = [row for row in rows if row["kind"] == "item"]
    subtotals = [row for row in rows if row["kind"] == "subtotal"]
    taxes = [row for row in rows if row["kind"] == "tax"]
    totals = [row for row in rows if row["kind"] == "total"]

    if not totals:
        result["reason"] = "no_reported_total"
        return result

    expected = (
        ["item"] * len(items)
        + ["subtotal"] * len(subtotals)
        + ["tax"] * len(taxes)
        + ["total"]
    )
    if (
        len(totals) != 1
        or len(subtotals) > 1
        or not items
        or (taxes and not subtotals)
        or [row["kind"] for row in rows] != expected
    ):
        result["reason"] = "unsupported_layout"
        return result

    if any(row["status"] != "captured" or row["issues"] for row in rows):
        result["reason"] = "row_needs_review"
        return result

    amounts = [row["reported_amount"] for row in rows]
    if any(
        item["status"] != "extracted" or item["amount"] is None
        for item in amounts
    ):
        result["reason"] = "reported_amount_unresolved"
        return result

    currencies = {item["currency"] for item in amounts}
    declared = set(table["currency_candidates"])
    if (
        len(currencies) != 1
        or not all(currencies)
        or (declared and declared != currencies)
    ):
        result["reason"] = "currency_mismatch"
        return result

    try:
        with localcontext() as context:
            context.traps[Inexact] = True
            values = [Decimal(item["amount"]) for item in amounts]
            if any(not value.is_finite() or value < 0 for value in values):
                result["reason"] = "invalid_numeric_value"
                return result
            for value in values:
                value.quantize(Decimal("0.01"))

            count = len(items)
            item_sum = sum(values[:count], Decimal("0"))
            tax_start = count + len(subtotals)
            tax_sum = sum(values[tax_start:-1], Decimal("0"))
            base = values[count] if subtotals else item_sum
            calculated_total = base + tax_sum
            subtotal_difference = values[count] - item_sum if subtotals else None
            total_difference = values[-1] - calculated_total
    except Inexact:
        result["reason"] = "arithmetic_precision_unresolved"
        return result
    except (ArithmeticError, ValueError, TypeError):
        result["reason"] = "invalid_numeric_value"
        return result

    matched = total_difference == 0 and (
        subtotal_difference is None or subtotal_difference == 0
    )
    result.update(
        status="matched" if matched else "mismatch",
        currency=next(iter(currencies)),
        item_sum=f"{item_sum:.2f}",
        tax_sum=f"{tax_sum:.2f}",
        calculated_total=f"{calculated_total:.2f}",
        subtotal_difference=(
            f"{subtotal_difference:.2f}" if subtotal_difference is not None else None
        ),
        total_difference=f"{total_difference:.2f}",
    )
    return result


def check_report_total(record, report):
    """Compare one item-only Cost Summary with its separate prose total."""
    result = {
        "status": "not_checked",
        "reason": None,
        "table_id": None,
        "currency": None,
        "item_sum": None,
        "difference": None,
    }

    def unchecked(reason):
        result["reason"] = reason
        return result

    cost = record["reported_total_cost"]
    if cost["status"] == "currency_unknown":
        return unchecked("report_currency_unresolved")
    if cost["status"] != "extracted" or cost["amount"] is None:
        return unchecked("report_total_unresolved")
    if not cost["currency"]:
        return unchecked("report_currency_unresolved")

    tables = record["reported_cost_tables"]
    if not tables:
        return unchecked("no_cost_summary_table")
    if len(tables) != 1:
        return unchecked("ambiguous_cost_tables")

    table = tables[0]
    section_pattern = r"(?:\d+[.)]?\s*)?cost\s+summary"
    if not re.fullmatch(section_pattern, table["section"] or "", re.IGNORECASE):
        return unchecked("no_cost_summary_table")
    result["table_id"] = table["id"]

    rows = table["rows"]
    if any(row["kind"] == "total" for row in rows):
        return unchecked("total_already_checked_in_table")
    if not rows or any(row["kind"] != "item" for row in rows):
        return unchecked("unsupported_table_layout")
    if any(row["status"] != "captured" or row["issues"] for row in rows):
        return unchecked("row_needs_review")

    amounts = [row["reported_amount"] for row in rows]
    if any(
        amount["status"] != "extracted" or amount["amount"] is None
        for amount in amounts
    ):
        return unchecked("reported_amount_unresolved")
    currencies = {amount["currency"] for amount in amounts}
    declared = set(table["currency_candidates"])
    if currencies != {cost["currency"]} or (declared and declared != currencies):
        return unchecked("currency_mismatch")

    if record["id"] != report["id"] or record["source"] != report["source"]:
        return unchecked("source_mismatch")
    matches = cost["evidence"]
    if len(matches) != 1 or len(matches[0]["source_lines"]) != 1:
        return unchecked("report_total_scope_unresolved")
    total_span = matches[0]["source_lines"][0]
    if total_span["quote"].lstrip().startswith("|"):
        return unchecked("report_total_scope_unresolved")

    raw_lines = report["lines"]
    spans = [
        table["header_evidence"],
        *[row["evidence"] for row in rows],
        total_span,
    ]
    for span in spans:
        number = span["line_start"]
        if (
            number != span["line_end"]
            or not 1 <= number <= len(raw_lines)
            or raw_lines[number - 1]["line_number"] != number
            or raw_lines[number - 1]["text"] != span["quote"]
        ):
            return unchecked("source_mismatch")

    positions = [span["line_start"] for span in spans]
    if any(left >= right for left, right in zip(positions, positions[1:])):
        return unchecked("report_total_scope_unresolved")

    header_line = positions[0]
    previous = next(
        (
            line["text"].strip()
            for line in reversed(raw_lines[:header_line - 1])
            if line["text"].strip()
        ),
        "",
    )
    heading = previous.lstrip("#").replace("**", "").strip().rstrip(":").strip()
    if heading.casefold() != table["section"].casefold():
        return unchecked("report_total_scope_unresolved")

    # The table must remain contiguous within the verified section.
    last_row_line = positions[-2]
    if any(
        line["text"].strip() and not line["text"].lstrip().startswith("|")
        for line in raw_lines[header_line:last_row_line]
    ):
        return unchecked("report_total_scope_unresolved")

    # Unknown intervening prose may describe an adjustment or a new section.
    # Accept only blank lines or this explicit no-charge inventory note.
    for line in raw_lines[last_row_line:positions[-1] - 1]:
        text = line["text"].replace("*", "").strip()
        if text and not re.fullmatch(
            r"Note:\s+Parts supplied from customer['\u2019]s "
            r"on-site inventory;\s+no charge\.",
            text,
            re.IGNORECASE,
        ):
            return unchecked("unhandled_scope_note")

    try:
        with localcontext() as context:
            context.traps[Inexact] = True
            values = [Decimal(amount["amount"]) for amount in amounts]
            reported = Decimal(cost["amount"])
            for value in [*values, reported]:
                if not value.is_finite() or value < 0:
                    return unchecked("invalid_numeric_value")
                value.quantize(Decimal("0.01"))
            item_sum = sum(values, Decimal("0"))
            difference = reported - item_sum
    except Inexact:
        return unchecked("arithmetic_precision_unresolved")
    except (ArithmeticError, ValueError, TypeError):
        return unchecked("invalid_numeric_value")

    result.update(
        status="matched" if difference == 0 else "mismatch",
        currency=cost["currency"],
        item_sum=f"{item_sum:.2f}",
        difference=f"{difference:.2f}",
    )
    return result


def extract_report_location(report):
    """Capture explicitly labelled locations without inferring geography."""
    def location_value(text):
        value = text.strip()

        if (
            not value
            or value.startswith(
                ("#", "|", "- ", "* ", "+ ", ">", "`", "**")
            )
            or ":" in value
            or not any(character.isalpha() for character in value)
            or re.match(
                r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|"
                r"\d{1,2}\s+[A-Za-z]+\s+\d{2,4}\b|"
                r"[A-Za-z]+\s+\d{1,2},?\s+\d{2,4}\b)",
                value,
            )
            or value.strip("[]()").casefold() in {
                "not provided", "not available", "not specified",
                "unknown", "unspecified", "n/a", "na", "none",
                "tbd", "missing",
            }
            or re.fullmatch(
                r"(?:equipment(?: details| information| serviced)?|"
                r"tasks(?: performed)?|findings|work completed|"
                r"cost summary|visit details|materials and parts used)",
                value,
                re.IGNORECASE,
            )
        ):
            return None

        return value

    evidence = []
    lines = report["lines"]

    for index, line in enumerate(lines):
        text = (
            line["text"]
            .replace("**", "")
            .strip()
            .removeprefix("- ")
            .strip()
        )
        match = re.fullmatch(
            r"(?:site location|site|location|loc)\s*:\s*(.*)",
            text,
            re.IGNORECASE,
        )
        if not match:
            continue

        raw_value = match.group(1).strip()
        value = location_value(raw_value)
        span = source_line(line)

        if not raw_value and index + 1 < len(lines):
            next_line = lines[index + 1]
            value = location_value(next_line["text"])

            if value is not None:
                span = {
                    "line_start": line["line_number"],
                    "line_end": next_line["line_number"],
                    "quote": line["text"] + "\n" + next_line["text"],
                }

        evidence.append({"value": value, **span})

    values = {
        item["value"]
        for item in evidence
        if item["value"] is not None
    }

    return {
        "value": next(iter(values)) if len(values) == 1 else None,
        "status": (
            "extracted" if len(values) == 1
            else "conflicting" if values
            else "not_extracted"
        ),
        "evidence": evidence,
    }


def extract_report_equipment(report):
    """Capture declared equipment and models without narrative inference."""
    lines = report["lines"]
    evidence = {"description": [], "model": []}
    equipment_headings = {
        "equipment details",
        "equipment information",
        "equipment serviced",
    }
    other_heading = (
        r"(?:\d+[.)]?\s*)?(?:tasks(?: performed(?:\s*&\s*time spent)?)?|"
        r"summary of work|work summary|summary|work performed|work completed|"
        r"findings|cost summary|materials(?: and parts)? used|visit details|report)"
        r"\s*:?"
    )

    def clean(text):
        return text.replace("**", "").strip().removeprefix("- ").strip()

    def read_value(text):
        value = text.strip()

        if (
            not value
            or value.startswith(
                ("#", "|", "- ", "* ", "+ ", ">", "`", "**")
            )
            or not any(character.isalnum() for character in value)
            or value.rstrip(":").casefold() in equipment_headings
            or value.strip("[]()").casefold() in {
                "not provided", "not available", "not specified",
                "unknown", "unspecified", "n/a", "na", "none",
                "tbd", "missing",
            }
            or re.match(r"[A-Za-z][A-Za-z /.-]*\s*:", value)
            or re.match(
                r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|"
                r"\d{1,2}\s+[A-Za-z]+\s+\d{2,4}\b|"
                r"[A-Za-z]+\s+\d{1,2},?\s+\d{2,4}\b)",
                value,
            )
            or re.fullmatch(other_heading, value, re.IGNORECASE)
        ):
            return None

        return value

    def capture(field, raw_value, index, start=None):
        value = read_value(raw_value)
        end = index

        if not raw_value.strip() and index + 1 < len(lines):
            candidate = read_value(lines[index + 1]["text"])

            if candidate is not None:
                value = candidate
                end = index + 1

        first = index if start is None else start
        evidence[field].append({
            "value": value,
            "line_start": lines[first]["line_number"],
            "line_end": lines[end]["line_number"],
            "quote": "\n".join(
                line["text"] for line in lines[first:end + 1]
            ),
        })
        return end

    section_start = None
    allow_bullet = False
    consumed_through = -1

    for index, line in enumerate(lines):
        if index <= consumed_through:
            continue

        raw = line["text"].strip()
        text = clean(raw)

        if not text:
            continue

        heading = (
            text.lstrip("#").strip().rstrip(":").strip().casefold()
        )
        is_equipment_heading = heading in equipment_headings

        if is_equipment_heading:
            section_start = index
            allow_bullet = True

        direct = re.fullmatch(
            r"(?:equipment(?: serviced| type| worked on)?|equip)"
            r"\s*:\s*(.*)",
            text,
            re.IGNORECASE,
        )

        if direct:
            consumed_through = capture(
                "description", direct.group(1), index
            )
            allow_bullet = False
            continue

        if is_equipment_heading:
            continue

        if (
            raw.startswith("#")
            or re.fullmatch(r"[-*_]{3,}", raw)
            or re.fullmatch(other_heading, text, re.IGNORECASE)
        ):
            section_start = None
            allow_bullet = False
            continue

        if section_start is None:
            continue

        scoped = re.fullmatch(
            r"(type|model)\s*:\s*(.*)", text, re.IGNORECASE
        )

        if scoped:
            field = (
                "model"
                if scoped.group(1).casefold() == "model"
                else "description"
            )
            consumed_through = capture(
                field, scoped.group(2), index, section_start
            )
            allow_bullet = False
            continue

        if re.match(
            r"(?:serial(?: numbers?| no\.?|/asset no\.?)?|"
            r"application|manufacturer)\s*:",
            text,
            re.IGNORECASE,
        ):
            allow_bullet = False
            continue

        if (
            allow_bullet
            and raw.startswith("- ")
            and read_value(text) is not None
        ):
            capture("description", text, index, section_start)
            allow_bullet = False
            continue

        section_start = None
        allow_bullet = False

    result = {}

    for field, matches in evidence.items():
        values = {
            item["value"]
            for item in matches
            if item["value"] is not None
        }
        result[field] = {
            "value": next(iter(values)) if len(values) == 1 else None,
            "status": (
                "extracted" if len(values) == 1
                else "conflicting" if values
                else "not_extracted"
            ),
            "evidence": matches,
        }

    return result


def extract_report_work(report):
    """Copy bounded work passages without interpreting the reported actions."""
    lines = report["lines"]
    passages = []
    section_names = {
        "work completed": "Work Completed",
        "summary of work": "Summary of Work",
    }

    def heading_text(text):
        text = text.strip().replace("**", "")
        text = re.sub(r"^#{1,6}\s*", "", text).strip().rstrip("#").strip()
        text = re.sub(r"^\d+[.)]\s*", "", text)
        return text.rstrip(":").strip()

    def section_name(text):
        return section_names.get(heading_text(text).casefold())

    def boundary(text):
        raw = text.strip()
        plain = heading_text(raw)

        return bool(
            raw.startswith(("#", "|"))
            or re.fullmatch(r"(?:[-*_]\s*){3,}", raw)
            or re.fullmatch(r"\*\*[^*]+\*\*:?", raw)
            or section_name(raw)
            or re.fullmatch(
                r"findings|recommendations|cost summary|"
                r"materials(?: and parts)? used|materials / parts used|"
                r"tasks performed|equipment (?:details|information|serviced)",
                plain,
                re.IGNORECASE,
            )
            or re.match(
                r"[A-Za-z][A-Za-z /().-]*\s*:", raw.replace("**", "")
            )
        )

    for index, line in enumerate(lines):
        section = section_name(line["text"])
        if section is None:
            continue

        start = index + 1
        while start < len(lines) and not lines[start]["text"].strip():
            start += 1

        end = start
        while (
            end < len(lines)
            and lines[end]["text"].strip()
            and not boundary(lines[end]["text"])
        ):
            end += 1

        if end == start:
            continue

        passages.append({
            "section": section,
            "text": "\n".join(
                item["text"] for item in lines[start:end]
            ),
            "evidence": {
                "line_start": line["line_number"],
                "line_end": lines[end - 1]["line_number"],
                "quote": "\n".join(
                    item["text"] for item in lines[index:end]
                ),
            },
        })

    return {
        "status": "captured" if passages else "not_extracted",
        "passages": passages,
    }


def prepare_service_records():
    records = []

    for report in load_service_reports():
        record = extract_report_hours(report)
        record["reported_location"] = extract_report_location(report)
        record["reported_equipment"] = extract_report_equipment(report)
        record["reported_work"] = extract_report_work(report)
        record["reported_total_cost"] = extract_report_cost(report)
        record["reported_cost_tables"] = extract_cost_tables(report)
        for table in record["reported_cost_tables"]:
            table["totals_check"] = check_table_totals(table)
            for row in table["rows"]:
                row["arithmetic_check"] = check_line_arithmetic(row)
        record["report_total_check"] = check_report_total(record, report)
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
