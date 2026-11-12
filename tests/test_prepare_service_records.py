"""Check service cost parsing and arithmetic without private reports or models."""

import copy
import unittest
from decimal import getcontext

from prepare_service_records import (
    check_line_arithmetic,
    check_report_total,
    check_table_totals,
    extract_cost_tables,
    extract_report_equipment,
    extract_report_cost,
    extract_report_location,
    extract_report_work,
    normalize_cost_rate,
)


def report_with_rows(*rows):
    text_lines = [
        "## Cost Summary",
        "| Item | Quantity | Rate | Total (USD) |",
        "| --- | --- | --- | --- |",
        *rows,
    ]
    return {
        "id": "example",
        "lines": [
            {"line_number": number, "text": text}
            for number, text in enumerate(text_lines, start=1)
        ],
    }


class NormalizeCostRateTests(unittest.TestCase):
    def test_rate_column_can_precede_or_follow_quantity(self):
        before = normalize_cost_rate(
            ["Item", "Rate", "Quantity", "Total (USD)"],
            ["Work", "$17.50/hr", "2 hr", "$35.00"],
            "item", ["USD"],
        )
        after = normalize_cost_rate(
            ["Item", "Quantity", "Unit Rate (USD)", "Total (USD)"],
            ["Work", "2 hr", "$17.50/hr", "$35.00"],
            "item", ["USD"],
        )
        self.assertEqual(before, after)
        self.assertEqual(before["amount"], "17.50")
        self.assertEqual(before["currency"], "USD")
        self.assertEqual(before["per_unit"], "hours")
        self.assertEqual(before["unit_status"], "extracted")

    def test_quantity_does_not_supply_a_missing_rate_basis(self):
        report = report_with_rows("| Lubricant | 150 ml | $7.25 | $7.25 |")
        row = extract_cost_tables(report)[0]["rows"][0]

        self.assertEqual(row["reported_quantity"]["unit"], "ml")
        self.assertEqual(row["reported_rate"]["amount"], "7.25")
        self.assertEqual(row["reported_rate"]["status"], "extracted")
        self.assertIsNone(row["reported_rate"]["per_unit"])
        self.assertEqual(row["reported_rate"]["unit_status"], "unit_unknown")

        unsupported = normalize_cost_rate(
            ["Rate"], ["$7.25/container"], "item", ["USD"],
        )
        self.assertEqual(unsupported["amount"], "7.25")
        self.assertIsNone(unsupported["per_unit"])
        self.assertEqual(unsupported["unit_status"], "unit_unknown")

    def test_zero_row_amount_does_not_turn_missing_rate_into_zero(self):
        report = report_with_rows(
            "| Included work | 1 hr | N/A* | $0.00 |",
            "| Included material | 1 ea | \u2013 | $0.00 |",
        )
        for row in extract_cost_tables(report)[0]["rows"]:
            with self.subTest(rate=row["raw_cells"][2]):
                self.assertEqual(row["reported_amount"]["amount"], "0.00")
                self.assertIsNone(row["reported_rate"]["amount"])
                self.assertEqual(row["reported_rate"]["status"], "not_extracted")
                self.assertEqual(row["reported_rate"]["unit_status"], "not_extracted")

    def test_currency_uncertainty_is_independent_of_explicit_basis(self):
        cases = [
            ("$8.50/hr", [], "currency_unknown", []),
            ("EUR 8.50/hr", ["USD"], "conflicting", ["EUR", "USD"]),
        ]
        for text, currencies, status, candidates in cases:
            with self.subTest(text=text):
                result = normalize_cost_rate(["Rate"], [text], "item", currencies)
                self.assertEqual(result["amount"], "8.50")
                self.assertIsNone(result["currency"])
                self.assertEqual(result["status"], status)
                self.assertEqual(result["currency_candidates"], candidates)
                self.assertEqual(result["per_unit"], "hours")
                self.assertEqual(result["unit_status"], "extracted")

    def test_summary_rows_do_not_get_item_rates(self):
        for kind in ("subtotal", "tax", "total"):
            with self.subTest(kind=kind):
                result = normalize_cost_rate(["Rate"], ["USD 8/hr"], kind)
                self.assertIsNone(result["amount"])
                self.assertIsNone(result["per_unit"])
                self.assertEqual(result["status"], "not_applicable")
                self.assertEqual(result["unit_status"], "not_applicable")

    def test_ambiguous_or_misaligned_columns_do_not_select_a_rate(self):
        cases = [
            (["Rate", "Unit cost"], ["USD 8/hr", "USD 9/hr"]),
            (["Rate", " RATE "], ["USD 8/hr", "USD 9/hr"]),
            (["Item", "Rate"], ["USD 8/hr"]),
        ]
        for columns, cells in cases:
            with self.subTest(columns=columns):
                result = normalize_cost_rate(columns, cells, "item")
                self.assertIsNone(result["amount"])
                self.assertEqual(result["status"], "not_extracted")

    def test_rate_addition_preserves_raw_cells_evidence_and_input(self):
        quote = "| **Work** | 3 hrs | **$9.25/hr** | $27.75 |"
        report = report_with_rows(quote)
        original = copy.deepcopy(report)
        row = extract_cost_tables(report)[0]["rows"][0]

        self.assertEqual(report, original)
        self.assertEqual(row["raw_cells"], ["Work", "3 hrs", "$9.25/hr", "$27.75"])
        self.assertEqual(row["cells"]["Rate"], "$9.25/hr")
        self.assertEqual(row["reported_rate"]["raw_text"], "$9.25/hr")
        self.assertEqual(row["evidence"], {
            "line_start": 4, "line_end": 4, "quote": quote,
        })
        self.assertEqual(row["status"], "captured")
        self.assertEqual(row["issues"], [])


class CheckLineArithmeticTests(unittest.TestCase):
    def item_row(self):
        report = report_with_rows("| Work | 2 hr | $17.50/hr | $35.00 |")
        return extract_cost_tables(report)[0]["rows"][0]

    def check_without_mutation(self, row):
        original = copy.deepcopy(row)
        result = check_line_arithmetic(row)
        self.assertEqual(row, original)
        return result

    def test_exact_match_and_signed_discrepancies(self):
        cases = [
            ("35.00", "matched", "0.00"),
            ("34.50", "mismatch", "-0.50"),
            ("35.50", "mismatch", "0.50"),
        ]
        for reported, status, difference in cases:
            with self.subTest(reported=reported):
                row = self.item_row()
                row["reported_amount"]["amount"] = reported
                result = self.check_without_mutation(row)
                self.assertEqual(result, {
                    "status": status,
                    "reason": None,
                    "calculated_amount": "35.00",
                    "difference": difference,
                    "currency": "USD",
                })

    def test_unresolved_inputs_do_not_receive_an_arithmetic_verdict(self):
        cases = [
            ("reported_quantity", {"status": "unit_unknown", "unit": None},
             "quantity_or_unit_unresolved"),
            ("reported_rate", {"unit_status": "unit_unknown", "per_unit": None},
             "rate_basis_unresolved"),
            ("reported_rate", {"status": "currency_unknown", "currency": None},
             "rate_unresolved"),
            ("reported_amount", {"status": "not_extracted", "amount": None},
             "reported_amount_unresolved"),
        ]
        for field, update, reason in cases:
            with self.subTest(field=field, update=update):
                row = self.item_row()
                row[field].update(update)
                result = self.check_without_mutation(row)
                self.assertEqual(result["status"], "not_checked")
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["calculated_amount"])
                self.assertIsNone(result["difference"])

    def test_known_incompatible_units_or_currencies_are_not_multiplied(self):
        cases = [
            ("reported_quantity", "unit", "litre", "unit_mismatch"),
            ("reported_rate", "currency", "EUR", "currency_mismatch"),
        ]
        for field, key, value, reason in cases:
            with self.subTest(reason=reason):
                row = self.item_row()
                row[field][key] = value
                result = self.check_without_mutation(row)
                self.assertEqual(result["status"], "not_checked")
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["calculated_amount"])

    def test_structural_issues_and_summary_rows_are_excluded(self):
        for update in (
            {"status": "needs_review"},
            {"issues": ["column_count_mismatch"]},
        ):
            with self.subTest(update=update):
                row = self.item_row()
                row.update(update)
                result = self.check_without_mutation(row)
                self.assertEqual(result["status"], "not_checked")
                self.assertEqual(result["reason"], "row_needs_review")

        for kind in ("subtotal", "tax", "total"):
            with self.subTest(kind=kind):
                row = self.item_row()
                row["kind"] = kind
                result = self.check_without_mutation(row)
                self.assertEqual(result["status"], "not_applicable")
                self.assertIsNone(result["calculated_amount"])

    def test_explicit_zero_is_checked_but_fractional_cents_need_a_policy(self):
        row = self.item_row()
        row["reported_rate"]["amount"] = "0.00"
        row["reported_amount"]["amount"] = "0.00"
        result = self.check_without_mutation(row)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["calculated_amount"], "0.00")
        self.assertEqual(result["difference"], "0.00")

        row = self.item_row()
        row["reported_quantity"]["value"] = "1.5"
        row["reported_rate"]["amount"] = "0.01"
        row["reported_amount"]["amount"] = "0.02"
        result = self.check_without_mutation(row)
        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["reason"], "rounding_policy_required")
        self.assertEqual(result["calculated_amount"], "0.015")
        self.assertIsNone(result["difference"])

    def test_invalid_numeric_values_are_not_compared(self):
        fields = [
            ("reported_quantity", "value"),
            ("reported_rate", "amount"),
            ("reported_amount", "amount"),
        ]
        for field, key in fields:
            for value in ("NaN", "Infinity", "-1", "invalid", None):
                with self.subTest(field=field, value=value):
                    row = self.item_row()
                    row[field][key] = value
                    result = self.check_without_mutation(row)
                    self.assertEqual(result["status"], "not_checked")
                    self.assertEqual(result["reason"], "invalid_numeric_value")
                    self.assertIsNone(result["difference"])

    def test_precision_loss_cannot_turn_a_difference_into_a_match(self):
        row = self.item_row()
        row["reported_quantity"]["value"] = "1.00000000000000000000000000001"
        row["reported_rate"]["amount"] = "1.00"
        row["reported_amount"]["amount"] = "1.00"
        context = getcontext().copy()

        result = self.check_without_mutation(row)

        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["reason"], "arithmetic_precision_unresolved")
        self.assertIsNone(result["calculated_amount"])
        self.assertIsNone(result["difference"])
        self.assertEqual(getcontext().prec, context.prec)
        self.assertEqual(getcontext().rounding, context.rounding)
        self.assertEqual(getcontext().traps, context.traps)
        self.assertEqual(getcontext().flags, context.flags)


class CheckTableTotalsTests(unittest.TestCase):
    def table_with_amounts(self, *entries):
        rows = [
            f"| {label} | N/A | N/A | ${amount} |"
            for label, amount in entries
        ]
        return extract_cost_tables(report_with_rows(*rows))[0]

    def check_without_mutation(self, table):
        original = copy.deepcopy(table)
        result = check_table_totals(table)
        self.assertEqual(table, original)
        return result

    def test_explicit_amounts_are_summed_without_quantity_or_rate_inference(self):
        for reported, status, difference in (
            ("30.00", "matched", "0.00"),
            ("29.99", "mismatch", "-0.01"),
            ("30.01", "mismatch", "0.01"),
        ):
            with self.subTest(reported=reported):
                table = self.table_with_amounts(
                    ("Work", "10.00"), ("Material", "20.00"),
                    ("Total", reported),
                )
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["currency"], "USD")
                self.assertEqual(result["item_sum"], "30.00")
                self.assertEqual(result["tax_sum"], "0.00")
                self.assertEqual(result["calculated_total"], "30.00")
                self.assertIsNone(result["subtotal_difference"])
                self.assertEqual(result["total_difference"], difference)

    def test_subtotal_and_stated_taxes_are_checked_separately(self):
        for taxes, expected_tax, expected_total in (
            ([], "0.00", "100.00"),
            ([("VAT", "20.00"), ("Tax fee", "2.50")], "22.50", "122.50"),
        ):
            with self.subTest(taxes=taxes):
                table = self.table_with_amounts(
                    ("Work", "75.00"), ("Material", "25.00"),
                    ("Subtotal", "100.00"), *taxes,
                    ("Total", expected_total),
                )
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], "matched")
                self.assertEqual(result["item_sum"], "100.00")
                self.assertEqual(result["tax_sum"], expected_tax)
                self.assertEqual(result["calculated_total"], expected_total)
                self.assertEqual(result["subtotal_difference"], "0.00")
                self.assertEqual(result["total_difference"], "0.00")

    def test_either_equation_can_fail_independently(self):
        for subtotal, total, subtotal_difference, total_difference in (
            ("110.00", "132.00", "10.00", "0.00"),
            ("100.00", "121.99", "0.00", "-0.01"),
            ("110.00", "130.00", "10.00", "-2.00"),
        ):
            with self.subTest(subtotal=subtotal, total=total):
                table = self.table_with_amounts(
                    ("Work", "100.00"), ("Subtotal", subtotal),
                    ("VAT", "22.00"), ("Total", total),
                )
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], "mismatch")
                self.assertEqual(result["subtotal_difference"], subtotal_difference)
                self.assertEqual(result["total_difference"], total_difference)

    def test_explicit_zero_is_valid_but_absent_total_is_not_reconstructed(self):
        table = self.table_with_amounts(("Included item", "0.00"), ("Total", "0.00"))
        result = self.check_without_mutation(table)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["calculated_total"], "0.00")

        table = self.table_with_amounts(("Work", "10.00"))
        table["reported_total_cost"] = {"amount": "10.00", "currency": "USD"}
        table["rows"][0]["issues"].append("unparsed_amount")
        result = self.check_without_mutation(table)
        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["reason"], "no_reported_total")
        self.assertIsNone(result["calculated_total"])

    def test_unsupported_or_ambiguous_layouts_are_not_summed(self):
        layouts = [
            ["total"],
            ["item", "total", "total"],
            ["item", "subtotal", "subtotal", "total"],
            ["item", "tax", "total"],
            ["subtotal", "item", "total"],
            ["item", "total", "item"],
            ["item", "unknown", "total"],
        ]
        template = self.table_with_amounts(("Work", "10.00"))["rows"][0]
        for kinds in layouts:
            with self.subTest(kinds=kinds):
                table = self.table_with_amounts(("Work", "10.00"))
                table["rows"] = [dict(copy.deepcopy(template), kind=kind) for kind in kinds]
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], "not_checked")
                self.assertEqual(result["reason"], "unsupported_layout")
                self.assertIsNone(result["item_sum"])

    def test_unresolved_amounts_and_currencies_never_get_a_verdict(self):
        cases = [
            ({"amount": None}, "reported_amount_unresolved"),
            ({"status": "not_extracted"}, "reported_amount_unresolved"),
            ({"status": "currency_unknown", "currency": None}, "reported_amount_unresolved"),
            ({"status": "conflicting"}, "reported_amount_unresolved"),
            ({"currency": None}, "currency_mismatch"),
            ({"currency": "EUR"}, "currency_mismatch"),
        ]
        for update, reason in cases:
            with self.subTest(update=update):
                table = self.table_with_amounts(("Work", "10.00"), ("Total", "10.00"))
                table["rows"][0]["reported_amount"].update(update)
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], "not_checked")
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["total_difference"])

        table = self.table_with_amounts(("Work", "10.00"), ("Total", "10.00"))
        table["currency_candidates"] = ["EUR"]
        result = self.check_without_mutation(table)
        self.assertEqual(result["reason"], "currency_mismatch")

    def test_structural_issues_in_any_used_row_prevent_reconciliation(self):
        for index in range(4):
            for update in ({"status": "needs_review"}, {"issues": ["missing_closing_pipe"]}):
                with self.subTest(index=index, update=update):
                    table = self.table_with_amounts(
                        ("Work", "10.00"), ("Subtotal", "10.00"),
                        ("VAT", "2.00"), ("Total", "12.00"),
                    )
                    table["rows"][index].update(update)
                    result = self.check_without_mutation(table)
                    self.assertEqual(result["status"], "not_checked")
                    self.assertEqual(result["reason"], "row_needs_review")

    def test_invalid_or_inexact_numeric_values_are_not_rounded_into_matches(self):
        cases = ["NaN", "Infinity", "-1.00", "invalid", "0.001", "10000000000000000000000000000.01"]
        for amount in cases:
            with self.subTest(amount=amount):
                table = self.table_with_amounts(("Work", "10.00"), ("Total", "10.00"))
                table["rows"][0]["reported_amount"]["amount"] = amount
                table["rows"][1]["reported_amount"]["amount"] = amount
                result = self.check_without_mutation(table)
                self.assertEqual(result["status"], "not_checked")
                self.assertIn(result["reason"], {
                    "invalid_numeric_value", "arithmetic_precision_unresolved",
                })
                self.assertIsNone(result["total_difference"])

    def test_check_preserves_evidence_review_state_and_decimal_context(self):
        for amount in ("10.00", "10.01", "0.001"):
            with self.subTest(amount=amount):
                table = self.table_with_amounts(("Work", "10.00"), ("Total", "10.00"))
                table["review_status"] = "pending"
                table["uncertainty_notes"] = ["Synthetic evidence remains unreviewed."]
                table["rows"][0]["reported_amount"]["amount"] = amount
                context = getcontext().copy()
                self.check_without_mutation(table)
                self.assertEqual(getcontext().prec, context.prec)
                self.assertEqual(getcontext().rounding, context.rounding)
                self.assertEqual(getcontext().traps, context.traps)
                self.assertEqual(getcontext().flags, context.flags)


class CheckReportTotalTests(unittest.TestCase):
    def report_and_record(self, total="30.00", between=()):
        report = report_with_rows(
            "| Work | N/A | N/A | $10.00 |",
            "| Material | N/A | N/A | $20.00 |",
        )
        report["source"] = "example.txt"
        for text in ["", *between, f"Total service cost: USD {total}"]:
            report["lines"].append({
                "line_number": len(report["lines"]) + 1,
                "text": text,
            })
        record = {
            "id": report["id"],
            "source": report["source"],
            "review_status": "pending",
            "uncertainty_notes": ["Synthetic evidence remains unreviewed."],
            "reported_total_cost": extract_report_cost(report),
            "reported_cost_tables": extract_cost_tables(report),
        }
        return report, record

    def check_without_mutation(self, record, report):
        original_record = copy.deepcopy(record)
        original_report = copy.deepcopy(report)
        result = check_report_total(record, report)
        self.assertEqual(record, original_record)
        self.assertEqual(report, original_report)
        return result

    def assert_unchecked(self, record, report, reason):
        result = self.check_without_mutation(record, report)
        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["reason"], reason)
        self.assertIsNone(result["item_sum"])
        self.assertIsNone(result["difference"])

    def test_prose_total_matches_or_reports_a_signed_difference(self):
        for total, status, difference in (
            ("30.00", "matched", "0.00"),
            ("29.99", "mismatch", "-0.01"),
            ("30.01", "mismatch", "0.01"),
        ):
            with self.subTest(total=total):
                report, record = self.report_and_record(total)
                result = self.check_without_mutation(record, report)
                self.assertEqual(result, {
                    "status": status,
                    "reason": None,
                    "table_id": record["reported_cost_tables"][0]["id"],
                    "currency": "USD",
                    "item_sum": "30.00",
                    "difference": difference,
                })

    def test_unresolved_report_totals_and_currencies_are_not_compared(self):
        for update, reason in (
            ({"status": "not_extracted", "amount": None}, "report_total_unresolved"),
            ({"status": "conflicting"}, "report_total_unresolved"),
            ({"status": "currency_unknown", "currency": None}, "report_currency_unresolved"),
            ({"amount": None}, "report_total_unresolved"),
            ({"currency": None}, "report_currency_unresolved"),
        ):
            with self.subTest(update=update):
                report, record = self.report_and_record()
                record["reported_total_cost"].update(update)
                self.assert_unchecked(record, report, reason)

    def test_requires_one_cost_summary_and_skips_its_existing_total(self):
        for change, reason in (
            ("none", "no_cost_summary_table"),
            ("multiple", "ambiguous_cost_tables"),
            ("different_section", "no_cost_summary_table"),
            ("no_section", "no_cost_summary_table"),
            ("table_total", "total_already_checked_in_table"),
        ):
            with self.subTest(change=change):
                report, record = self.report_and_record()
                tables = record["reported_cost_tables"]
                if change == "none":
                    tables.clear()
                elif change == "multiple":
                    tables.append(copy.deepcopy(tables[0]))
                elif change == "different_section":
                    tables[0]["section"] = "Materials"
                elif change == "no_section":
                    tables[0]["section"] = None
                else:
                    total = copy.deepcopy(tables[0]["rows"][-1])
                    total["kind"] = "total"
                    tables[0]["rows"].append(total)
                self.assert_unchecked(record, report, reason)

    def test_unsupported_rows_structural_flags_and_missing_amounts_are_excluded(self):
        cases = [
            ("empty", None, "unsupported_table_layout"),
            ("kind", "subtotal", "unsupported_table_layout"),
            ("kind", "tax", "unsupported_table_layout"),
            ("status", "needs_review", "row_needs_review"),
            ("issues", ["missing_closing_pipe"], "row_needs_review"),
            ("amount", {"amount": None}, "reported_amount_unresolved"),
            ("amount", {"status": "conflicting"}, "reported_amount_unresolved"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field, value=value):
                report, record = self.report_and_record()
                rows = record["reported_cost_tables"][0]["rows"]
                if field == "empty":
                    rows.clear()
                elif field == "amount":
                    rows[-1]["reported_amount"].update(value)
                else:
                    rows[-1][field] = value
                self.assert_unchecked(record, report, reason)

    def test_different_unknown_or_conflicting_currencies_are_not_pooled(self):
        for change in ("item_currency", "unknown_currency", "declared", "report_currency"):
            with self.subTest(change=change):
                report, record = self.report_and_record()
                table = record["reported_cost_tables"][0]
                if change == "declared":
                    table["currency_candidates"] = ["USD", "EUR"]
                elif change == "report_currency":
                    record["reported_total_cost"]["currency"] = "EUR"
                else:
                    table["rows"][0]["reported_amount"]["currency"] = (
                        None if change == "unknown_currency" else "EUR"
                    )
                self.assert_unchecked(record, report, "currency_mismatch")

    def test_source_identity_and_exact_span_coordinates_must_agree(self):
        for change in ("id", "source", "quote", "bounds", "multiline", "numbering"):
            with self.subTest(change=change):
                report, record = self.report_and_record()
                span = record["reported_cost_tables"][0]["rows"][0]["evidence"]
                if change in {"id", "source"}:
                    record[change] = "other"
                elif change == "quote":
                    span["quote"] += " altered"
                elif change == "bounds":
                    span["line_start"] = span["line_end"] = 999
                elif change == "multiline":
                    span["line_end"] += 1
                else:
                    report["lines"][span["line_start"] - 1]["line_number"] += 1
                self.assert_unchecked(record, report, "source_mismatch")

    def test_ordered_contiguous_table_and_same_section_are_required(self):
        for change in ("evidence_count", "span_count", "table_quote", "row_order", "heading", "table_boundary", "total_before_table"):
            with self.subTest(change=change):
                report, record = self.report_and_record()
                table = record["reported_cost_tables"][0]
                matches = record["reported_total_cost"]["evidence"]
                span = matches[0]["source_lines"][0]
                if change == "evidence_count":
                    matches.append(copy.deepcopy(matches[0]))
                elif change == "span_count":
                    matches[0]["source_lines"].append(copy.deepcopy(span))
                elif change == "table_quote":
                    span["quote"] = "| Total | USD 30.00 |"
                elif change == "row_order":
                    table["rows"].reverse()
                elif change == "heading":
                    report["lines"][0]["text"] = "## Different section"
                elif change == "table_boundary":
                    report["lines"][2]["text"] = "## Different section"
                else:
                    report["lines"][0]["text"] = span["quote"]
                    span["line_start"] = span["line_end"] = 1
                self.assert_unchecked(record, report, "report_total_scope_unresolved")

    def test_only_the_explicit_no_charge_note_is_allowed_between_table_and_total(self):
        for note in (
            "Note: Parts supplied from customer's on-site inventory; no charge.",
            "**Note: Parts supplied from customer\u2019s on-site inventory; no charge.**",
        ):
            with self.subTest(note=note):
                report, record = self.report_and_record(between=[note])
                self.assertEqual(self.check_without_mutation(record, report)["status"], "matched")

        for note in (
            "Note: Parts supplied from customer's on-site inventory; additional charge.",
            "## Additional work",
            "Adjustment: USD 5.00",
        ):
            with self.subTest(note=note):
                report, record = self.report_and_record(between=[note])
                self.assert_unchecked(record, report, "unhandled_scope_note")

    def test_invalid_fractional_or_inexact_numbers_do_not_get_rounded_verdicts(self):
        for field in ("item", "report"):
            for value in ("NaN", "Infinity", "-1.00", "invalid", "0.001", "10000000000000000000000000000.01"):
                with self.subTest(field=field, value=value):
                    report, record = self.report_and_record()
                    target = (
                        record["reported_total_cost"] if field == "report"
                        else record["reported_cost_tables"][0]["rows"][0]["reported_amount"]
                    )
                    target["amount"] = value
                    result = self.check_without_mutation(record, report)
                    self.assertEqual(result["status"], "not_checked")
                    self.assertIn(result["reason"], {
                        "invalid_numeric_value", "arithmetic_precision_unresolved",
                    })
                    self.assertIsNone(result["difference"])

        report, record = self.report_and_record()
        rows = record["reported_cost_tables"][0]["rows"]
        rows[0]["reported_amount"]["amount"] = "99999999999999999999999999.99"
        rows[1]["reported_amount"]["amount"] = "1.02"
        self.assert_unchecked(record, report, "arithmetic_precision_unresolved")

    def test_zero_review_state_tax_evidence_and_decimal_context_are_preserved(self):
        for total, amount, status in (
            ("0.00", "0.00", "matched"),
            ("0.00", "0.01", "mismatch"),
            ("0.00", "0.001", "not_checked"),
        ):
            with self.subTest(amount=amount):
                report, record = self.report_and_record(total)
                for row in record["reported_cost_tables"][0]["rows"]:
                    row["reported_amount"]["amount"] = amount
                record["reported_total_cost"]["tax_evidence"] = [{
                    "line_start": 99, "line_end": 99,
                    "quote": "Synthetic tax basis remains unspecified.",
                }]
                context = getcontext().copy()
                result = self.check_without_mutation(record, report)
                self.assertEqual(result["status"], status)
                self.assertEqual(getcontext().prec, context.prec)
                self.assertEqual(getcontext().rounding, context.rounding)
                self.assertEqual(getcontext().traps, context.traps)
                self.assertEqual(getcontext().flags, context.flags)


class ExtractReportLocationTests(unittest.TestCase):
    def report_with_lines(self, *text_lines):
        return {
            "id": "example",
            "source": "example.txt",
            "review_status": "pending",
            "uncertainty_notes": ["Synthetic example remains unreviewed."],
            "text": "\n".join(text_lines),
            "lines": [
                {"line_number": number, "text": text}
                for number, text in enumerate(text_lines, start=1)
            ],
        }

    def extract_without_mutation(self, report):
        original = copy.deepcopy(report)
        result = extract_report_location(report)
        self.assertEqual(report, original)
        return result

    def test_inline_aliases_and_label_formatting_keep_exact_source_evidence(self):
        for label in ("Site Location", "Site", "Location", "Loc", "lOcAtIoN"):
            for template in ("  {label}:   Example Site, AU  ", "- **{label}:** Example Site, AU"):
                with self.subTest(label=label, template=template):
                    quote = template.format(label=label)
                    report = self.report_with_lines("Report header", quote)
                    result = self.extract_without_mutation(report)
                    self.assertEqual(result, {
                        "value": "Example Site, AU",
                        "status": "extracted",
                        "evidence": [{
                            "value": "Example Site, AU",
                            "line_start": 2,
                            "line_end": 2,
                            "quote": quote,
                        }],
                    })

    def test_immediately_following_value_has_an_exact_inclusive_span(self):
        for label in ("Site Location", "Site", "Location", "Loc"):
            with self.subTest(label=label):
                label_line = f"**{label}:**"
                value_line = "  Exampleville, FIN  "
                report = self.report_with_lines("Report header", label_line, value_line)
                result = self.extract_without_mutation(report)
                self.assertEqual(result, {
                    "value": "Exampleville, FIN",
                    "status": "extracted",
                    "evidence": [{
                        "value": "Exampleville, FIN",
                        "line_start": 2,
                        "line_end": 3,
                        "quote": label_line + "\n" + value_line,
                    }],
                })

    def test_unlabelled_places_and_mentions_do_not_supply_a_location(self):
        report = self.report_with_lines(
            "Exampleville, Finland",
            "Customer: Example Mining",
            "Customer Site: Example Site",
            "The location is Example Site.",
            "| Location | Example Site |",
        )
        self.assertEqual(self.extract_without_mutation(report), {
            "value": None,
            "status": "not_extracted",
            "evidence": [],
        })

    def test_placeholders_remain_unresolved_with_the_label_evidence(self):
        for value in (
            "Not provided", "not available", "Not specified", "UNKNOWN",
            "unspecified", "N/A", "NA", "None", "TBD", "Missing",
            "[not provided]", "(unknown)",
        ):
            for next_line in (False, True):
                with self.subTest(value=value, next_line=next_line):
                    texts = ("Location:", value) if next_line else (f"Location: {value}",)
                    report = self.report_with_lines(*texts)
                    result = self.extract_without_mutation(report)
                    self.assertEqual(result["status"], "not_extracted")
                    self.assertIsNone(result["value"])
                    self.assertEqual(result["evidence"], [{
                        "value": None,
                        "line_start": 1,
                        "line_end": 1,
                        "quote": texts[0],
                    }])

    def test_headings_dates_other_fields_tables_and_dividers_are_not_values(self):
        rejected = (
            "## Equipment", "Equipment Details", "Tasks Performed",
            "Findings", "Work Completed", "Cost Summary", "Visit Details",
            "Materials and Parts Used", "Date: 2026-11-10",
            "2026-11-10", "10 November 2026", "November 10, 2026",
            "| Item | Quantity |", "---", "___", "12345",
            "- Example Site", "> Example Site", "`Example Site`",
        )
        for value in rejected:
            for next_line in (False, True):
                with self.subTest(value=value, next_line=next_line):
                    texts = ("Site:", value) if next_line else (f"Site: {value}",)
                    result = self.extract_without_mutation(self.report_with_lines(*texts))
                    self.assertEqual(result["status"], "not_extracted")
                    self.assertIsNone(result["value"])
                    self.assertIsNone(result["evidence"][0]["value"])

        report = self.report_with_lines("Location:", "**Example heading**")
        self.assertEqual(self.extract_without_mutation(report)["status"], "not_extracted")

    def test_missing_values_do_not_jump_over_blank_lines_or_end_of_report(self):
        for texts in (("Location:",), ("Location:", "", "Example Site"), ("Location:", "   ", "Example Site")):
            with self.subTest(texts=texts):
                result = self.extract_without_mutation(self.report_with_lines(*texts))
                self.assertEqual(result, {
                    "value": None,
                    "status": "not_extracted",
                    "evidence": [{
                        "value": None,
                        "line_start": 1,
                        "line_end": 1,
                        "quote": "Location:",
                    }],
                })

    def test_repeated_conflicting_and_unknown_declarations_retain_all_evidence(self):
        for values, status, expected in (
            (["Example Site", "Example Site"], "extracted", "Example Site"),
            (["Example Site", "Other Site"], "conflicting", None),
            (["N/A", "Example Site"], "extracted", "Example Site"),
        ):
            with self.subTest(values=values):
                report = self.report_with_lines(*(f"Site: {value}" for value in values))
                result = self.extract_without_mutation(report)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["value"], expected)
                self.assertEqual(len(result["evidence"]), len(values))
                self.assertEqual(
                    [item["value"] for item in result["evidence"]],
                    [None if value == "N/A" else value for value in values],
                )
                for number, evidence in enumerate(result["evidence"], start=1):
                    self.assertEqual(evidence["line_start"], number)
                    self.assertEqual(evidence["line_end"], number)
                    self.assertEqual(evidence["quote"], report["lines"][number - 1]["text"])

    def test_abbreviations_and_city_only_values_remain_literal_and_unreviewed(self):
        for value in ("Example Site, AU", "Exampleville, FIN", "Demo Plant, Swed", "Exampleville"):
            with self.subTest(value=value):
                report = self.report_with_lines(f"Loc: {value}")
                result = self.extract_without_mutation(report)
                self.assertEqual(result["status"], "extracted")
                self.assertEqual(result["value"], value)
                self.assertEqual(set(result), {"value", "status", "evidence"})
                self.assertEqual(report["review_status"], "pending")
                self.assertEqual(report["uncertainty_notes"], ["Synthetic example remains unreviewed."])


class ExtractReportEquipmentTests(unittest.TestCase):
    def report_with_lines(self, *texts):
        return {
            "id": "example",
            "source": "example.txt",
            "review_status": "pending",
            "uncertainty_notes": ["Synthetic equipment remains unreviewed."],
            "text": "\n".join(texts),
            "lines": [
                {"line_number": number, "text": text}
                for number, text in enumerate(texts, start=1)
            ],
        }

    def extract_without_mutation(self, report):
        original = copy.deepcopy(report)
        result = extract_report_equipment(report)
        self.assertEqual(report, original)
        for field in result.values():
            for evidence in field["evidence"]:
                self.assertEqual(evidence["quote"], "\n".join(
                    line["text"]
                    for line in report["lines"][
                        evidence["line_start"] - 1:evidence["line_end"]
                    ]
                ))
        return result

    def test_direct_aliases_accept_inline_or_immediate_split_values(self):
        for label in (
            "Equipment", "Equipment Serviced", "Equipment Type",
            "Equipment Worked On", "Equip",
        ):
            for split in (False, True):
                with self.subTest(label=label, split=split):
                    texts = (
                        [f"- **{label}:**", "  Example pressure filter  "]
                        if split else [f"- **{label}:** Example pressure filter"]
                    )
                    result = self.extract_without_mutation(self.report_with_lines(*texts))
                    self.assertEqual(result["description"], {
                        "value": "Example pressure filter",
                        "status": "extracted",
                        "evidence": [{
                            "value": "Example pressure filter",
                            "line_start": 1,
                            "line_end": len(texts),
                            "quote": "\n".join(texts),
                        }],
                    })
                    self.assertEqual(result["model"], {
                        "value": None, "status": "not_extracted", "evidence": [],
                    })

    def test_scoped_type_and_model_include_the_equipment_heading(self):
        for heading in (
            "Equipment Details", "**Equipment Information**", "## Equipment Serviced",
        ):
            with self.subTest(heading=heading):
                report = self.report_with_lines(
                    "Report header", heading, "",
                    "- Type: Example pressure filter", "- Model: Q-42",
                )
                result = self.extract_without_mutation(report)
                for field, value, end in (
                    ("description", "Example pressure filter", 4),
                    ("model", "Q-42", 5),
                ):
                    self.assertEqual(result[field]["value"], value)
                    self.assertEqual(result[field]["status"], "extracted")
                    self.assertEqual(result[field]["evidence"][0]["line_start"], 2)
                    self.assertEqual(result[field]["evidence"][0]["line_end"], end)

        report = self.report_with_lines("## Equipment Details", "Model:", "Q-42")
        result = self.extract_without_mutation(report)
        self.assertEqual(result["model"]["value"], "Q-42")
        self.assertEqual(result["model"]["evidence"][0]["line_end"], 3)
        self.assertEqual(result["description"]["status"], "not_extracted")

    def test_unlabelled_bullet_must_be_the_first_nonblank_section_entry(self):
        for prefix, expected in (
            ([], "Example pressure filter"),
            ([""], "Example pressure filter"),
            (["Serial: UNIT-42"], None),
            (["Application: Slurry filtration"], None),
            (["Unclassified section note"], None),
        ):
            with self.subTest(prefix=prefix):
                report = self.report_with_lines(
                    "## Equipment Details", *prefix, "- Example pressure filter",
                )
                result = self.extract_without_mutation(report)
                self.assertEqual(result["description"]["value"], expected)
                if expected:
                    span = result["description"]["evidence"][0]
                    self.assertEqual(span["line_start"], 1)
                    self.assertEqual(span["line_end"], len(report["lines"]))
                else:
                    self.assertEqual(result["description"]["evidence"], [])

    def test_complementary_fields_and_conflicts_are_resolved_independently(self):
        for extra, description_status, model_status in (
            ([], "extracted", "extracted"),
            (["Type: Other filter"], "conflicting", "extracted"),
            (["Model: OTHER-42"], "extracted", "conflicting"),
        ):
            with self.subTest(extra=extra):
                report = self.report_with_lines(
                    "## Equipment Details", "Type: Example pressure filter",
                    "Model: Q-42", *extra,
                )
                result = self.extract_without_mutation(report)
                self.assertEqual(result["description"]["status"], description_status)
                self.assertEqual(result["model"]["status"], model_status)
                self.assertEqual(
                    result["description"]["value"],
                    None if description_status == "conflicting" else "Example pressure filter",
                )
                self.assertEqual(
                    result["model"]["value"],
                    None if model_status == "conflicting" else "Q-42",
                )

    def test_new_headings_dividers_and_unknown_content_end_section_scope(self):
        for boundary in (
            "Tasks Performed", "**Findings**", "## Other section",
            "Cost Summary:", "---", "___", "***", "Unclassified section note",
        ):
            with self.subTest(boundary=boundary):
                report = self.report_with_lines(
                    "## Equipment Details", "Type: Example pressure filter",
                    boundary, "Model: MUST-NOT-CAPTURE", "- Another filter",
                )
                result = self.extract_without_mutation(report)
                self.assertEqual(result["description"]["value"], "Example pressure filter")
                self.assertEqual(len(result["description"]["evidence"]), 1)
                self.assertEqual(result["model"]["status"], "not_extracted")
                self.assertEqual(result["model"]["evidence"], [])

    def test_metadata_and_outside_scope_model_mentions_are_not_equipment(self):
        report = self.report_with_lines(
            "Type: Narrative type", "Model: Narrative model",
            "A Q-42 filter was recommended for future installation.",
            "## Equipment Details", "Serial Number: UNIT-42",
            "Serial/Asset No.: ASSET-42", "Application: Slurry filtration",
            "Manufacturer: Example Works", "Model: Q-42",
        )
        result = self.extract_without_mutation(report)
        self.assertEqual(result["description"], {
            "value": None, "status": "not_extracted", "evidence": [],
        })
        self.assertEqual(result["model"]["value"], "Q-42")
        self.assertEqual(len(result["model"]["evidence"]), 1)
        self.assertEqual(result["model"]["evidence"][0]["line_start"], 4)

    def test_bare_equipment_headings_cannot_fill_a_preceding_empty_field(self):
        for label in ("Equipment", "Type", "Model"):
            for heading in ("Equipment Details", "Equipment Information", "Equipment Serviced"):
                with self.subTest(label=label, heading=heading):
                    prefix = [] if label == "Equipment" else ["## Equipment Details"]
                    report = self.report_with_lines(*prefix, f"{label}:", heading)
                    result = self.extract_without_mutation(report)
                    field = "model" if label == "Model" else "description"
                    self.assertIsNone(result[field]["value"])
                    self.assertEqual(result[field]["status"], "not_extracted")
                    self.assertEqual(len(result[field]["evidence"]), 1)
                    span = result[field]["evidence"][0]
                    self.assertIsNone(span["value"])
                    self.assertEqual(span["line_end"], len(prefix) + 1)

    def test_invalid_values_and_blank_line_jumps_leave_explicit_fields_unknown(self):
        for value in (
            "Not provided", "N/A", "[unknown]", "TBD", "2026-11-10",
            "10 November 2026", "November 10, 2026", "Location: Exampleville",
            "| Equipment | Model |", "---", "## Other heading", "Tasks Performed",
        ):
            for split in (False, True):
                with self.subTest(value=value, split=split):
                    texts = ["Equipment:", value] if split else [f"Equipment: {value}"]
                    result = self.extract_without_mutation(self.report_with_lines(*texts))
                    self.assertEqual(result["description"]["status"], "not_extracted")
                    self.assertIsNone(result["description"]["value"])
                    self.assertIsNone(result["description"]["evidence"][0]["value"])

        for texts in (
            ["Equipment:"], ["Equipment:", "", "Example filter"],
            ["## Equipment Details", "Model:", "", "Q-42"],
        ):
            with self.subTest(texts=texts):
                result = self.extract_without_mutation(self.report_with_lines(*texts))
                self.assertEqual(result["description"]["status"], "not_extracted")
                self.assertEqual(result["model"]["status"], "not_extracted")

    def test_parenthetical_colons_and_unreadable_serial_suffixes_remain_literal(self):
        for value in (
            "Example pressure filter (Units: 3)",
            "Example thickener (Unit ID: Q-01)",
            "Example crusher - serial number unreadable",
        ):
            with self.subTest(value=value):
                report = self.report_with_lines("## Equipment Details", f"- {value}")
                result = self.extract_without_mutation(report)
                self.assertEqual(result["description"]["value"], value)
                self.assertEqual(result["description"]["status"], "extracted")
                self.assertEqual(result["description"]["evidence"][0]["value"], value)
                self.assertEqual(result["model"]["status"], "not_extracted")

    def test_repeated_and_unresolved_declarations_preserve_evidence_and_review_state(self):
        report = self.report_with_lines(
            "Equipment: N/A", "Equipment: Example pressure filter",
            "Equipment: Example pressure filter", "## Equipment Details",
            "Model: Q-42", "Model: Q-42",
        )
        result = self.extract_without_mutation(report)
        self.assertEqual(set(result), {"description", "model"})
        self.assertEqual(result["description"]["status"], "extracted")
        self.assertEqual(result["model"]["status"], "extracted")
        self.assertEqual(
            [item["value"] for item in result["description"]["evidence"]],
            [None, "Example pressure filter", "Example pressure filter"],
        )
        self.assertEqual(len(result["model"]["evidence"]), 2)
        self.assertEqual(report["review_status"], "pending")
        self.assertEqual(report["uncertainty_notes"], ["Synthetic equipment remains unreviewed."])


class ExtractReportWorkTests(unittest.TestCase):
    def report_with_lines(self, *texts):
        return {
            "id": "example",
            "source": "example.txt",
            "review_status": "pending",
            "uncertainty_notes": ["Synthetic work text remains unreviewed."],
            "text": "\n".join(texts),
            "lines": [
                {"line_number": number, "text": text}
                for number, text in enumerate(texts, start=1)
            ],
        }

    def extract_without_mutation(self, report):
        original = copy.deepcopy(report)
        result = extract_report_work(report)
        self.assertEqual(report, original)
        for passage in result["passages"]:
            span = passage["evidence"]
            self.assertEqual(span["quote"], "\n".join(
                line["text"]
                for line in report["lines"][span["line_start"] - 1:span["line_end"]]
            ))
        return result

    def test_supported_headings_accept_markdown_bold_numbering_and_case(self):
        for section in ("Work Completed", "Summary of Work"):
            for template in (
                "{section}", "## {section}", "**{section}:**",
                "3. {section}:", "## 3) **{section}:** ###",
            ):
                with self.subTest(section=section, template=template):
                    heading = template.format(section=section.lower())
                    report = self.report_with_lines(heading, "Inspected the filter.")
                    self.assertEqual(self.extract_without_mutation(report), {
                        "status": "captured",
                        "passages": [{
                            "section": section,
                            "text": "Inspected the filter.",
                            "evidence": {
                                "line_start": 1,
                                "line_end": 2,
                                "quote": heading + "\nInspected the filter.",
                            },
                        }],
                    })

    def test_body_and_inclusive_evidence_preserve_raw_whitespace_and_unicode(self):
        heading = "  **2) Summary of Work:**  "
        body = ["  - Rinsed the filter \u2013 no damage.  ", "\tTested at M\u00e4kel\u00e4 plant.\t"]
        report = self.report_with_lines("Ignored preface", heading, "", "   ", *body, "")
        result = self.extract_without_mutation(report)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["passages"], [{
            "section": "Summary of Work",
            "text": "\n".join(body),
            "evidence": {
                "line_start": 2,
                "line_end": 6,
                "quote": "\n".join([heading, "", "   ", *body]),
            },
        }])

    def test_structural_known_heading_and_nonbullet_metadata_boundaries_stop_capture(self):
        for boundary in (
            "## Different section", "**Inspection results**", "---", "* * *", "___",
            "| Item | Quantity |", "Findings", "3) Recommendations:",
            "Cost Summary", "Equipment Details", "Technician: Example Person",
            "**Location:** Exampleville",
        ):
            with self.subTest(boundary=boundary):
                report = self.report_with_lines(
                    "Work Completed", "Inspected the filter.",
                    boundary, "This later content is outside the captured block.",
                )
                result = self.extract_without_mutation(report)
                self.assertEqual(len(result["passages"]), 1)
                self.assertEqual(result["passages"][0]["text"], "Inspected the filter.")
                self.assertEqual(result["passages"][0]["evidence"]["line_end"], 2)

    def test_task_bullets_and_numbered_tasks_with_colons_remain_in_the_body(self):
        body = [
            "- Clamps: Replaced worn seals.",
            "* Cloths: Cleaned the cloths.",
            "+ Drive: Inspected the coupling.",
            "1. Bearings: Lubricated the bearings.",
            "2) Alignment: Verified the alignment.",
            "The inspection found no additional damage.",
        ]
        report = self.report_with_lines("## Work Completed", *body)
        result = self.extract_without_mutation(report)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["passages"][0]["text"], "\n".join(body))
        self.assertEqual(result["passages"][0]["evidence"]["line_end"], 7)

    def test_leading_blanks_are_skipped_but_blocks_are_not_joined_after_a_blank(self):
        for separator in ("", "   ", "\t"):
            with self.subTest(separator=separator):
                report = self.report_with_lines(
                    "Summary of Work", "", "   ", "Inspected the filter.",
                    separator, "A separate paragraph must not be joined.",
                )
                result = self.extract_without_mutation(report)
                self.assertEqual(len(result["passages"]), 1)
                self.assertEqual(result["passages"][0]["text"], "Inspected the filter.")
                self.assertEqual(result["passages"][0]["evidence"]["line_end"], 4)

    def test_empty_eof_immediate_boundaries_and_adjacent_sections_are_handled(self):
        for texts in (
            [], ["Work Completed"], ["Summary of Work", "", "   "],
            ["Work Completed", "Findings", "No damage found."],
            ["Work Completed", "| Item | Quantity |"],
        ):
            with self.subTest(texts=texts):
                self.assertEqual(
                    self.extract_without_mutation(self.report_with_lines(*texts)),
                    {"status": "not_extracted", "passages": []},
                )

        report = self.report_with_lines(
            "Work Completed", "Summary of Work", "Inspected the filter.",
            "Work Completed", "Inspected the filter.",
        )
        result = self.extract_without_mutation(report)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(
            [passage["section"] for passage in result["passages"]],
            ["Summary of Work", "Work Completed"],
        )
        self.assertEqual(
            [(passage["evidence"]["line_start"], passage["evidence"]["line_end"])
             for passage in result["passages"]],
            [(2, 3), (4, 5)],
        )

    def test_unsupported_aliases_and_narrative_mentions_are_not_work_sections(self):
        for heading in (
            "Work Summary", "Work Performed", "Tasks Performed", "Service Activities",
            "The Work Completed section follows below.", "- Work Completed",
        ):
            with self.subTest(heading=heading):
                report = self.report_with_lines(heading, "Inspected the filter.")
                self.assertEqual(self.extract_without_mutation(report), {
                    "status": "not_extracted", "passages": [],
                })

    def test_future_work_qualifications_and_review_state_are_preserved(self):
        qualification = (
            "Rotor replacement was scheduled for the next visit; "
            "it was not performed today."
        )
        report = self.report_with_lines("Work Completed", qualification)
        result = self.extract_without_mutation(report)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(set(result), {"status", "passages"})
        self.assertEqual(set(result["passages"][0]), {"section", "text", "evidence"})
        self.assertEqual(result["passages"][0]["text"], qualification)
        self.assertEqual(report["review_status"], "pending")
        self.assertEqual(report["uncertainty_notes"], ["Synthetic work text remains unreviewed."])


if __name__ == "__main__":
    unittest.main()
