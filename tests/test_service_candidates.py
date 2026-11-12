"""Check service selection and evidence formatting with synthetic records."""

import copy
import io
import json
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import service_candidates
from service_candidates import find_service_candidates, format_service_context


def example_record(record_id="example"):
    return {
        "id": record_id,
        "source": record_id + ".txt",
        "review_status": "pending",
        "uncertainty_notes": ["Synthetic historical evidence remains unreviewed."],
        "reported_equipment": {
            "description": {
                "value": "Larox PF-DS pressure filter",
                "status": "extracted",
                "evidence": [{
                    "value": "Larox PF-DS pressure filter",
                    "line_start": 3,
                    "line_end": 3,
                    "quote": "Equipment: Larox PF-DS pressure filter",
                }],
            },
            "model": {"value": None, "status": "not_extracted", "evidence": []},
        },
        "reported_work": {
            "status": "captured",
            "passages": [{
                "section": "Work Completed",
                "text": "Replaced the filter cloths.",
                "evidence": {
                    "line_start": 5,
                    "line_end": 6,
                    "quote": "Work Completed\nReplaced the filter cloths.",
                },
            }],
        },
        "reported_location": {
            "value": "Exampleville, Chile",
            "status": "extracted",
            "evidence": [{
                "value": "Exampleville, Chile",
                "line_start": 2,
                "line_end": 2,
                "quote": "Location: Exampleville, Chile",
            }],
        },
        "reported_onsite_hours": {
            "value": 3.5,
            "unit": "hours",
            "status": "extracted",
            "evidence": [{
                "value": 3.5,
                "line_start": 7,
                "line_end": 7,
                "quote": "On-site hours: 3.5",
            }],
        },
        "reported_travel_hours": {
            "value": None,
            "unit": "hours",
            "status": "not_extracted",
            "evidence": [],
        },
        "reported_total_cost": {
            "amount": None,
            "currency": None,
            "status": "not_extracted",
            "evidence": [],
            "tax_evidence": [],
        },
        "reported_cost_tables": [],
        "report_total_check": {
            "status": "not_checked",
            "reason": "report_total_unresolved",
            "table_id": None,
            "currency": None,
            "item_sum": None,
            "difference": None,
        },
    }


class FindServiceCandidatesTests(unittest.TestCase):
    def select_without_mutation(self, records, **terms):
        original = copy.deepcopy(records)
        result = find_service_candidates(records, **terms)
        self.assertEqual(records, original)
        return result

    def test_candidates_are_sorted_by_id_with_original_records_and_passage_references(self):
        records = [example_record(record_id) for record_id in ("visit-Z", "visit-A", "visit-B")]
        result = self.select_without_mutation(records)
        self.assertEqual([item["record"]["id"] for item in result], ["visit-A", "visit-B", "visit-Z"])
        originals = {record["id"]: record for record in records}
        for candidate in result:
            record = originals[candidate["record"]["id"]]
            self.assertIs(candidate["record"], record)
            self.assertEqual(candidate["matched_equipment_fields"], ["description"])
            self.assertIs(candidate["matched_work_passages"][0], record["reported_work"]["passages"][0])
            self.assertIs(candidate["requested_country_mentioned"], True)
        self.assertEqual([record["id"] for record in records], ["visit-Z", "visit-A", "visit-B"])

    def test_only_extracted_equipment_fields_match_without_model_description_inference(self):
        for field in ("description", "model"):
            with self.subTest(field=field):
                record = example_record()
                record["reported_equipment"]["description"].update(
                    status="not_extracted", value=None,
                )
                record["reported_equipment"][field].update(status="extracted", value="pf-ds")
                result = self.select_without_mutation([record])
                self.assertEqual(result[0]["matched_equipment_fields"], [field])

        record = example_record()
        record["reported_equipment"]["model"].update(status="extracted", value="PF-DS")
        self.assertEqual(
            self.select_without_mutation([record])[0]["matched_equipment_fields"],
            ["description", "model"],
        )
        for status in ("not_extracted", "conflicting", "pending"):
            with self.subTest(status=status):
                record = example_record()
                record["reported_equipment"]["description"].update(status=status, value=None)
                self.assertEqual(self.select_without_mutation([record]), [])

    def test_captured_work_is_required_and_only_matching_passages_are_returned(self):
        for status in ("not_extracted", "pending"):
            with self.subTest(status=status):
                record = example_record()
                record["reported_work"]["status"] = status
                self.assertEqual(self.select_without_mutation([record]), [])

        record = example_record()
        work = record["reported_work"]
        match = work["passages"][0]
        unrelated = copy.deepcopy(match)
        unrelated["text"] = "Inspected the pump."
        future_work = copy.deepcopy(match)
        future_work["text"] = "Cloth replacement is scheduled for a future visit."
        work["passages"] = [unrelated, match, future_work]
        result = self.select_without_mutation([record])
        self.assertEqual(result[0]["matched_work_passages"], [match, future_work])
        self.assertIs(result[0]["matched_work_passages"][1], future_work)
        self.assertEqual(future_work["text"], "Cloth replacement is scheduled for a future visit.")

        for passages in ([], [unrelated]):
            with self.subTest(passages=passages):
                record["reported_work"]["passages"] = passages
                self.assertEqual(self.select_without_mutation([record]), [])

    def test_whole_terms_do_not_match_near_family_names_or_work_substrings(self):
        for equipment, work_text, expected in (
            ("pf-ds pressure filter", "Cleaned CLOTHS.", True),
            ("PF-DSX pressure filter", "Replaced cloths.", False),
            ("XPF-DS pressure filter", "Replaced cloths.", False),
            ("PF-DS pressure filter", "Changed protective clothing.", False),
            ("PF-DS pressure filter", "Cleaned the dishcloths.", False),
        ):
            with self.subTest(equipment=equipment, work=work_text):
                record = example_record()
                record["reported_equipment"]["description"]["value"] = equipment
                record["reported_work"]["passages"][0]["text"] = work_text
                self.assertEqual(bool(self.select_without_mutation([record])), expected)

    def test_regex_metacharacters_are_literal_and_surrounding_search_whitespace_is_ignored(self):
        for equipment, work_text, expected in (
            ("Q[8]+ filter", "Inspected A/B? component.", True),
            ("Q888 filter", "Inspected A/B? component.", False),
            ("Q[8]+ filter", "Inspected A/B component.", False),
        ):
            with self.subTest(equipment=equipment, work=work_text):
                record = example_record()
                record["reported_equipment"]["description"]["value"] = equipment
                record["reported_work"]["passages"][0]["text"] = work_text
                result = self.select_without_mutation(
                    [record], equipment_term=" Q[8]+ ", work_terms=iter([" A/B? "]),
                )
                self.assertEqual(bool(result), expected)

    def test_country_is_a_literal_mention_flag_and_never_filters_candidates(self):
        for value, status, expected in (
            ("Exampleville, CHILE", "extracted", True),
            ("Exampleville, Sweden", "extracted", False),
            ("Chilean site", "extracted", False),
            ("ChileMine", "extracted", False),
            (None, "not_extracted", None),
            ("Chile", "conflicting", None),
        ):
            with self.subTest(value=value, status=status):
                record = example_record()
                record["reported_location"].update(value=value, status=status)
                result = self.select_without_mutation([record])
                self.assertEqual(len(result), 1)
                self.assertIs(result[0]["requested_country_mentioned"], expected)
        result = self.select_without_mutation([example_record()], country_term=None)
        self.assertIsNone(result[0]["requested_country_mentioned"])

    def test_missing_costs_provenance_tax_unknowns_and_review_status_are_preserved(self):
        record = example_record()
        record["reported_total_cost"]["tax_evidence"] = [{
            "line_start": 8, "line_end": 8, "quote": "Tax basis is unspecified.",
        }]
        record["reported_cost_tables"] = [{"id": "example:table:9", "rows": []}]
        result = self.select_without_mutation([record])
        self.assertIs(result[0]["record"]["reported_total_cost"], record["reported_total_cost"])
        self.assertIsNone(result[0]["record"]["reported_total_cost"]["amount"])
        self.assertEqual(result[0]["record"]["review_status"], "pending")
        self.assertEqual(set(result[0]), {
            "record", "matched_equipment_fields", "matched_work_passages",
            "requested_country_mentioned",
        })
        del record["reported_total_cost"]
        self.assertIs(self.select_without_mutation([record])[0]["record"], record)
        self.assertNotIn("reported_total_cost", record)

    def test_invalid_terms_raise_before_selection(self):
        for terms in (
            {"equipment_term": ""}, {"equipment_term": "   "},
            {"equipment_term": None}, {"equipment_term": 8},
            {"work_terms": "cloth"}, {"work_terms": b"cloth"},
            {"work_terms": None}, {"work_terms": 1}, {"work_terms": []},
            {"work_terms": [""]}, {"work_terms": ["cloth", None]},
            {"country_term": ""}, {"country_term": "   "}, {"country_term": 8},
        ):
            with self.subTest(terms=terms):
                with self.assertRaises(ValueError):
                    find_service_candidates([], **terms)
        self.assertEqual(find_service_candidates([]), [])

    def test_import_does_not_read_data_write_diagnostics_or_print(self):
        output = io.StringIO()
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected data read")):
            with patch.object(Path, "write_text", side_effect=AssertionError("Unexpected diagnostic write")):
                with redirect_stdout(output):
                    namespace = runpy.run_path(service_candidates.__file__, run_name="candidate_import_check")
        self.assertTrue(callable(namespace["find_service_candidates"]))
        self.assertEqual(output.getvalue(), "")

    def test_main_saves_synthetic_candidate_evidence_to_its_configured_output(self):
        records = [example_record("visit-Z"), example_record("visit-A")]
        records[1]["reported_total_cost"].update(
            amount="12.50", currency="USD", status="extracted",
        )
        records[0]["reported_location"].update(value=None, status="not_extracted")
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "records.json"
            output_path = Path(directory) / "diagnostics" / "candidates.json"
            data_path.write_text(json.dumps(records), encoding="utf-8")
            original_bytes = data_path.read_bytes()
            output = io.StringIO()
            with patch.object(service_candidates, "DATA_PATH", data_path):
                with patch.object(service_candidates, "OUTPUT_PATH", output_path):
                    with redirect_stdout(output):
                        service_candidates.main()
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data_path.read_bytes(), original_bytes)
            self.assertEqual(saved["search"], {
                "equipment_term": "PF-DS", "work_terms": ["cloth", "cloths"],
                "country_term": "Chile",
            })
            self.assertEqual(
                saved["candidates"],
                find_service_candidates(records, **saved["search"]),
            )
            self.assertIn("Candidates: 2", output.getvalue())
            self.assertIn("visit-A | Exampleville, Chile | reported total: USD 12.50", output.getvalue())
            self.assertIn("visit-Z | location unknown | reported total: unknown (not_extracted)", output.getvalue())
            self.assertTrue(output_path.read_bytes().endswith(b"\n"))
            context_path = output_path.with_name("service-context.txt")
            self.assertEqual(
                context_path.read_text(encoding="utf-8"),
                format_service_context(find_service_candidates(records, **saved["search"])),
            )
            self.assertIn(f"Saved service context: {context_path}", output.getvalue())


def example_cost_table(table_id="example:table:10", line_number=10):
    """One synthetic table, including independently unresolved rate units."""
    return {
        "id": table_id,
        "section": "Materials and Parts Used",
        "columns": ["Item", "Quantity", "Rate (EUR)", "Total (EUR)"],
        "currency_candidates": ["EUR"],
        "header_evidence": {
            "line_start": line_number,
            "line_end": line_number,
            "quote": "| Item | Quantity | Rate (EUR) | Total (EUR) |",
        },
        "rows": [{
            "kind": "item",
            "cells": {
                "Item": "Cloth", "Quantity": "3", "Rate (EUR)": "5.00",
                "Total (EUR)": "15.00",
            },
            "raw_cells": ["Cloth", "3", "5.00", "15.00"],
            "reported_quantity": {"value": "3", "unit": None, "status": "unit_unknown"},
            "reported_rate": {
                "amount": "5.00", "currency": "EUR", "status": "extracted",
                "currency_candidates": ["EUR"], "raw_text": "5.00",
                "per_unit": None, "unit_status": "unit_unknown",
            },
            "reported_amount": {
                "amount": "15.00", "currency": "EUR", "status": "extracted",
                "currency_candidates": ["EUR"],
            },
            "status": "captured",
            "issues": [],
            "evidence": {
                "line_start": line_number + 1,
                "line_end": line_number + 1,
                "quote": "| Cloth | 3 | 5.00 | 15.00 |",
            },
            "arithmetic_check": {
                "status": "not_checked", "reason": "quantity_or_unit_unresolved",
                "calculated_amount": None, "difference": None, "currency": None,
            },
        }],
        "totals_check": {
            "status": "not_checked", "reason": "no_reported_total",
            "currency": None, "item_sum": None, "tax_sum": None,
            "calculated_total": None, "subtotal_difference": None,
            "total_difference": None,
        },
    }


class FormatServiceContextTests(unittest.TestCase):
    def render_without_mutation(self, candidates):
        original = copy.deepcopy(candidates)
        text = format_service_context(candidates)
        self.assertEqual(candidates, original)
        self.assertTrue(text.endswith("\n"))
        return text

    def field_details(self, text, title):
        prefix = title + ": "
        values = []
        for line in text.splitlines():
            if line.startswith(prefix):
                try:
                    values.append(json.loads(line[len(prefix):]))
                except json.JSONDecodeError:
                    # Unchanged source quotes may reuse labels such as Location.
                    continue
        return values

    def test_exact_source_quotes_unicode_whitespace_and_field_citations_are_preserved(self):
        record = example_record()
        quote = "Work Completed\n\n  - Replaced the filter cloths; retained café notes.\t  "
        passage = record["reported_work"]["passages"][0]
        passage["text"] = "  - Replaced the filter cloths; retained café notes.\t  "
        passage["evidence"].update(line_end=7, quote=quote)
        candidates = find_service_candidates([record])
        text = self.render_without_mutation(candidates)

        self.assertIn("[example.txt, lines 5-7]\n" + quote + "\n", text)
        for field in (
            record["reported_location"], record["reported_equipment"]["description"],
            record["reported_onsite_hours"],
        ):
            span = field["evidence"][0]
            self.assertIn(
                f"[example.txt, lines {span['line_start']}-{span['line_end']}]\n{span['quote']}\n",
                text,
            )
        self.assertIs(candidates[0]["record"], record)
        self.assertIs(candidates[0]["matched_work_passages"][0], passage)

    def test_nested_final_cost_matches_and_separate_tax_spans_are_retained(self):
        record = example_record()
        cost = record["reported_total_cost"]
        cost.update(
            amount="18.00", currency="EUR", status="extracted",
            evidence=[{
                "amount": "18.00", "currency": "EUR", "currency_candidates": ["EUR"],
                "source_lines": [
                    {"line_start": 10, "line_end": 10, "quote": "| Cost | Amount (EUR) |"},
                    {"line_start": 14, "line_end": 14, "quote": "| Total | 18.00 |"},
                ],
            }],
            tax_evidence=[
                {"line_start": 12, "line_end": 12, "quote": "| Subtotal | 15.00 |"},
                {"line_start": 13, "line_end": 13, "quote": "| Tax | 3.00 |"},
            ],
        )
        text = self.render_without_mutation(find_service_candidates([record]))
        spans = cost["evidence"][0]["source_lines"] + cost["tax_evidence"]
        for span in spans:
            self.assertIn(
                f"[example.txt, lines {span['line_start']}-{span['line_end']}]\n{span['quote']}\n",
                text,
            )
        self.assertEqual(self.field_details(text, "Reported final total"), [{
            "amount": "18.00", "currency": "EUR", "status": "extracted",
        }])
        self.assertEqual(self.field_details(text, "Evidence match"), [{
            "amount": "18.00", "currency": "EUR", "currency_candidates": ["EUR"],
        }])
        self.assertNotIn("Not captured; tax basis remains unspecified.", text)

    def test_separate_tables_keep_raw_incomplete_rows_and_all_normalization_and_checks(self):
        record = example_record()
        first = example_cost_table("example:parts:10", 10)
        second = example_cost_table("example:summary:20", 20)
        second["section"] = "Cost Summary"
        row = second["rows"][0]
        row["cells"]["Total (EUR)"] = "€"
        row["raw_cells"][-1] = "€"
        row["reported_amount"].update(amount=None, status="not_extracted")
        row["status"] = "needs_review"
        row["issues"] = ["missing_closing_pipe", "unparsed_amount"]
        row["evidence"]["quote"] = "| Cloth | 3 | 5.00 | €"
        row["arithmetic_check"]["reason"] = "row_needs_review"
        record["reported_cost_tables"] = [first, second]
        text = self.render_without_mutation(find_service_candidates([record]))

        self.assertIn("Monetary tables captured: 2", text)
        self.assertLess(text.index(first["id"]), text.index(first["rows"][0]["evidence"]["quote"]))
        self.assertLess(text.index(first["rows"][0]["evidence"]["quote"]), text.index(second["id"]))
        self.assertLess(text.index(second["id"]), text.index(row["evidence"]["quote"]))
        self.assertIn(row["evidence"]["quote"] + "\n", text)
        self.assertNotIn(row["evidence"]["quote"] + " |", text)
        self.assertEqual(self.field_details(text, "Separate cost table"), [
            {key: value for key, value in table.items() if key not in {"rows", "header_evidence"}}
            for table in record["reported_cost_tables"]
        ])
        rendered_rows = self.field_details(text, "Row fields and checks")
        for actual, table in zip(rendered_rows, record["reported_cost_tables"]):
            expected = {
                key: value for key, value in table["rows"][0].items()
                if key not in {"cells", "raw_cells", "evidence"}
            }
            self.assertEqual(actual, expected)
            self.assertIn(table["header_evidence"]["quote"], text)
        self.assertIsNone(record["reported_total_cost"]["amount"])

    def test_all_captured_work_is_rendered_including_future_work_outside_the_match_subset(self):
        record = example_record()
        future = {
            "section": "Work Completed",
            "text": "Rotor replacement is scheduled for the next visit.",
            "evidence": {
                "line_start": 30, "line_end": 31,
                "quote": "Work Completed\nRotor replacement is scheduled for the next visit.",
            },
        }
        record["reported_work"]["passages"].append(future)
        candidates = find_service_candidates([record])
        self.assertEqual(len(candidates[0]["matched_work_passages"]), 1)
        text = self.render_without_mutation(candidates)
        self.assertIn(future["evidence"]["quote"], text)
        self.assertIn("[example.txt, lines 30-31]", text)
        self.assertEqual(text.count("Work section: Work Completed"), 2)

    def test_missing_totals_tax_and_travel_aggregates_remain_unknown_despite_priced_rows(self):
        record = example_record()
        table = example_cost_table()
        row = table["rows"][0]
        row["raw_cells"] = ["Travel", "2 hours", "5.00/hour", "10.00"]
        row["cells"] = dict(zip(table["columns"], row["raw_cells"]))
        row["reported_quantity"].update(value="2", unit="hours", status="extracted")
        row["reported_rate"].update(raw_text="5.00/hour", per_unit="hours", unit_status="extracted")
        row["reported_amount"]["amount"] = "10.00"
        row["arithmetic_check"].update(
            status="matched", reason=None, calculated_amount="10.00",
            difference="0.00", currency="EUR",
        )
        row["evidence"]["quote"] = "| Travel | 2 hours | 5.00/hour | 10.00 |"
        record["reported_cost_tables"] = [table]
        text = self.render_without_mutation(find_service_candidates([record]))
        self.assertEqual(self.field_details(text, "Reported travel hours"), [{
            "value": None, "unit": "hours", "status": "not_extracted",
        }])
        self.assertEqual(self.field_details(text, "Reported final total"), [{
            "amount": None, "currency": None, "status": "not_extracted",
        }])
        self.assertEqual(
            self.field_details(text, "Report-total check"), [record["report_total_check"]],
        )
        self.assertIn("Not captured; tax basis remains unspecified.", text)
        self.assertNotIn("tax-free", text)
        self.assertNotIn("Reviewed", text)

    def test_known_zero_amount_and_unresolved_currency_are_preserved_independently(self):
        record = example_record()
        cost = record["reported_total_cost"]
        cost.update(amount="0.00", currency=None, status="currency_unknown")
        record["reported_location"].update(value=None, status="conflicting")
        text = self.render_without_mutation(find_service_candidates([record]))
        self.assertEqual(self.field_details(text, "Reported final total"), [{
            "amount": "0.00", "currency": None, "status": "currency_unknown",
        }])
        self.assertEqual(self.field_details(text, "Requested country term mentioned"), [None])
        self.assertEqual(self.field_details(text, "Location"), [{
            "value": None, "status": "conflicting",
        }])

    def test_empty_candidates_have_an_explicit_result_without_fabricated_records(self):
        text = self.render_without_mutation([])
        self.assertIn("No historical candidates matched.", text)
        self.assertNotIn("ID:", text)

    def test_nonfinite_numbers_are_rejected_without_mutating_records(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=invalid):
                record = example_record()
                candidates = find_service_candidates([record])
                record["reported_onsite_hours"]["value"] = invalid
                original = copy.deepcopy(candidates)
                with self.assertRaises(ValueError):
                    format_service_context(candidates)
                self.assertEqual(candidates, original)


if __name__ == "__main__":
    unittest.main()
