"""Basic tests for the safety validator.

Run from the project root:
    python -m unittest safety_validator.test_validator
"""

import os
import tempfile
import unittest

import pandas as pd

from safety_validator.validator import validate_schedule


class TestSafetyValidator(unittest.TestCase):
    def _write_temp_schedule(self, rows):
        """Create a temporary schedule CSV and return its path."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        pd.DataFrame(rows).to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name

    def _validate_rows(self, rows):
        path = self._write_temp_schedule(rows)
        try:
            return validate_schedule(path)
        finally:
            os.remove(path)

    def test_valid_schedule(self):
        result = self._validate_rows(
            [
                {
                    "job_id": "J1",
                    "operation_id": 1,
                    "machine_id": 1,
                    "start_time": 0,
                    "end_time": 5,
                },
                {
                    "job_id": "J1",
                    "operation_id": 2,
                    "machine_id": 2,
                    "start_time": 5,
                    "end_time": 9,
                },
                {
                    "job_id": "J2",
                    "operation_id": 1,
                    "machine_id": 1,
                    "start_time": 5,
                    "end_time": 8,
                },
            ]
        )

        self.assertTrue(result["is_safe"])
        self.assertEqual(result["summary"]["total_violations"], 0)

    def test_machine_overlap_invalid(self):
        result = self._validate_rows(
            [
                {
                    "job_id": "J1",
                    "operation_id": 1,
                    "machine_id": 1,
                    "start_time": 0,
                    "end_time": 5,
                },
                {
                    "job_id": "J2",
                    "operation_id": 1,
                    "machine_id": 1,
                    "start_time": 4,
                    "end_time": 7,
                },
            ]
        )

        checks = [violation["check"] for violation in result["violations"]]
        self.assertFalse(result["is_safe"])
        self.assertIn("machine_overlap", checks)

    def test_job_precedence_invalid(self):
        result = self._validate_rows(
            [
                {
                    "job_id": "J1",
                    "operation_id": 1,
                    "machine_id": 1,
                    "start_time": 0,
                    "end_time": 5,
                },
                {
                    "job_id": "J1",
                    "operation_id": 2,
                    "machine_id": 2,
                    "start_time": 3,
                    "end_time": 8,
                },
            ]
        )

        checks = [violation["check"] for violation in result["violations"]]
        self.assertFalse(result["is_safe"])
        self.assertIn("job_precedence", checks)


if __name__ == "__main__":
    unittest.main()
