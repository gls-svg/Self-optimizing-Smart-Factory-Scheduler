"""Safety validator for generated job-shop schedules.

Run from the project root:
    python safety_validator/validator.py outputs/rl_schedule.csv
"""

import os
import sys

import pandas as pd

try:
    from safety_validator.constraints import (
        REQUIRED_COLUMNS,
        coerce_schedule_types,
        normalize_failure_columns,
        normalize_schedule_columns,
    )
except ModuleNotFoundError:
    # Allows this file to run directly as: python safety_validator/validator.py
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from safety_validator.constraints import (
        REQUIRED_COLUMNS,
        coerce_schedule_types,
        normalize_failure_columns,
        normalize_schedule_columns,
    )


def _make_violation(check_name, message, row=None):
    """Create one consistent violation dictionary."""
    violation = {
        "check": check_name,
        "message": message,
    }
    if row is not None:
        violation.update(row)
    return violation


def _row_identity(row):
    """Return the schedule fields that make a violation easy to locate."""
    return {
        "job_id": row.get("job_id"),
        "operation_id": row.get("operation_id"),
        "machine_id": row.get("machine_id"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
    }


def _check_missing_values(schedule_df):
    violations = []

    for index, row in schedule_df.iterrows():
        missing_columns = [
            column for column in REQUIRED_COLUMNS if pd.isna(row.get(column))
        ]
        if missing_columns:
            details = _row_identity(row)
            details["row_number"] = int(index) + 2
            details["missing_columns"] = ", ".join(missing_columns)
            violations.append(
                _make_violation(
                    "missing_value",
                    "Required field is empty: " + ", ".join(missing_columns),
                    details,
                )
            )

    return violations


def _check_time_validity(schedule_df):
    violations = []

    for index, row in schedule_df.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]

        if pd.isna(start_time) or pd.isna(end_time):
            continue

        if start_time < 0:
            details = _row_identity(row)
            details["row_number"] = int(index) + 2
            violations.append(
                _make_violation(
                    "time_validity",
                    "start_time must be greater than or equal to 0.",
                    details,
                )
            )

        if end_time <= start_time:
            details = _row_identity(row)
            details["row_number"] = int(index) + 2
            details["duration"] = end_time - start_time
            violations.append(
                _make_violation(
                    "time_validity",
                    "end_time must be greater than start_time.",
                    details,
                )
            )

    return violations


def _check_machine_overlaps(schedule_df):
    violations = []

    for machine_id, machine_ops in schedule_df.groupby("machine_id"):
        machine_ops = machine_ops.sort_values(["start_time", "end_time"])
        previous_row = None

        for _, current_row in machine_ops.iterrows():
            if previous_row is not None:
                if current_row["start_time"] < previous_row["end_time"]:
                    details = _row_identity(current_row)
                    details["previous_job_id"] = previous_row["job_id"]
                    details["previous_operation_id"] = previous_row["operation_id"]
                    details["previous_end_time"] = previous_row["end_time"]
                    violations.append(
                        _make_violation(
                            "machine_overlap",
                            f"Machine {machine_id} has overlapping operations.",
                            details,
                        )
                    )

            previous_row = current_row

    return violations


def _check_job_precedence(schedule_df):
    violations = []

    for job_id, job_ops in schedule_df.groupby("job_id"):
        job_ops = job_ops.sort_values(["operation_id", "start_time"])
        previous_row = None

        for _, current_row in job_ops.iterrows():
            if previous_row is not None:
                if current_row["start_time"] < previous_row["end_time"]:
                    details = _row_identity(current_row)
                    details["previous_operation_id"] = previous_row["operation_id"]
                    details["previous_end_time"] = previous_row["end_time"]
                    violations.append(
                        _make_violation(
                            "job_precedence",
                            f"Job {job_id} operation starts before the previous operation ends.",
                            details,
                        )
                    )

            previous_row = current_row

    return violations


def _check_duplicate_operations(schedule_df):
    violations = []
    duplicate_mask = schedule_df.duplicated(
        subset=["job_id", "operation_id"], keep=False
    )

    for index, row in schedule_df[duplicate_mask].iterrows():
        details = _row_identity(row)
        details["row_number"] = int(index) + 2
        violations.append(
            _make_violation(
                "duplicate_operation",
                "Same job_id and operation_id appears more than once.",
                details,
            )
        )

    return violations


def _prepare_failure_windows(failure_path):
    """Read failure data and return normalized failure windows.

    If failure columns are missing or unreadable, return (None, message) so the
    main validator can continue gracefully.
    """
    if not failure_path or not os.path.exists(failure_path):
        return None, "No failure file supplied or found."

    try:
        failure_df = pd.read_csv(failure_path)
    except Exception as exc:
        return None, f"Could not read failure file: {exc}"

    failure_df = normalize_failure_columns(failure_df)
    if failure_df is None:
        return None, "Failure file skipped because required columns were not found."

    failure_df["timestamp"] = pd.to_datetime(failure_df["timestamp"], errors="coerce")
    failure_df["downtime_minutes"] = pd.to_numeric(
        failure_df["downtime_minutes"], errors="coerce"
    )
    failure_df = failure_df.dropna(subset=["timestamp", "machine_id", "downtime_minutes"])

    if failure_df.empty:
        return None, "Failure file skipped because no usable rows were found."

    # Schedules in this project use numeric times. Treat those numbers as minutes
    # after the first recorded failure timestamp.
    first_failure_time = failure_df["timestamp"].min()
    failure_df["failure_start"] = (
        failure_df["timestamp"] - first_failure_time
    ).dt.total_seconds() / 60.0
    failure_df["failure_end"] = (
        failure_df["failure_start"] + failure_df["downtime_minutes"]
    )

    return failure_df, "Failure file checked."


def _check_machine_failures(schedule_df, failure_path):
    violations = []
    failure_df, status_message = _prepare_failure_windows(failure_path)

    if failure_df is None:
        return violations, status_message

    for _, operation in schedule_df.iterrows():
        machine_failures = failure_df[
            failure_df["machine_id"].astype(str) == str(operation["machine_id"])
        ]

        for _, failure in machine_failures.iterrows():
            overlaps_failure = (
                operation["start_time"] < failure["failure_end"]
                and operation["end_time"] > failure["failure_start"]
            )
            if overlaps_failure:
                details = _row_identity(operation)
                details["failure_start"] = failure["failure_start"]
                details["failure_end"] = failure["failure_end"]
                details["failure_timestamp"] = failure["timestamp"]
                violations.append(
                    _make_violation(
                        "machine_failure",
                        "Operation is scheduled during a machine failure window.",
                        details,
                    )
                )

    return violations, status_message


def validate_schedule(schedule_path, failure_path=None):
    """Validate one generated job-shop schedule.

    Args:
        schedule_path: Path to a schedule CSV file.
        failure_path: Optional path to machine failure CSV data.

    Returns:
        A dictionary with is_safe, violations, and summary fields.
    """
    violations = []

    try:
        schedule_df = pd.read_csv(schedule_path)
        schedule_df = normalize_schedule_columns(schedule_df)
        schedule_df = coerce_schedule_types(schedule_df)
    except Exception as exc:
        return {
            "is_safe": False,
            "violations": [
                _make_violation("file_error", f"Could not validate schedule: {exc}")
            ],
            "summary": {
                "schedule_path": schedule_path,
                "total_operations": 0,
                "total_violations": 1,
                "failure_check": "Not run.",
            },
        }

    violations.extend(_check_missing_values(schedule_df))

    # Drop rows missing required values before checks that compare times.
    comparable_df = schedule_df.dropna(subset=REQUIRED_COLUMNS).copy()

    violations.extend(_check_time_validity(comparable_df))
    positive_time_df = comparable_df[
        (comparable_df["start_time"] >= 0)
        & (comparable_df["end_time"] > comparable_df["start_time"])
    ].copy()

    violations.extend(_check_machine_overlaps(positive_time_df))
    violations.extend(_check_job_precedence(positive_time_df))
    violations.extend(_check_duplicate_operations(comparable_df))

    failure_status = "Not run."
    if failure_path is not None:
        failure_violations, failure_status = _check_machine_failures(
            positive_time_df, failure_path
        )
        violations.extend(failure_violations)

    summary = {
        "schedule_path": schedule_path,
        "total_operations": int(len(schedule_df)),
        "total_violations": int(len(violations)),
        "machine_count": int(schedule_df["machine_id"].nunique(dropna=True)),
        "job_count": int(schedule_df["job_id"].nunique(dropna=True)),
        "failure_check": failure_status,
    }

    return {
        "is_safe": len(violations) == 0,
        "violations": violations,
        "summary": summary,
    }


def _print_cli_result(result):
    if result["is_safe"]:
        print("SAFE SCHEDULE")
    else:
        print("UNSAFE SCHEDULE")

    print(f"Total operations: {result['summary'].get('total_operations', 0)}")
    print(f"Total violations: {result['summary'].get('total_violations', 0)}")

    for violation in result["violations"]:
        print(f"- [{violation.get('check')}] {violation.get('message')}")
        for key, value in violation.items():
            if key not in ["check", "message"]:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python safety_validator/validator.py <schedule_csv> [failure_csv]")
        sys.exit(1)

    schedule_csv = sys.argv[1]
    failure_csv = sys.argv[2] if len(sys.argv) > 2 else None
    validation_result = validate_schedule(schedule_csv, failure_csv)
    _print_cli_result(validation_result)
