"""Column handling helpers for schedule safety validation.

The project has a few schedule formats, so this file keeps the accepted
column aliases in one place. The validator normalizes every schedule to these
standard column names before running checks.
"""

import pandas as pd


REQUIRED_COLUMNS = ["job_id", "operation_id", "machine_id", "start_time", "end_time"]


COLUMN_ALIASES = {
    "job_id": [
        "job_id",
        "job",
        "jobs",
        "global_job_id",
        "jobid",
        "job name",
        "job_name",
    ],
    "operation_id": [
        "operation_id",
        "operation",
        "op",
        "op_id",
        "operationid",
        "operation_number",
    ],
    "machine_id": [
        "machine_id",
        "machine",
        "machineid",
        "machine_number",
        "resource",
    ],
    "start_time": [
        "start_time",
        "start",
        "starttime",
        "begin",
        "begin_time",
    ],
    "end_time": [
        "end_time",
        "end",
        "endtime",
        "finish",
        "finish_time",
        "completion_time",
    ],
}


FAILURE_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "failure_time", "start_time", "start"],
    "machine_id": ["machine_id", "machine", "machineid"],
    "downtime_minutes": [
        "downtime_minutes",
        "downtime",
        "duration",
        "duration_minutes",
        "repair_time",
    ],
}


def _clean_column_name(column_name):
    """Make a column name easier to compare with known aliases."""
    return str(column_name).strip().lower().replace(" ", "_")


def _build_rename_map(columns, aliases):
    """Return a mapping from original column names to standard names."""
    cleaned_lookup = {_clean_column_name(column): column for column in columns}
    rename_map = {}

    for standard_name, possible_names in aliases.items():
        for possible_name in possible_names:
            cleaned_name = _clean_column_name(possible_name)
            if cleaned_name in cleaned_lookup:
                rename_map[cleaned_lookup[cleaned_name]] = standard_name
                break

    return rename_map


def normalize_schedule_columns(schedule_df):
    """Normalize schedule columns and return a copied DataFrame.

    Raises:
        ValueError: If one or more required schedule columns are missing.
    """
    normalized_df = schedule_df.copy()
    normalized_df = normalized_df.rename(
        columns=_build_rename_map(normalized_df.columns, COLUMN_ALIASES)
    )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in normalized_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required schedule columns after normalization: "
            + ", ".join(missing_columns)
        )

    return normalized_df


def normalize_failure_columns(failure_df):
    """Normalize machine failure columns.

    Failure validation is optional, so this function returns None instead of
    raising an error when the failure file does not have usable columns.
    """
    normalized_df = failure_df.copy()
    normalized_df = normalized_df.rename(
        columns=_build_rename_map(normalized_df.columns, FAILURE_COLUMN_ALIASES)
    )

    needed_columns = ["timestamp", "machine_id", "downtime_minutes"]
    if any(column not in normalized_df.columns for column in needed_columns):
        return None

    return normalized_df


def coerce_schedule_types(schedule_df):
    """Convert time and operation columns to numeric values when possible."""
    coerced_df = schedule_df.copy()

    for column in ["operation_id", "start_time", "end_time"]:
        coerced_df[column] = pd.to_numeric(coerced_df[column], errors="coerce")

    return coerced_df
