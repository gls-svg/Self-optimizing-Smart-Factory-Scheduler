"""Generate safety validation reports for job-shop schedules."""

import os

import pandas as pd

try:
    from safety_validator.validator import validate_schedule
except ModuleNotFoundError:
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from safety_validator.validator import validate_schedule


def generate_report(validation_results, output_dir="outputs"):
    """Save a text summary and CSV list of all violations.

    Args:
        validation_results: Either one validation result dictionary or a
            dictionary like {"rl_schedule.csv": result}.
        output_dir: Folder where report files should be written.

    Returns:
        Paths to the generated text and CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "is_safe" in validation_results:
        validation_results = {
            validation_results["summary"].get("schedule_path", "schedule"):
            validation_results
        }

    report_path = os.path.join(output_dir, "safety_validation_report.txt")
    violations_path = os.path.join(output_dir, "safety_violations.csv")

    report_lines = [
        "SAFETY VALIDATION REPORT",
        "========================",
        "",
    ]
    all_violations = []

    for schedule_name, result in validation_results.items():
        status = "SAFE" if result["is_safe"] else "UNSAFE"
        summary = result.get("summary", {})

        report_lines.extend(
            [
                f"Schedule: {schedule_name}",
                f"Status: {status}",
                f"Total operations: {summary.get('total_operations', 0)}",
                f"Total violations: {summary.get('total_violations', 0)}",
                f"Machines: {summary.get('machine_count', 0)}",
                f"Jobs: {summary.get('job_count', 0)}",
                f"Failure check: {summary.get('failure_check', 'Not run.')}",
                "",
            ]
        )

        if result.get("violations"):
            report_lines.append("Violations:")
            for violation in result["violations"]:
                report_lines.append(
                    f"- [{violation.get('check')}] {violation.get('message')}"
                )
                violation_row = {"schedule": schedule_name}
                violation_row.update(violation)
                all_violations.append(violation_row)
        else:
            report_lines.append("Violations: None")

        report_lines.append("")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))

    if all_violations:
        pd.DataFrame(all_violations).to_csv(violations_path, index=False)
    else:
        pd.DataFrame(columns=["schedule", "check", "message"]).to_csv(
            violations_path, index=False
        )

    return report_path, violations_path


def validate_and_report(schedule_paths, failure_path=None, output_dir="outputs"):
    """Validate multiple schedule files and generate report files."""
    results = {}

    for schedule_path in schedule_paths:
        if os.path.exists(schedule_path):
            results[os.path.basename(schedule_path)] = validate_schedule(
                schedule_path, failure_path
            )

    return results, generate_report(results, output_dir=output_dir)
