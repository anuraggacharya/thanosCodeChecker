import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



APP_NAME = "Thanos"
THANOS_IMAGE = Path(__file__).parent / "assets" / "thanos.png"

DEFAULT_SELECTED_RULES = "E,W,F,I,B,S,UP,SIM,C4,RET,ARG,PL,PT,ASYNC,RUF"

RUFF_RULE_GROUPS = {
    "S": "Security",
    "E": "Style",
    "W": "Style",
    "F": "Correctness",
    "B": "Bug risk",
    "SIM": "Simplification",
    "C4": "Modernization",
    "UP": "Modernization",
    "I": "Imports",
    "N": "Naming",
    "RET": "Readability",
    "ARG": "Unused arguments",
    "PL": "Pylint-style checks",
    "PT": "Pytest style",
    "D": "Docstrings",
    "ANN": "Type hints",
    "ASYNC": "Async correctness",
    "RUF": "Ruff-specific",
}

SEVERITY_WEIGHTS = {
    "Critical": 12,
    "High": 7,
    "Medium": 4,
    "Low": 2,
    "Info": 0.5,
}

SECURITY_HIGH_CODES = {
    "S102", "S105", "S106", "S107", "S301", "S302", "S310",
    "S324", "S506", "S602", "S603", "S605", "S606", "S607",
    "S608", "S612",
}

st.set_page_config(
    page_title=f"{APP_NAME} — Python & PySpark Quality Platform",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# UI helpers
# -----------------------------

def inject_thanos_watermark():
    if not THANOS_IMAGE.exists():
        st.sidebar.warning("Add assets/thanos.png to display the Thanos watermark.")
        return

    image_b64 = base64.b64encode(THANOS_IMAGE.read_bytes()).decode("utf-8")

    st.markdown(
    f"""
    <style>
    .thanos-watermark {{
        position: fixed;
        right: 1px;
        bottom: 1px;
        width: 350px;
        height: 350px;
        border-radius: 50%;
        object-fit: cover;
        z-index: 999999;
        pointer-events: none;
        opacity: 1;
        # border: 2px solid rgba(139, 126, 156, 0.35);
        # box-shadow: 0 8px 24px rgba(63, 71, 83, 0.12);
        # filter: drop-shadow(0 0 10px rgba(139, 126, 156, 0.22));
    }}

    @media (max-width: 768px) {{
        .thanos-watermark {{
            width: 100px;
            height: 100px;
            right: 10px;
            bottom: 10px;
            opacity: 0.22;
        }}
    }}
    </style>

    <img
        class="thanos-watermark"
        src="data:image/png;base64,{image_b64}"
        alt="Thanos"
    />
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Ruff analysis
# -----------------------------

def parse_rule_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_ruff_toml(
    selected_rules: str,
    ignored_rules: str,
    line_length: int,
    per_file_ignores: dict[str, list[str]],
    max_complexity: int,
) -> str:
    selected = parse_rule_list(selected_rules)
    ignored = parse_rule_list(ignored_rules)

    if not selected:
        selected = ["E", "F", "S"]

    lines = [
        f"line-length = {line_length}",
        "",
        "[lint]",
        "select = [",
    ]

    lines.extend(f'  "{rule}",' for rule in selected)
    lines.append("]")

    if ignored:
        lines.extend(["", "ignore = ["])
        lines.extend(f'  "{rule}",' for rule in ignored)
        lines.append("]")

    if per_file_ignores:
        lines.extend(["", "[lint.per-file-ignores]"])

        for pattern, rules in per_file_ignores.items():
            if not rules:
                continue

            rule_list = ", ".join(f'"{rule}"' for rule in rules)
            lines.append(f'"{pattern}" = [{rule_list}]')

    lines.extend(
        [
            "",
            "[lint.mccabe]",
            f"max-complexity = {max_complexity}",
        ]
    )

    return "\n".join(lines)


def category_for_rule(code: str) -> str:
    code = str(code)

    for prefix, category in RUFF_RULE_GROUPS.items():
        if code.startswith(prefix):
            return category

    return "General"


def infer_ruff_severity(code: str, message: str) -> str:
    code = str(code)
    message_lower = message.lower()

    if code in SECURITY_HIGH_CODES:
        return "Critical"

    if code.startswith("S"):
        return "High"

    if code.startswith(("F", "B")):
        return "Medium"

    if code.startswith(("E", "W", "PL", "ARG")):
        return "Low"

    if "security" in message_lower or "injection" in message_lower:
        return "High"

    return "Info"


def run_ruff_on_files(
    files: list[tuple[str, bytes]],
    selected_rules: str,
    ignored_rules: str,
    line_length: int,
    per_file_ignores: dict[str, list[str]],
    max_complexity: int,
) -> tuple[list[dict], str | None]:
    if shutil.which("ruff") is None:
        return [], "Ruff executable was not found. Install it using `pip install ruff`."

    findings = []

    ruff_toml = build_ruff_toml(
        selected_rules=selected_rules,
        ignored_rules=ignored_rules,
        line_length=line_length,
        per_file_ignores=per_file_ignores,
        max_complexity=max_complexity,
    )

    with tempfile.TemporaryDirectory(prefix="thanos_ruff_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "ruff.toml"
        config_path.write_text(ruff_toml, encoding="utf-8")

        for original_name, content in files:
            safe_name = Path(original_name).name
            target = temp_path / safe_name
            target.write_bytes(content)

        command = [
            "ruff",
            "check",
            "--config",
            str(config_path),
            "--output-format",
            "json",
            str(temp_path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [], "Ruff analysis timed out."

        if result.returncode not in (0, 1):
            return [], result.stderr.strip() or "Ruff failed to analyze the uploaded files."

        try:
            raw_findings = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return [], "Could not parse Ruff JSON output."

        for item in raw_findings:
            code = item.get("code") or "UNKNOWN"
            message = item.get("message") or "No message provided"
            fix = item.get("fix")

            findings.append(
                {
                    "file": Path(item.get("filename", "unknown.py")).name,
                    "line": item.get("location", {}).get("row", 0),
                    "column": item.get("location", {}).get("column", 0),
                    "code": code,
                    "rule": code,
                    "message": message,
                    "category": category_for_rule(code),
                    "severity": infer_ruff_severity(code, message),
                    "fixable": bool(fix),
                    "url": item.get("url", ""),
                }
            )

    return findings, None


# -----------------------------
# Pytest analysis
# -----------------------------

def outcome_to_status(outcome: str) -> str:
    outcome = outcome.lower()

    if outcome == "passed":
        return "Passed"
    if outcome == "failed":
        return "Failed"
    if outcome == "skipped":
        return "Skipped"
    if outcome == "error":
        return "Error"
    if outcome == "xfailed":
        return "Expected failure"
    if outcome == "xpassed":
        return "Unexpected pass"
    return outcome.title()


def status_severity(status: str) -> str:
    if status in ("Failed", "Error"):
        return "Critical"
    if status == "Unexpected pass":
        return "High"
    if status == "Skipped":
        return "Low"
    if status == "Expected failure":
        return "Info"
    return "Info"


def calculate_pytest_score(report: dict) -> int:
    summary = report.get("summary", {})
    total = summary.get("total", 0)

    if total == 0:
        return 100

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    error = summary.get("error", 0)
    skipped = summary.get("skipped", 0)
    xfailed = summary.get("xfailed", 0)
    xpassed = summary.get("xpassed", 0)

    penalty = (
        failed * 12
        + error * 12
        + xpassed * 7
        + skipped * 2
        + xfailed * 0.5
    )

    return max(0, round(100 - (penalty / total) * 100))


def run_pytest_on_files(
    files: list[tuple[str, bytes]],
    pytest_args: list[str],
    selected_tests: str,
    deselect_tests: str,
    marker_expr: str,
    timeout_seconds: int | None,
    enable_coverage: bool,
    coverage_target: str,
) -> tuple[dict | None, list[dict], str | None]:
    if shutil.which("pytest") is None:
        return None, [], "Pytest executable was not found. Install it using `pip install pytest`."

    try:
        import pytest_jsonreport  # noqa: F401
    except ImportError:
        return None, [], (
            "The `pytest-json-report` plugin is required. "
            "Install it using `pip install pytest-json-report`."
        )

    with tempfile.TemporaryDirectory(prefix="thanos_pytest_") as temp_dir:
        temp_path = Path(temp_dir)
        report_path = temp_path / "thanos_pytest_report.json"

        for original_name, content in files:
            safe_name = Path(original_name).name
            target = temp_path / safe_name
            target.write_bytes(content)

        command = [
            "pytest",
            str(temp_path),
            "--json-report",
            f"--json-report-file={report_path}",
            "-p",
            "no:cacheprovider",
        ]

        command.extend(pytest_args)

        if selected_tests:
            command.extend(["-k", selected_tests])

        if deselect_tests:
            for test_id in parse_rule_list(deselect_tests):
                command.extend(["--deselect", test_id])

        if marker_expr:
            command.extend(["-m", marker_expr])

        if timeout_seconds:
            command.extend(["--timeout", str(timeout_seconds)])

        if enable_coverage:
            command.extend(["--cov", coverage_target or "."])

        try:
            result = subprocess.run(
                command,
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, [], "Pytest run timed out."

        if not report_path.exists():
            return None, [], (
                result.stderr.strip()
                or result.stdout.strip()
                or "Pytest did not generate a JSON report."
            )

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, [], "Could not parse the Pytest JSON report."

        tests = []

        for test in report.get("tests", []):
            outcome = test.get("outcome", "unknown")
            status = outcome_to_status(outcome)
            call = test.get("call", {}) or {}
            setup = test.get("setup", {}) or {}
            teardown = test.get("teardown", {}) or {}

            longrepr = (
                call.get("longrepr")
                or setup.get("longrepr")
                or teardown.get("longrepr")
                or ""
            )

            crash = call.get("crash") or {}

            tests.append(
                {
                    "test_id": test.get("nodeid", "unknown"),
                    "file": Path(test.get("nodeid", "unknown.py")).name.split("::")[0],
                    "status": status,
                    "severity": status_severity(status),
                    "duration_seconds": round(test.get("duration", 0.0), 4),
                    "line": crash.get("line", ""),
                    "error_message": str(longrepr)[:4000],
                    "outcome": outcome,
                }
            )

        return report, tests, None


# -----------------------------
# Scores
# -----------------------------

def calculate_score(findings: list[dict]) -> int:
    if not findings:
        return 100

    penalty = sum(
        SEVERITY_WEIGHTS.get(item["severity"], 1)
        for item in findings
    )

    return max(0, round(100 - penalty))


def score_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Needs improvement"
    if score >= 40:
        return "Poor"
    return "Critical risk"


# -----------------------------
# Report components
# -----------------------------

def render_metric_cards(metrics: list[tuple[str, str, str]]):
    columns = st.columns(len(metrics))

    for column, (label, value, delta) in zip(columns, metrics):
        column.metric(label, value, delta)


def render_score_gauge(score: int, title: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7C3AED"},
                "steps": [
                    {"range": [0, 40], "color": "#FEE2E2"},
                    {"range": [40, 60], "color": "#FFEDD5"},
                    {"range": [60, 75], "color": "#FEF9C3"},
                    {"range": [75, 90], "color": "#DCFCE7"},
                    {"range": [90, 100], "color": "#BBF7D0"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_status_charts(tests: list[dict], title_prefix: str = ""):
    if not tests:
        st.info("No test results available.")
        return

    df = pd.DataFrame(tests)

    col1, col2 = st.columns(2)

    status_order = [
        "Passed",
        "Failed",
        "Error",
        "Skipped",
        "Expected failure",
        "Unexpected pass",
    ]

    status_counts = (
        df["status"]
        .value_counts()
        .reindex(status_order)
        .fillna(0)
    )

    fig_status = px.bar(
        x=status_counts.index,
        y=status_counts.values,
        color=status_counts.index,
        title=f"{title_prefix}Test outcomes",
        labels={"x": "Status", "y": "Count"},
        color_discrete_map={
            "Passed": "#16A34A",
            "Failed": "#DC2626",
            "Error": "#B91C1C",
            "Skipped": "#CA8A04",
            "Expected failure": "#64748B",
            "Unexpected pass": "#EA580C",
        },
    )

    fig_status.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    col1.plotly_chart(fig_status, use_container_width=True)

    file_counts = (
        df.groupby("file")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    fig_files = px.bar(
        x=file_counts.values,
        y=file_counts.index,
        orientation="h",
        title=f"{title_prefix}Tests by file",
        labels={"x": "Test count", "y": "File"},
        color_discrete_sequence=["#7C3AED"],
    )

    fig_files.update_layout(
        yaxis={"autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    col2.plotly_chart(fig_files, use_container_width=True)

    slowest = df.sort_values("duration_seconds", ascending=False).head(10)

    fig_duration = px.bar(
        slowest,
        x="duration_seconds",
        y="test_id",
        orientation="h",
        title=f"{title_prefix}Slowest tests",
        labels={"duration_seconds": "Duration (seconds)", "test_id": "Test"},
        color="status",
        color_discrete_map={
            "Passed": "#16A34A",
            "Failed": "#DC2626",
            "Error": "#B91C1C",
            "Skipped": "#CA8A04",
        },
    )

    fig_duration.update_layout(
        yaxis={"autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    st.plotly_chart(fig_duration, use_container_width=True)


def render_download_report(
    payload: dict,
    file_base_name: str,
):
    json_bytes = io.BytesIO(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download JSON report",
        data=json_bytes.getvalue(),
        file_name=f"{file_base_name}.json",
        mime="application/json",
        use_container_width=True,
    )


# -----------------------------
# Ruff tab
# -----------------------------

def render_ruff_tab(uploaded_files: list):
    st.subheader("🧹 Ruff static analysis")
    st.caption("Security, correctness, style, complexity, and maintainability checks.")

    with st.expander("⚙️ Ruff configuration", expanded=False):
        col1, col2 = st.columns(2)

        selected_rules = st.text_area(
            "Enabled rule groups",
            value=DEFAULT_SELECTED_RULES,
            height=100,
            help="Comma-separated Ruff rule prefixes or codes.",
        )

        ignored_rules = st.text_area(
            "Ignored rules",
            value="E501",
            height=80,
            help="Comma-separated rule codes to disable globally.",
        )

        line_length = col1.number_input(
            "Line length",
            min_value=40,
            max_value=200,
            value=100,
        )

        max_complexity = col2.number_input(
            "Max complexity",
            min_value=1,
            max_value=50,
            value=12,
        )

        st.markdown("#### Per-file exceptions")

        ignore_tests = st.checkbox(
            "Relax rules for tests",
            value=True,
            help="Ignores S101 and PLR2004 in tests/*.",
        )

        ignore_scripts = st.checkbox(
            "Relax rules for scripts",
            value=False,
            help="Ignores T201 and INP001 in scripts/*.",
        )

        per_file_ignores: dict[str, list[str]] = {}

        if ignore_tests:
            per_file_ignores["tests/*"] = ["S101", "PLR2004"]

        if ignore_scripts:
            per_file_ignores["scripts/*"] = ["T201", "INP001"]

        custom_pattern = st.text_input(
            "Custom file pattern",
            placeholder="pyspark_jobs/*",
        )

        custom_rules = st.text_input(
            "Rules to ignore for that pattern",
            placeholder="E402, S101",
        )

        if custom_pattern and custom_rules:
            per_file_ignores[custom_pattern] = parse_rule_list(custom_rules)

        uploaded_ruff_config = st.file_uploader(
            "Upload custom ruff.toml",
            type=["toml"],
            help="If uploaded, this overrides the generated configuration.",
            key="ruff_config_uploader",
        )

        with st.expander("Preview generated ruff.toml"):
            if uploaded_ruff_config:
                st.code(
                    uploaded_ruff_config.getvalue().decode("utf-8", errors="replace"),
                    language="toml",
                )
            else:
                st.code(
                    build_ruff_toml(
                        selected_rules=selected_rules,
                        ignored_rules=ignored_rules,
                        line_length=int(line_length),
                        per_file_ignores=per_file_ignores,
                        max_complexity=int(max_complexity),
                    ),
                    language="toml",
                )

    if st.button(
        "🚀 Run Ruff analysis",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        files = [(file.name, file.getvalue()) for file in uploaded_files]

        with st.spinner("Running Ruff static analysis..."):
            findings, error = run_ruff_on_files(
                files=files,
                selected_rules=selected_rules,
                ignored_rules=ignored_rules,
                line_length=int(line_length),
                per_file_ignores=per_file_ignores,
                max_complexity=int(max_complexity),
            )

        if error:
            st.error(error)
            return

        st.session_state["ruff_findings"] = findings
        st.session_state["ruff_score"] = calculate_score(findings)

    findings = st.session_state.get("ruff_findings", [])
    score = st.session_state.get("ruff_score", 100)

    if not findings:
        st.success("No Ruff issues found in the last analysis.")
        render_score_gauge(100, "Ruff Code Health Score")
        return

    security_count = sum(
        1 for item in findings if item["category"] == "Security"
    )
    fixable_count = sum(
        1 for item in findings if item["fixable"]
    )
    critical_high_count = sum(
        1 for item in findings
        if item["severity"] in ("Critical", "High")
    )

    render_metric_cards(
        [
            ("Ruff score", f"{score}/100", score_label(score)),
            ("Total issues", str(len(findings)), ""),
            ("Security issues", str(security_count), ""),
            ("Auto-fixable", str(fixable_count), ""),
            ("Critical / High", str(critical_high_count), ""),
        ]
    )

    col_chart, col_summary = st.columns([1.1, 1])

    with col_chart:
        render_score_gauge(score, "Ruff Code Health Score")

    with col_summary:
        st.subheader("Executive summary")
        st.markdown(
            f"""
            - **Total issues:** {len(findings)}
            - **Security findings:** {security_count}
            - **Auto-fixable findings:** {fixable_count}
            - **Critical / High findings:** {critical_high_count}
            - **Overall assessment:** **{score_label(score)}**
            """
        )

    st.divider()

    df = pd.DataFrame(findings)

    col1, col2 = st.columns(2)

    severity_order = ["Critical", "High", "Medium", "Low", "Info"]
    severity_counts = (
        df["severity"]
        .value_counts()
        .reindex(severity_order)
        .fillna(0)
    )

    fig_severity = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        color=severity_counts.index,
        title="Ruff issues by severity",
        labels={"x": "Severity", "y": "Count"},
        color_discrete_map={
            "Critical": "#DC2626",
            "High": "#EA580C",
            "Medium": "#CA8A04",
            "Low": "#0284C7",
            "Info": "#64748B",
        },
    )

    fig_severity.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    col1.plotly_chart(fig_severity, use_container_width=True)

    category_counts = df["category"].value_counts()

    fig_category = px.pie(
        names=category_counts.index,
        values=category_counts.values,
        hole=0.55,
        title="Ruff issues by category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig_category.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
    )

    col2.plotly_chart(fig_category, use_container_width=True)

    security_findings = [
        item for item in findings if item["category"] == "Security"
    ]

    st.subheader("🛡️ Security findings")

    if not security_findings:
        st.success("No security-related Ruff findings were detected.")
    else:
        st.error(
            f"{len(security_findings)} potential security issue(s) detected."
        )

        for finding in security_findings:
            with st.expander(
                f"{finding['severity']} — {finding['code']} — "
                f"{finding['file']}:{finding['line']}"
            ):
                st.markdown(f"**Message:** {finding['message']}")
                st.markdown(f"**Category:** {finding['category']}")
                st.markdown(
                    f"**Auto-fixable:** {'Yes' if finding['fixable'] else 'No'}"
                )

                if finding.get("url"):
                    st.markdown(
                        f"[Ruff rule documentation]({finding['url']})"
                    )

    st.subheader("🛠️ Auto-fixable issues")

    fixable = [item for item in findings if item["fixable"]]

    if fixable:
        st.success(
            f"{len(fixable)} issue(s) can potentially be fixed automatically by Ruff."
        )

        st.dataframe(
            pd.DataFrame(fixable)[
                ["severity", "file", "line", "code", "message"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.code(
            "ruff check --fix .\n\n"
            "# Include potentially unsafe fixes:\n"
            "ruff check --fix --unsafe-fixes .",
            language="bash",
        )
    else:
        st.info("No automatically fixable issues were detected.")

    st.subheader("📋 All Ruff findings")

    st.dataframe(
        df[
            [
                "severity",
                "file",
                "line",
                "code",
                "category",
                "fixable",
                "message",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "severity": st.column_config.TextColumn("Severity"),
            "file": st.column_config.TextColumn("File"),
            "line": st.column_config.NumberColumn("Line"),
            "code": st.column_config.TextColumn("Rule"),
            "category": st.column_config.TextColumn("Category"),
            "fixable": st.column_config.CheckboxColumn("Auto-fixable"),
            "message": st.column_config.TextColumn("Issue", width="large"),
        },
    )

    st.divider()

    report_payload = {
        "app": APP_NAME,
        "analysis": "Ruff Analysis",
        "generated_at": datetime.now().isoformat(),
        "overall_score": score,
        "score_label": score_label(score),
        "total_issues": len(findings),
        "security_issues": security_count,
        "fixable_issues": fixable_count,
        "findings": findings,
    }

    render_download_report(report_payload, "thanos_ruff_report")


# -----------------------------
# Pytest tab
# -----------------------------

def render_pytest_tab(uploaded_files: list):
    st.subheader("🧪 Pytest test report")
    st.caption("Run uploaded pytest files and inspect outcomes, failures, durations, and coverage.")

    with st.expander("⚙️ Pytest configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        verbosity = col1.selectbox(
            "Verbosity",
            options=["-q", "default", "-v", "-vv"],
            index=1,
        )

        traceback_mode = col2.selectbox(
            "Traceback style",
            options=["short", "long", "auto", "line", "native", "no"],
            index=0,
        )

        max_failures = col3.number_input(
            "Stop after N failures",
            min_value=0,
            max_value=100,
            value=0,
            help="0 means run all tests.",
        )

        col4, col5, col6 = st.columns(3)

        stop_first_failure = col4.checkbox(
            "Stop at first failure (-x)",
            value=False,
        )

        show_locals = col5.checkbox(
            "Show local variables in tracebacks",
            value=False,
        )

        disable_warnings = col6.checkbox(
            "Disable warnings summary",
            value=False,
        )

        col7, col8, col9 = st.columns(3)

        run_durations = col7.number_input(
            "Show N slowest tests",
            min_value=0,
            max_value=50,
            value=10,
            help="0 disables the slowest-test summary.",
        )

        timeout_seconds = col8.number_input(
            "Per-test timeout (seconds)",
            min_value=0,
            max_value=3600,
            value=0,
            help="Requires pytest-timeout. 0 disables timeout.",
        )

        random_order = col9.checkbox(
            "Random test order",
            value=False,
            help="Requires pytest-random-order.",
        )

        st.markdown("#### Test selection")

        col10, col11 = st.columns(2)

        selected_tests = col10.text_input(
            "Select tests by expression (-k)",
            placeholder="login and not slow",
        )

        marker_expr = col11.text_input(
            "Select tests by marker (-m)",
            placeholder="smoke or not slow",
        )

        deselect_tests = st.text_input(
            "Deselect test IDs",
            placeholder="test_app.py::test_login, test_app.py::test_payment",
        )

        st.markdown("#### Output and coverage")

        col12, col13 = st.columns(2)

        enable_coverage = col12.checkbox(
            "Enable coverage",
            value=False,
            help="Requires pytest-cov.",
        )

        coverage_target = col13.text_input(
            "Coverage target",
            value=".",
            help="File, folder, or module to measure.",
        )

        include_captured_output = st.checkbox(
            "Include captured stdout/stderr in report",
            value=True,
        )

        include_tracebacks = st.checkbox(
            "Include tracebacks in report",
            value=True,
        )

        include_logs = st.checkbox(
            "Include captured logs in report",
            value=True,
        )

        uploaded_pytest_config = st.file_uploader(
            "Upload custom pytest.ini / setup.cfg / pyproject.toml",
            type=["ini", "cfg", "toml"],
            help="If uploaded, it is placed in the temporary test directory before execution.",
            key="pytest_config_uploader",
        )

    if st.button(
        "🧪 Run Pytest tests",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        pytest_args: list[str] = []

        if verbosity == "-q":
            pytest_args.append("-q")
        elif verbosity == "-v":
            pytest_args.append("-v")
        elif verbosity == "-vv":
            pytest_args.append("-vv")

        pytest_args.extend(["--tb", traceback_mode])

        if max_failures > 0:
            pytest_args.extend(["--maxfail", str(int(max_failures))])

        if stop_first_failure:
            pytest_args.append("-x")

        if show_locals:
            pytest_args.append("-l")

        if disable_warnings:
            pytest_args.append("-W")
            pytest_args.append("ignore")

        if run_durations > 0:
            pytest_args.extend(["--durations", str(int(run_durations))])

        if random_order:
            pytest_args.append("-p")
            pytest_args.append("no:randomly")

        omit_fields = []

        if not include_captured_output:
            omit_fields.append("streams")

        if not include_tracebacks:
            omit_fields.append("traceback")

        if not include_logs:
            omit_fields.append("log")

        if omit_fields:
            pytest_args.extend(
                [
                    "--json-report-omit",
                    " ".join(omit_fields),
                ]
            )

        report, tests, error = run_pytest_on_files(
            files=[(file.name, file.getvalue()) for file in uploaded_files],
            pytest_args=pytest_args,
            selected_tests=selected_tests,
            deselect_tests=deselect_tests,
            marker_expr=marker_expr,
            timeout_seconds=int(timeout_seconds) if timeout_seconds else None,
            enable_coverage=enable_coverage,
            coverage_target=coverage_target,
        )

        if error:
            st.error(error)
            return

        st.session_state["pytest_report"] = report
        st.session_state["pytest_tests"] = tests
        st.session_state["pytest_score"] = calculate_pytest_score(report)

    report = st.session_state.get("pytest_report")
    tests = st.session_state.get("pytest_tests", [])
    score = st.session_state.get("pytest_score", 100)

    if not report:
        st.info("Run Pytest to generate the test report.")
        return

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    error_count = summary.get("error", 0)
    skipped = summary.get("skipped", 0)
    xfailed = summary.get("xfailed", 0)
    xpassed = summary.get("xpassed", 0)

    pass_rate = round((passed / total) * 100, 1) if total else 0.0
    total_duration = round(report.get("duration", 0.0), 3)

    render_metric_cards(
        [
            ("Pytest score", f"{score}/100", score_label(score)),
            ("Pass rate", f"{pass_rate}%", ""),
            ("Passed", str(passed), ""),
            ("Failed", str(failed), ""),
            ("Errors", str(error_count), ""),
        ]
    )

    col_chart, col_summary = st.columns([1.1, 1])

    with col_chart:
        render_score_gauge(score, "Pytest Health Score")

    with col_summary:
        st.subheader("Executive summary")

        st.markdown(
            f"""
            - **Total tests:** {total}
            - **Passed:** {passed}
            - **Failed:** {failed}
            - **Errors:** {error_count}
            - **Skipped:** {skipped}
            - **Expected failures:** {xfailed}
            - **Unexpected passes:** {xpassed}
            - **Total duration:** {total_duration} seconds
            - **Overall assessment:** **{score_label(score)}**
            """
        )

        if failed == 0 and error_count == 0 and total > 0:
            st.success("All collected tests passed.")
        elif failed or error_count:
            st.error("Test failures or collection errors were detected.")
        else:
            st.warning("No failures, but some tests were skipped or not executed.")

    st.divider()
    render_status_charts(tests, "Pytest — ")

    failed_tests = [
        test for test in tests
        if test["status"] in ("Failed", "Error")
    ]

    st.subheader("❌ Failed tests and errors")

    if not failed_tests:
        st.success("No failed tests or collection errors.")
    else:
        for test in failed_tests:
            with st.expander(
                f"{test['status']} — {test['test_id']}"
            ):
                st.markdown(f"**Severity:** {test['severity']}")
                st.markdown(f"**Duration:** {test['duration_seconds']} seconds")

                if test["line"]:
                    st.markdown(f"**Failure line:** {test['line']}")

                if test["error_message"]:
                    st.code(test["error_message"], language="python")

    st.subheader("⏱️ Test durations")

    duration_df = pd.DataFrame(tests)

    if not duration_df.empty:
        st.dataframe(
            duration_df[
                [
                    "status",
                    "test_id",
                    "duration_seconds",
                    "severity",
                ]
            ].sort_values(
                "duration_seconds",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "test_id": st.column_config.TextColumn("Test", width="large"),
                "duration_seconds": st.column_config.NumberColumn(
                    "Duration (s)",
                    format="%.4f",
                ),
                "severity": st.column_config.TextColumn("Severity"),
            },
        )

    if enable_coverage:
        st.subheader("📈 Coverage")

        coverage = report.get("coverage", {})

        if coverage:
            percent = coverage.get("percent", 0)
            covered = coverage.get("covered_lines", 0)
            total_lines = coverage.get("num_statements", 0)
            missing = coverage.get("missing_lines", 0)

            col_cov1, col_cov2, col_cov3, col_cov4 = st.columns(4)

            col_cov1.metric("Coverage", f"{percent}%")
            col_cov2.metric("Covered lines", covered)
            col_cov3.metric("Total lines", total_lines)
            col_cov4.metric("Missing lines", missing)

            fig_coverage = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percent,
                    title={"text": "Coverage"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#16A34A"},
                        "steps": [
                            {"range": [0, 50], "color": "#FEE2E2"},
                            {"range": [50, 75], "color": "#FEF9C3"},
                            {"range": [75, 90], "color": "#DCFCE7"},
                            {"range": [90, 100], "color": "#BBF7D0"},
                        ],
                    },
                )
            )

            fig_coverage.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#111827",
            )

            st.plotly_chart(fig_coverage, use_container_width=True)
        else:
            st.warning(
                "Coverage data was not returned. Ensure `pytest-cov` is installed "
                "and the coverage target is valid."
            )

    st.subheader("📋 All test results")

    if tests:
        st.dataframe(
            pd.DataFrame(tests)[
                [
                    "status",
                    "test_id",
                    "duration_seconds",
                    "severity",
                    "error_message",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "test_id": st.column_config.TextColumn("Test", width="large"),
                "duration_seconds": st.column_config.NumberColumn("Duration (s)"),
                "severity": st.column_config.TextColumn("Severity"),
                "error_message": st.column_config.TextColumn("Failure / Error", width="large"),
            },
        )

    st.divider()

    report_payload = {
        "app": APP_NAME,
        "analysis": "Pytest Report",
        "generated_at": datetime.now().isoformat(),
        "pytest_score": score,
        "score_label": score_label(score),
        "summary": summary,
        "total_duration_seconds": total_duration,
        "pass_rate": pass_rate,
        "tests": tests,
        "raw_report": report,
    }

    render_download_report(report_payload, "thanos_pytest_report")


# -----------------------------
# Sidebar
# -----------------------------

def render_sidebar():
    with st.sidebar:
        st.header(f"🟣 {APP_NAME}")
        st.markdown(
            "Upload Python or PySpark scripts, then use the Ruff and Pytest tabs."
        )

        st.divider()
        st.subheader("Ruff scoring")

        st.markdown(
            """
            | Severity | Penalty |
            |---|---:|
            | Critical | 12 |
            | High | 7 |
            | Medium | 4 |
            | Low | 2 |
            | Info | 0.5 |
            """
        )

        st.divider()
        st.subheader("Pytest scoring")

        st.markdown(
            """
            The Pytest score is based on the proportion of passed, failed,
            skipped, expected-failure, and unexpected-pass tests.
            """
        )

        st.divider()
        st.caption(
            "Ruff performs static lint/security analysis. "
            "Pytest executes tests and reports runtime results."
        )


# -----------------------------
# Main app
# -----------------------------

def main():
    inject_thanos_watermark()
    render_sidebar()

    st.title(f"🟣 {APP_NAME}")
    st.caption(
        "Python and PySpark code-quality, security, and test-reporting platform."
    )

    uploaded_files = st.file_uploader(
        "Upload Python, PySpark, or Pytest files",
        type=["py"],
        accept_multiple_files=True,
        help="Upload source files and/or test files. Both tabs use these uploads.",
    )

    if not uploaded_files:
        st.info("Upload Python, PySpark, or Pytest `.py` files to begin.")
        return

    st.success(f"{len(uploaded_files)} file(s) ready for analysis.")

    ruff_tab, pytest_tab = st.tabs(
        [
            "🧹 Ruff Analysis",
            "🧪 Pytest Report",
        ]
    )

    with ruff_tab:
        render_ruff_tab(uploaded_files)

    with pytest_tab:
        render_pytest_tab(uploaded_files)


if __name__ == "__main__":
    main()