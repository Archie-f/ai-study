import base64
import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from rag_notes.models import RetrievalEvalReport, JudgeResult

STYLE_BLOCK = """
<style>
    body {
        font-family: -apple-system, "Segoe UI", sans-serif;
        background: #f4f5f7;
        margin: 0;
        padding: 2rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 16px;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 1.5rem;
    }
    .chart-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 1.5rem;
    }
    .chart-row img {
        width: 100%;
        display: block;
    }
    .card {
        background: #ffffff;
        border: 0.5px solid #d8dbe0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-sizing: border-box;
    }
    .card h2 {
        margin: 0 0 8px;
        font-size: 16px;
    }
    .card .meta {
        margin: 2px 0;
        font-size: 24px;
        color: #6b7280;
    }
</style>
"""


def _bar_chart_to_base64(values: dict[str, float], title: str, color: str = "#2E7FE0") -> str:
    """Render a 0-100% bar chart for a {label: value} mapping (values 0.0-1.0),
    return it as a base64 PNG data URI. No file is written to disk."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    labels = list(values.keys())

    for i, label in enumerate(labels):
        pct = values[label] * 100
        ax.bar(i, pct, color=color)
        ax.text(i, pct + 1.5, f"{pct:.0f}%", ha="center")

    ax.set_title(title)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(label).capitalize() for label in labels])
    ax.set_ylabel("%")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_full_eval_html(
    mode_reports: dict[str, RetrievalEvalReport],
    judge_results: list[JudgeResult],
) -> str:
    """Build a self-contained HTML eval report from already-computed results.

    Args:
        mode_reports: one RetrievalEvalReport per retrieval mode, as returned
            by compare_retrieval_modes()
        judge_results: one JudgeResult per golden-set question, as produced by
            running the full retrieval-generation-judge chain
    Returns:
        A complete, self-contained HTML document string (charts embedded as
        base64 PNGs, no external files referenced) — ready to write to disk
    """
    # --- retrieval side: one card per mode, one MRR chart ---
    mrr_by_mode = {mode: report.mrr for mode, report in mode_reports.items()}
    mrr_chart = _bar_chart_to_base64(mrr_by_mode, title="MRR per retrieval mode")

    # grid-template-columns is set inline per call, since the number of cards
    # depends on how many modes were compared (mode_reports isn't fixed-size) —
    # the .grid class in STYLE_BLOCK provides everything else (width, gap, box model)
    mode_count = len(mode_reports)
    grid_style = f"grid-template-columns: repeat({mode_count}, 1fr);"

    retrieval_cards = []
    for mode, report in mode_reports.items():
        recall_lines = "".join(
            f"<p class='meta'>recall@{k}: {v:.0%}</p>" for k, v in sorted(report.recall_k.items())
        )
        retrieval_cards.append(f"""
        <div class="card">
            <h1>{mode.capitalize()}</h1>
            {recall_lines}
            <p class="meta">MRR: {report.mrr:.2f}</p>
        </div>
        """)

    # --- judge side: pass rate + failure table ---
    total = len(judge_results)
    passed = sum(1 for r in judge_results if r.passed)
    pass_rate = passed / total if total else 0.0
    judge_chart = _bar_chart_to_base64({"judge": pass_rate}, title="Judge pass rate")

    result_rows = []
    for r in judge_results:
        status = "PASS" if r.passed else "FAIL"
        status_color = "#1a7f37" if r.passed else "#cf222e"
        row_bg = "#f0fdf4" if r.passed else "#fef2f2"
        score = f"{r.score:.2f}" if r.score is not None else "None"
        result_rows.append(f"""
        <tr data-status="{'pass' if r.passed else 'fail'}" style="background:{row_bg};">
            <td style="padding:6px; font-weight:600; color:{status_color};">{status}</td>
            <td style="padding:6px;">{r.question}</td>
            <td style="padding:6px;">{r.answer}</td>
            <td style="padding:6px;">{score}</td>
            <td style="padding:6px;">{r.reason}</td>
        </tr>
        """)

    results_block = f"""
        <h2 style="margin-top:2rem;">Judge Results — Per Question</h2>
        <table id="judgeResultsTable" style="width:100%; border-collapse:collapse; font-size:20px;">
            <tr style="background:#1F3864; color:white;">
                <th style="padding:6px; text-align:left;">
                    Result
                    <select id="resultFilter" onchange="filterJudgeResults()"
                            style="margin-left:6px; font-size:12px; font-weight:normal;">
                        <option value="all">All</option>
                        <option value="pass">Pass</option>
                        <option value="fail">Fail</option>
                    </select>
                </th>
                <th style="padding:6px; text-align:left;">Question</th>
                <th style="padding:6px; text-align:left;">Answer</th>
                <th style="padding:6px; text-align:left;">Score</th>
                <th style="padding:6px; text-align:left;">Reason</th>
            </tr>
            {''.join(result_rows)}
        </table>
        <script>
            function filterJudgeResults() {{
                var filter = document.getElementById("resultFilter").value;
                var rows = document.querySelectorAll("#judgeResultsTable tr[data-status]");
                rows.forEach(function(row) {{
                    var status = row.getAttribute("data-status");
                    row.style.display = (filter === "all" || filter === status) ? "" : "none";
                }});
            }}
        </script>
        """

    return f"""
        <html>
            <head><meta charset="UTF-8">{STYLE_BLOCK}</head>
            <body>
                <div style="padding: 1.5rem 0 0.5rem;">
                    <h1 style="margin: 0 0 4px;">rag-notes-qa — Full Eval Report</h1>
                    <p style="margin: 0 0 1.5rem; font-size: 14px; color: #6b7280;">
                        Judge pass rate: {passed}/{total} ({pass_rate:.0%})
                    </p>
                </div>
                <div class="grid" style="{grid_style}">
                    {''.join(retrieval_cards)}
                </div>
                <div class="chart-row">
                    <div><img src="{mrr_chart}"></div>
                    <div><img src="{judge_chart}"></div>
                </div>
                {results_block}
            </body>
        </html>
    """


def save_full_eval_html(
    mode_reports: dict[str, RetrievalEvalReport],
    judge_results: list[JudgeResult],
    reports_dir: Path,
) -> Path:
    """Build the full eval HTML report and write it to reports_dir with a timestamped filename.

    Args:
        mode_reports: one RetrievalEvalReport per retrieval mode, as returned
            by compare_retrieval_modes()
        judge_results: one JudgeResult per golden-set question, as produced by
            running the full retrieval-generation-judge chain
        reports_dir: folder to write the report into (created if missing)
    Returns:
        The path the report was written to
    """
    reports_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = reports_dir / f"eval_{timestamp}.html"
    path.write_text(build_full_eval_html(mode_reports, judge_results), encoding="utf-8")
    return path
