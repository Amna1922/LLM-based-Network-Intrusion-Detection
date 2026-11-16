"""Tests for CSV and LaTeX table output."""

import pandas as pd

from src.make_tables import write_table


def test_write_table_creates_csv_and_latex(tmp_path):
    """Write one focused table in both supported output formats."""
    frame = pd.DataFrame([{"metric": "macro F1", "value": 0.95}])

    csv_path, tex_path = write_table(
        "table_test",
        frame,
        tmp_path,
        caption="A measured test result.",
    )

    assert pd.read_csv(csv_path).to_dict(orient="records") == [
        {"metric": "macro F1", "value": 0.95}
    ]
    latex = tex_path.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in latex
    assert r"\caption{A measured test result.}" in latex
