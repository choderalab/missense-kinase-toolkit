"""Unit tests for the Bokeh sequence-alignment rendering in ``visualizers.py``."""

from bokeh.models import Label
from visualizers import render_alignment_plot

STR_MISSING_COLOR = "#DC143C"
"""str: Crimson used for a missing track's label and residue text."""


def test_missing_track_text_matches_label():
    """A missing (all "-") track has crimson text and label but keeps its cell colors."""
    plot = render_alignment_plot(
        list_sequences=["AC", "--"],
        list_ids=["present", "missing"],
        list_colors=[["#111111", "#222222"], ["#333333", "#444444"]],
    )

    source = plot.renderers[0].data_source
    assert list(source.data["colors"]) == ["#111111", "#222222", "#333333", "#444444"]
    assert list(source.data["text_colors"]) == [
        "black",
        "black",
        STR_MISSING_COLOR,
        STR_MISSING_COLOR,
    ]

    dict_label_color = {
        obj.text: obj.text_color for obj in plot.center if isinstance(obj, Label)
    }
    assert dict_label_color == {"present": "black", "missing": STR_MISSING_COLOR}
