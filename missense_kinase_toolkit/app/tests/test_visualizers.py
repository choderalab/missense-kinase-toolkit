"""Unit tests for the Bokeh sequence-alignment rendering in ``visualizers.py``."""

from bokeh.models import Label
from visualizers import render_alignment_plot

STR_MISSING_COLOR = "#DC143C"
"""str: Crimson used for a missing track's label and residue cells."""


def test_missing_track_cells_match_label():
    """A missing (all "-") track colors every cell crimson, matching its crimson label."""
    plot = render_alignment_plot(
        list_sequences=["AC", "--"],
        list_ids=["present", "missing"],
        list_colors=[["#111111", "#222222"], ["#333333", "#444444"]],
    )

    source = plot.renderers[0].data_source
    assert list(source.data["colors"]) == [
        "#111111",
        "#222222",
        STR_MISSING_COLOR,
        STR_MISSING_COLOR,
    ]

    dict_label_color = {
        obj.text: obj.text_color for obj in plot.center if isinstance(obj, Label)
    }
    assert dict_label_color == {"present": "black", "missing": STR_MISSING_COLOR}
