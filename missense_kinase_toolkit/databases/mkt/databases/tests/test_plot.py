import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mkt.databases.plot import (
    SequenceAlignment,
    _get_klifs_position_colors,
    _map_aligned_to_klifs_colors,
    apply_matplotlib_rc,
    convert_from_percentile,
    convert_to_percentile,
    generate_venn_diagram_dict,
    plot_dynamic_range,
    plot_metrics_boxplot,
    plot_ridgeline,
    plot_stacked_barchart,
    plot_venn_diagram,
)
from mkt.databases.plot_config import FamilyColorConfig, MatplotlibRCConfig

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# pure-computation tests
# ---------------------------------------------------------------------------


class TestConvertPercentile:
    def test_to_percentile(self):
        assert convert_to_percentile(5, orig_max=10) == 50.0

    def test_to_percentile_max(self):
        assert convert_to_percentile(10, orig_max=10) == 100.0

    def test_to_percentile_zero(self):
        assert convert_to_percentile(0, orig_max=10) == 0.0

    def test_from_percentile(self):
        assert convert_from_percentile(50, orig_max=10) == 5.0

    def test_from_percentile_max(self):
        assert convert_from_percentile(100, orig_max=10) == 10.0

    def test_from_percentile_zero(self):
        assert convert_from_percentile(0, orig_max=10) == 0.0

    def test_round_trip(self):
        val = 7.5
        assert convert_from_percentile(convert_to_percentile(val)) == val


class TestGenerateVennDiagramDict:
    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "kinase_name": ["EGFR", "ABL1", "BRAF", "CDK2"],
                "seq_construct_unaligned": ["ACGT", np.nan, "TGCA", "AAAA"],
                "seq_klifs_region_aligned": ["ACGT", "TTTT", np.nan, "AAAA"],
                "seq_klifs_residues_only": [np.nan, "TTTT", "TGCA", "AAAA"],
            }
        )

    def test_keys(self, sample_df):
        result = generate_venn_diagram_dict(sample_df)
        assert set(result.keys()) == {
            "Construct Unaligned",
            "KLIFS Region Aligned",
            "Klifs Residues Only",
        }

    def test_construct_unaligned_members(self, sample_df):
        result = generate_venn_diagram_dict(sample_df)
        assert set(result["Construct Unaligned"]) == {"EGFR", "BRAF", "CDK2"}

    def test_klifs_region_aligned_members(self, sample_df):
        result = generate_venn_diagram_dict(sample_df)
        assert set(result["KLIFS Region Aligned"]) == {"EGFR", "ABL1", "CDK2"}

    def test_klifs_residues_only_members(self, sample_df):
        result = generate_venn_diagram_dict(sample_df)
        assert set(result["Klifs Residues Only"]) == {"ABL1", "BRAF", "CDK2"}


class TestGetKLIFSPositionColors:
    def test_length(self):
        assert len(_get_klifs_position_colors()) == 85

    def test_tuple_format(self):
        colors = _get_klifs_position_colors()
        assert all(isinstance(c, tuple) and len(c) == 2 for c in colors)
        assert all(isinstance(c[0], str) and isinstance(c[1], str) for c in colors)


class TestMapAlignedToKLIFSColors:
    def test_color_mapping(self):
        seq_aligned = "A-BC"
        seq_klifs_only = "A-B"
        dict_aa_colors = {"A": "red", "B": "blue", "C": "green"}
        klifs_pos_colors = [
            ("I", "yellow"),
            ("g.l", "purple"),
            ("II", "orange"),
        ]

        colors = _map_aligned_to_klifs_colors(
            seq_aligned, seq_klifs_only, dict_aa_colors, klifs_pos_colors
        )
        assert len(colors) == 4
        assert colors[0] == "yellow"  # A matched to KLIFS pocket pos 0
        assert colors[1] == "white"  # gap
        assert colors[2] == "orange"  # B matched to KLIFS pocket pos 2
        assert colors[3] == "green"  # C not in KLIFS -> alphabet color


class TestSequenceAlignmentGetColors:
    def test_get_colors(self):
        colors = SequenceAlignment.get_colors(
            ["A", "B", "C"],
            {"A": "red", "B": "green", "C": "blue"},
        )
        assert colors == ["red", "green", "blue"]


class TestApplyMatplotlibRC:
    def test_sets_rcparams(self):
        rc = MatplotlibRCConfig(svg_fonttype="none", pdf_fonttype=3, text_usetex=False)
        apply_matplotlib_rc(rc)
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["pdf.fonttype"] == 3
        assert plt.rcParams["text.usetex"] is False


# ---------------------------------------------------------------------------
# smoke tests (verify execution, not visual output)
# ---------------------------------------------------------------------------


class TestPlotDynamicRangeSmoke:
    def test_produces_output_file(self, tmp_path):
        df_davis = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
        df_pkis2 = pd.DataFrame({"y": [10.0, 50.0, 90.0, 30.0, 70.0] * 20})

        output_path = str(tmp_path / "dynamic_range")
        plot_dynamic_range(df_davis, df_pkis2, output_path)
        plt.close("all")

        saved = list(tmp_path.glob("dynamic_range.*"))
        assert len(saved) >= 1


class TestPlotRidgelineSmoke:
    def test_produces_output_file(self, tmp_path):
        rng = np.random.default_rng(42)
        families = ["TK", "TKL", "STE", "Other"]
        rows = []
        for fam in families:
            for i in range(20):
                rows.append(
                    {
                        "kinase_name": f"{fam}_{i}",
                        "family": fam,
                        "fraction_construct": rng.uniform(0.3, 1.0),
                        "source": "Davis",
                    }
                )
        df = pd.DataFrame(rows)

        output_path = str(tmp_path / "ridgeline")
        plot_ridgeline(df, output_path, family_cfg=FamilyColorConfig())
        plt.close("all")

        saved = list(tmp_path.glob("ridgeline.*"))
        assert len(saved) >= 1


class TestPlotStackedBarchartSmoke:
    def test_produces_output_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "family": ["TK", "TK", "TKL", "TKL"],
                "bool_uniprot2refseq": [True, False, True, False],
                "count": [30, 10, 20, 5],
                "source": ["Davis", "Davis", "Davis", "Davis"],
            }
        )

        output_path = str(tmp_path / "stacked_barchart")
        plot_stacked_barchart(df, output_path, family_cfg=FamilyColorConfig())
        plt.close("all")

        saved = list(tmp_path.glob("stacked_barchart.*"))
        assert len(saved) >= 1


class TestPlotVennDiagramSmoke:
    def test_produces_output_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "kinase_name": ["EGFR", "ABL1", "BRAF", "CDK2", "SRC"],
                "seq_construct_unaligned": ["ACGT", "TTTT", np.nan, "AAAA", "CCCC"],
                "seq_klifs_region_aligned": ["ACGT", np.nan, "TGCA", "AAAA", "CCCC"],
                "seq_klifs_residues_only": [np.nan, "TTTT", "TGCA", "AAAA", "CCCC"],
            }
        )

        output_path = str(tmp_path / "venn_diagram")
        plot_venn_diagram(df, output_path, source_name="Test")
        plt.close("all")

        saved = list(tmp_path.glob("venn_diagram.*"))
        assert len(saved) >= 1


class TestPlotMetricsBoxplotSmoke:
    def test_produces_output_file(self, tmp_path):
        rng = np.random.default_rng(42)
        rows = []
        for col_kinase in [
            "construct_unaligned",
            "klifs_region_aligned",
            "klifs_residues_only",
        ]:
            for fold in range(5):
                rows.append(
                    {
                        "col_kinase": col_kinase,
                        "source": "davis",
                        "fold": fold,
                        "avg_stable_epoch": 10,
                        "mse": rng.normal(0.5, 0.1),
                    }
                )
        df = pd.DataFrame(rows)

        output_path = str(tmp_path / "boxplot")
        plot_metrics_boxplot(df, output_path)
        plt.close("all")

        saved = list(tmp_path.glob("boxplot.*"))
        assert len(saved) >= 1
