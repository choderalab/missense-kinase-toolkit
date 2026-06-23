from mkt.databases.plot_config import (
    ColKinaseColorConfig,
    DataSourceConfig,
    DynamicRangePlotConfig,
    FamilyColorConfig,
    MatplotlibRCConfig,
    MetricsBoxplotConfig,
    OutputConfig,
    PlotDatasetConfig,
    RidgelinePlotConfig,
    SequenceSchematicConfig,
    StackedBarchartConfig,
    VennDiagramConfig,
)


class TestConfigDefaults:
    def test_matplotlib_rc_svg_fonttype(self):
        assert MatplotlibRCConfig().svg_fonttype == "path"

    def test_matplotlib_rc_pdf_fonttype(self):
        assert MatplotlibRCConfig().pdf_fonttype == 42

    def test_matplotlib_rc_text_usetex(self):
        assert MatplotlibRCConfig().text_usetex is False

    def test_family_color_use_kinase_group(self):
        assert FamilyColorConfig().use_kinase_group_colors is True

    def test_col_kinase_color_construct_unaligned(self):
        assert ColKinaseColorConfig().construct_unaligned == [242, 101, 41]

    def test_dynamic_range_bins(self):
        assert DynamicRangePlotConfig().bins == 100

    def test_ridgeline_overlap(self):
        assert RidgelinePlotConfig().overlap == 0.1

    def test_stacked_barchart_figsize_height(self):
        assert StackedBarchartConfig().figsize_height == 7

    def test_venn_diagram_circle_alpha(self):
        assert VennDiagramConfig().circle_alpha == 0.6

    def test_metrics_boxplot_box_widths(self):
        assert MetricsBoxplotConfig().box_widths == 0.6

    def test_sequence_schematic_n_show_start(self):
        assert SequenceSchematicConfig().n_show_start == 40

    def test_data_source_davis_csv(self):
        assert DataSourceConfig().davis_csv == "data/davis_data_processed.csv"

    def test_output_bool_svg(self):
        assert OutputConfig().bool_svg is True


class TestPlotDatasetConfigAggregator:
    def test_nested_matplotlib_rc(self):
        cfg = PlotDatasetConfig()
        assert isinstance(cfg.matplotlib_rc, MatplotlibRCConfig)

    def test_nested_dynamic_range(self):
        cfg = PlotDatasetConfig()
        assert isinstance(cfg.dynamic_range, DynamicRangePlotConfig)

    def test_nested_data_sources(self):
        cfg = PlotDatasetConfig()
        assert isinstance(cfg.data_sources, DataSourceConfig)

    def test_output_bool_png(self):
        assert PlotDatasetConfig().output.bool_png is True


class TestFamilyColorConfigGetColors:
    def test_default_mode_returns_dict_with_tk(self):
        colors = FamilyColorConfig().get_colors()
        assert isinstance(colors, dict)
        assert len(colors) > 0
        assert "TK" in colors

    def test_filtered_families(self):
        cfg = FamilyColorConfig(families=["TK", "Other"])
        colors = cfg.get_colors()
        assert list(colors.keys()) == ["TK", "Other"]

    def test_seaborn_palette_mode(self):
        cfg = FamilyColorConfig(use_kinase_group_colors=False)
        colors = cfg.get_colors()
        assert isinstance(colors, dict)
        assert len(colors) > 0


class TestColKinaseColorConfigAsRGBDict:
    def test_keys(self):
        rgb = ColKinaseColorConfig().as_rgb_dict()
        assert set(rgb.keys()) == {
            "construct_unaligned",
            "klifs_region_aligned",
            "klifs_residues_only",
        }

    def test_values_scaled_0_to_1(self):
        rgb = ColKinaseColorConfig().as_rgb_dict()
        for v in rgb.values():
            assert len(v) == 3
            assert all(0 <= c <= 1 for c in v)

    def test_construct_unaligned_values(self):
        rgb = ColKinaseColorConfig().as_rgb_dict()
        assert abs(rgb["construct_unaligned"][0] - 242 / 255) < 1e-6
        assert abs(rgb["construct_unaligned"][1] - 101 / 255) < 1e-6
        assert abs(rgb["construct_unaligned"][2] - 41 / 255) < 1e-6


class TestPlotDatasetConfigFromYAML:
    def test_overridden_values(self, tmp_path):
        yaml_content = (
            "matplotlib_rc:\n"
            '  svg_fonttype: "none"\n'
            "  pdf_fonttype: 3\n"
            "dynamic_range:\n"
            "  bins: 50\n"
            "  alpha: 0.5\n"
        )
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = PlotDatasetConfig.from_yaml(yaml_file)
        assert cfg.matplotlib_rc.svg_fonttype == "none"
        assert cfg.matplotlib_rc.pdf_fonttype == 3
        assert cfg.dynamic_range.bins == 50
        assert cfg.dynamic_range.alpha == 0.5

    def test_unspecified_fields_keep_defaults(self, tmp_path):
        yaml_content = "dynamic_range:\n  bins: 50\n"
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = PlotDatasetConfig.from_yaml(yaml_file)
        assert cfg.ridgeline.overlap == 0.1
        assert cfg.output.bool_svg is True
