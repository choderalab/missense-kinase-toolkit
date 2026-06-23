import logging
import os

import pandas as pd
import pytest
from mkt.databases import sasa, utils
from mkt.schema.io_utils import deserialize_kinase_dict

# AKT1 kinase domain (P31749) spans UniProt residues 142-416 in the KinCore CIF
AKT1_START = 142
AKT1_END = 416
AKT1_N_RES = AKT1_END - AKT1_START + 1

EXPECTED_COLS = ["hgnc_name", "method", "uniprot_idx", "residue", "resname", "sasa"]


@pytest.fixture(scope="module")
def akt1_kinase():
    """Load the shipped AKT1 KinaseInfo object (offline, from package data)."""
    return deserialize_kinase_dict(list_ids=["AKT1"], bool_verbose=False)["AKT1"]


@pytest.fixture(scope="module")
def akt1_no_cif(akt1_kinase):
    """AKT1 object with its KinCore data stripped (no CIF structure)."""
    return akt1_kinase.model_copy(update={"kincore": None})


def _run_df(calc):
    """Run a ResidueSASA and return its cached ``df`` (``run`` returns None)."""
    calc.run()
    return calc.df


class TestConvertMmcifdict2Structure:
    def test_residues_numbered_by_uniprot(self, akt1_kinase):
        """auth_seq_id numbering means residue ids are UniProt positions."""
        structure = utils.convert_mmcifdict2structure(akt1_kinase.kincore.cif.cif)
        list_res = list(structure.get_residues())
        assert len(list_res) == AKT1_N_RES
        assert list_res[0].id[1] == AKT1_START
        assert list_res[-1].id[1] == AKT1_END


class TestResidueSASAConfig:
    def test_requires_a_backend(self, akt1_kinase):
        with pytest.raises(ValueError):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_biopython=False,
                bool_pymol=False,
            )

    def test_list_methods_reflects_flags(self, akt1_kinase):
        calc = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase},
            bool_biopython=True,
            bool_pymol=True,
        )
        assert calc.list_methods == ["biopython", "pymol"]

    def test_df_is_none_before_run(self, akt1_kinase):
        calc = sasa.ResidueSASA(dict_kinase={"AKT1": akt1_kinase})
        assert calc.df is None

    def test_list_ids_loads_offline(self):
        """list_ids deserializes the shipped objects when no dict is passed."""
        calc = sasa.ResidueSASA(list_ids=["AKT1"])
        assert set(calc.dict_kinase) == {"AKT1"}


class TestResidueSASABiopython:
    @pytest.fixture(scope="class")
    def df_sasa(self, akt1_kinase):
        return _run_df(
            sasa.ResidueSASA(dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False)
        )

    def test_run_caches_on_df(self, akt1_kinase):
        calc = sasa.ResidueSASA(dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False)
        assert calc.df is None
        calc.run()
        assert isinstance(calc.df, pd.DataFrame)
        assert len(calc.df) == AKT1_N_RES

    def test_columns(self, df_sasa):
        assert list(df_sasa.columns) == EXPECTED_COLS + ["rsa"]

    def test_one_row_per_residue(self, df_sasa):
        assert len(df_sasa) == AKT1_N_RES
        assert (df_sasa["method"] == "biopython").all()
        assert (df_sasa["hgnc_name"] == "AKT1").all()

    def test_uniprot_indexing(self, df_sasa):
        assert df_sasa["uniprot_idx"].iloc[0] == AKT1_START
        assert df_sasa["uniprot_idx"].iloc[-1] == AKT1_END

    def test_sorted_and_unique(self, df_sasa):
        assert df_sasa["uniprot_idx"].is_monotonic_increasing
        assert df_sasa["uniprot_idx"].is_unique

    def test_sasa_nonnegative(self, df_sasa):
        assert (df_sasa["sasa"] >= 0).all()

    def test_rsa_in_expected_range(self, df_sasa):
        # rsa = sasa / Tien max-ASA; chain termini can slightly exceed 1
        assert df_sasa["rsa"].between(0, 1.2).all()

    def test_hydrogens_excluded_by_default(self, akt1_kinase):
        """Including hydrogens should change per-residue SASA values."""
        df_no_h = _run_df(
            sasa.ResidueSASA(dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False)
        )
        # bool_relative must be False with explicit H (rSASA is heavy-atom only)
        df_with_h = _run_df(
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_pymol=False,
                bool_include_hydrogens=True,
                bool_relative=False,
            )
        )
        assert not df_no_h["sasa"].equals(df_with_h["sasa"])

    def test_no_relative_omits_rsa(self, akt1_kinase):
        df = _run_df(
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_pymol=False,
                bool_relative=False,
            )
        )
        assert "rsa" not in df.columns

    def test_skips_kinase_without_cif(self, akt1_kinase, akt1_no_cif):
        """Kinases without a CIF are skipped from the long-format output."""
        calc = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase, "NOCIF": akt1_no_cif},
            bool_pymol=False,
        )
        df = _run_df(calc)
        assert set(df["hgnc_name"].unique()) == {"AKT1"}
        assert len(df) == AKT1_N_RES

    def test_all_missing_cif_returns_empty(self, akt1_no_cif):
        df = _run_df(
            sasa.ResidueSASA(dict_kinase={"NOCIF": akt1_no_cif}, bool_pymol=False)
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestResidueSASAPymol:
    def test_pymol_matches_biopython(self, akt1_kinase):
        """PyMOL dot_solvent SASA agrees with Shrake-Rupley; both run together."""
        pytest.importorskip("pymol2")
        df = _run_df(
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_biopython=True,
                bool_pymol=True,
            )
        )
        assert set(df["method"].unique()) == {"biopython", "pymol"}
        assert len(df) == 2 * AKT1_N_RES

        df_bio = df[df["method"] == "biopython"]
        df_pml = df[df["method"] == "pymol"]
        df_merge = df_bio.merge(
            df_pml,
            on=["hgnc_name", "uniprot_idx", "residue", "resname"],
            suffixes=("_bio", "_pml"),
        )
        assert len(df_merge) == AKT1_N_RES
        assert df_merge["sasa_bio"].corr(df_merge["sasa_pml"]) > 0.99


class TestPymolUnavailable:
    """Behavior when pymol2 cannot be imported (e.g. broken/missing wheel)."""

    @pytest.fixture
    def _force_pymol_missing(self, monkeypatch):
        monkeypatch.setattr(
            sasa.ResidueSASA, "_pymol_available", staticmethod(lambda: False)
        )

    def test_falls_back_to_biopython(self, akt1_kinase, _force_pymol_missing):
        """Default (both backends) drops PyMOL and runs Bio.PDB only."""
        calc = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase},
            bool_biopython=True,
            bool_pymol=True,
        )
        assert calc._resolve_methods() == ["biopython"]
        df = _run_df(calc)
        assert set(df["method"].unique()) == {"biopython"}
        assert len(df) == AKT1_N_RES

    def test_pymol_only_raises(self, akt1_kinase, _force_pymol_missing):
        """PyMOL as the sole backend raises a clear ImportError."""
        calc = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase},
            bool_biopython=False,
            bool_pymol=True,
        )
        with pytest.raises(ImportError):
            calc.run()


class TestSASAConfigs:
    """Config presets, from_dataclass construction, and compatibility guards."""

    def test_from_dataclass_with_standard_preset(self, akt1_kinase):
        calc = sasa.ResidueSASA.from_dataclass(
            sasa.StandardSASAConfigs.BIOPYTHON_HEAVY,
            dict_kinase={"AKT1": akt1_kinase},
        )
        assert calc.list_methods == ["biopython"]
        assert calc.n_points == 960  # preset raises sampling for convergence

    def test_from_dataclass_with_dataclass_instance(self, akt1_kinase):
        calc = sasa.ResidueSASA.from_dataclass(
            sasa.PyMOLHeavyConfig(),
            dict_kinase={"AKT1": akt1_kinase},
        )
        assert calc.list_methods == ["pymol"]
        assert calc.dot_density == 4

    def test_from_dataclass_forwards_list_ids(self):
        calc = sasa.ResidueSASA.from_dataclass(
            sasa.StandardSASAConfigs.BIOPYTHON_HEAVY, list_ids=["AKT1"]
        )
        assert set(calc.dict_kinase) == {"AKT1"}

    @pytest.mark.parametrize("preset", list(sasa.StandardSASAConfigs))
    def test_all_standard_presets_construct(self, akt1_kinase, preset):
        """Every shipped preset builds a valid (non-empty-backend) calculator."""
        calc = sasa.ResidueSASA.from_dataclass(
            preset, dict_kinase={"AKT1": akt1_kinase}
        )
        assert len(calc.list_methods) >= 1

    @pytest.mark.parametrize(
        "preset",
        [
            sasa.StandardSASAConfigs.BIOPYTHON_HYDROGEN,
            sasa.StandardSASAConfigs.PYMOL_HYDROGEN,
            sasa.StandardSASAConfigs.CROSS_VALIDATION_HYDROGEN,
        ],
    )
    def test_hydrogen_presets_enable_h_and_disable_relative(self, akt1_kinase, preset):
        """*_HYDROGEN presets are all-atom (H on) with rSASA disabled."""
        calc = sasa.ResidueSASA.from_dataclass(
            preset, dict_kinase={"AKT1": akt1_kinase}
        )
        assert calc.bool_include_hydrogens is True
        assert calc.bool_relative is False

    def test_from_dataclass_suppresses_compat_warning(self, akt1_kinase, caplog):
        with caplog.at_level(logging.WARNING):
            sasa.ResidueSASA.from_dataclass(
                sasa.StandardSASAConfigs.BIOPYTHON_HEAVY,
                dict_kinase={"AKT1": akt1_kinase},
            )
        assert "constructed directly" not in caplog.text

    def test_direct_construction_warns(self, akt1_kinase, caplog):
        with caplog.at_level(logging.WARNING):
            sasa.ResidueSASA(dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False)
        assert "constructed directly" in caplog.text

    def test_hydrogen_with_relative_raises(self, akt1_kinase):
        with pytest.raises(ValueError):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_pymol=False,
                bool_include_hydrogens=True,
                bool_relative=True,
            )

    def test_dot_density_out_of_range_raises(self, akt1_kinase):
        with pytest.raises(ValueError):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase}, bool_pymol=True, dot_density=5
            )

    def test_inert_dot_density_warns(self, akt1_kinase, caplog):
        """Setting a PyMOL-only field with PyMOL off warns it will be ignored."""
        with caplog.at_level(logging.WARNING):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False, dot_density=4
            )
        assert "only affects the PyMOL backend" in caplog.text

    def test_hydrogen_guardrail_warns_when_structure_unprotonated(
        self, akt1_kinase, monkeypatch, caplog
    ):
        """Explicit-H run on an H-free structure warns about the silent no-op."""
        real_convert = sasa.convert_mmcifdict2structure

        def _strip_hydrogens(dict_cif, structure_id="kinase"):
            structure = real_convert(dict_cif, structure_id=structure_id)
            for residue in list(structure.get_residues()):
                for atom_id in [a.id for a in residue if a.element == "H"]:
                    residue.detach_child(atom_id)
            return structure

        monkeypatch.setattr(sasa, "convert_mmcifdict2structure", _strip_hydrogens)
        with caplog.at_level(logging.WARNING):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase},
                bool_pymol=False,
                bool_include_hydrogens=True,
                bool_relative=False,
            ).run()
        assert "no explicit hydrogens" in caplog.text


class TestParallelExecution:
    """n_jobs validation, worker resolution, and parallel/serial parity."""

    def test_n_jobs_zero_raises(self, akt1_kinase):
        with pytest.raises(ValueError):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False, n_jobs=0
            )

    def test_n_jobs_below_neg_one_raises(self, akt1_kinase):
        with pytest.raises(ValueError):
            sasa.ResidueSASA(
                dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False, n_jobs=-2
            )

    def test_resolve_n_workers(self, akt1_kinase):
        calc = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False, n_jobs=3
        )
        assert calc._resolve_n_workers() == 3
        calc_all = sasa.ResidueSASA(
            dict_kinase={"AKT1": akt1_kinase}, bool_pymol=False, n_jobs=-1
        )
        assert calc_all._resolve_n_workers() == (os.cpu_count() or 1)

    def test_parallel_matches_serial(self):
        """A process-pool run reproduces the serial result exactly."""
        dict_kinase = deserialize_kinase_dict(
            list_ids=["AKT1", "EGFR"], bool_verbose=False
        )
        serial = _run_df(
            sasa.ResidueSASA(dict_kinase=dict(dict_kinase), bool_pymol=False, n_jobs=1)
        )
        parallel = _run_df(
            sasa.ResidueSASA(dict_kinase=dict(dict_kinase), bool_pymol=False, n_jobs=2)
        )
        key = ["hgnc_name", "uniprot_idx"]
        df_serial = serial.sort_values(key).reset_index(drop=True)
        df_parallel = parallel.sort_values(key).reset_index(drop=True)
        assert df_serial.equals(df_parallel)
        assert set(df_parallel["hgnc_name"].unique()) == {"AKT1", "EGFR"}
