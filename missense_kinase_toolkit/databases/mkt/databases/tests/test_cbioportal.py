from types import SimpleNamespace

import pandas as pd
import pytest
from mkt.databases import cbioportal, config
from mkt.databases.isoform import SourceTier
from mkt.schema.utils import split_domain_suffix


@pytest.fixture(scope="module")
def cbioportal_instance(cbioportal_probe):
    """Create a cBioPortal client once for this module."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    instance = cbioportal.cBioPortal()
    # a None client means either an upstream outage or a broken client here; only
    # skip for the former, so a genuine regression still fails loudly
    if instance._cbioportal is None and not cbioportal_probe(instance.url):
        pytest.skip("cBioPortal unreachable; skipping live client tests")
    return instance


@pytest.fixture(scope="module")
def mutations_instance():
    """Query MSK-IMPACT 2017 mutations once."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    instance = cbioportal.Mutations(study_id="msk_impact_2017")
    # transient cBioPortal outages leave _df as None; skip rather than fail
    if instance._df is None:
        pytest.skip("cBioPortal unavailable; skipping live mutation tests")
    return instance


@pytest.fixture(scope="module")
def gene_panel_instance():
    """Query IMPACT341 gene panel once."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    instance = cbioportal.GenePanel(panel_id="IMPACT341")
    # transient cBioPortal outages leave _df as None; skip rather than fail
    if instance._df is None:
        pytest.skip("cBioPortal unavailable; skipping live gene panel tests")
    return instance


@pytest.mark.network
class TestCBioPortalClient:
    def test_instance_url(self, cbioportal_instance):
        assert cbioportal_instance.get_instance() == "www.cbioportal.org"

    def test_api_docs_url(self, cbioportal_instance):
        assert (
            cbioportal_instance.get_url()
            == "https://www.cbioportal.org/api/v2/api-docs"
        )

    def test_client_not_none(self, cbioportal_instance):
        assert cbioportal_instance._cbioportal is not None

    def test_server_status_up(self, cbioportal_instance):
        status = (
            cbioportal_instance._cbioportal.Server_running_status.getServerStatusUsingGET()
            .response()
            .result["status"]
        )
        assert status == "UP"


@pytest.mark.network
class TestMutations:
    def test_entity_id_exists(self, mutations_instance):
        assert mutations_instance.check_entity_id() is True

    def test_entity_id_value(self, mutations_instance):
        assert mutations_instance.get_entity_id() == "msk_impact_2017"

    def test_mutation_count(self, mutations_instance):
        assert mutations_instance._df.shape[0] == 78142


@pytest.mark.network
class TestGenePanel:
    def test_panel_entity_id_exists(self, gene_panel_instance):
        assert gene_panel_instance.check_entity_id() is True

    def test_panel_row_count(self, gene_panel_instance):
        assert gene_panel_instance._df.shape[0] == 341

    def test_panel_column_count(self, gene_panel_instance):
        assert gene_panel_instance._df.shape[1] == 2


# offline tests of the kinase missense mutation helpers (no cBioPortal client)

KMM = cbioportal.KinaseMissenseMutations
"""type: Shorthand for the class whose methods are called on a stub instance."""


def _kinase(*names):
    """Return a subset of the packaged kinase dictionary."""
    return {name: cbioportal.DICT_KINASE[name] for name in names}


def _stub(**kwargs):
    """Return a minimal stand-in for a KinaseMissenseMutations instance."""
    attrs = {
        "return_adjusted_colname": lambda colname, prefix="gene": f"{prefix}_{colname}",
        "try_except_middle_int": KMM.try_except_middle_int,
        "tuple_sources": (SourceTier.direct,),
        "str_isoform_override": "mskcc",
        "str_build": "GRCh37",
        "bool_drop_unreconciled": True,
        "str_blosom": "BLOSUM80",
    }
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _braf_rows():
    """Return BRAF rows (one matching, one mismatched residue) plus a matching KIN row."""
    seq = cbioportal.DICT_KINASE["BRAF"].uniprot.canonical_seq
    wrong = "W" if seq[9] != "W" else "C"
    df = pd.DataFrame(
        {
            "gene_hugoGeneSymbol": ["BRAF", "BRAF", "KIN"],
            "proteinChange": [f"{seq[599]}600E", f"{wrong}10A", "K2A"],
        }
    )
    return df, {"BRAF": seq, "KIN": "MKT"}


class TestApplyGeneReplacements:
    def test_missing_target_is_skipped(self):
        dict_in = {"STK19": "P49842", "BRAF": "P15056"}
        dict_out = cbioportal.apply_gene_replacements(
            dict_in, _kinase("BRAF"), {"STK19": "WHR1"}
        )
        assert dict_out == dict_in

    def test_present_target_is_applied(self):
        dict_kinase = {"WHR1": SimpleNamespace(uniprot_id="Q00001")}
        dict_out = cbioportal.apply_gene_replacements(
            {"STK19": "P49842"}, dict_kinase, {"STK19": "WHR1"}
        )
        assert dict_out == {"STK19": "Q00001"}


class TestGeneMaps:
    def test_multi_domain_gene_maps_to_all_domains(self):
        dict_kinase = _kinase("JAK2_1", "JAK2_2", "BRAF")
        accession_jak2 = split_domain_suffix(dict_kinase["JAK2_1"].uniprot_id)[0]
        dict_gene2names = cbioportal.return_gene2names(
            {
                "JAK2": accession_jak2,
                "BRAF": dict_kinase["BRAF"].uniprot_id,
                "TP53": "P04637",
                "NOHGNC": None,
            },
            dict_kinase,
        )
        assert dict_gene2names == {"JAK2": ["JAK2_1", "JAK2_2"], "BRAF": ["BRAF"]}

    def test_domains_share_canonical_sequence(self):
        dict_kinase = _kinase("JAK2_1", "JAK2_2")
        dict_gene2seq = cbioportal.return_gene2seq(
            {"JAK2": ["JAK2_1", "JAK2_2"]}, dict_kinase
        )
        assert dict_gene2seq == {"JAK2": dict_kinase["JAK2_1"].uniprot.canonical_seq}

    def test_differing_canonical_sequences_raise(self):
        dict_kinase = {
            name: SimpleNamespace(uniprot=SimpleNamespace(canonical_seq=seq))
            for name, seq in (("X_1", "MKT"), ("X_2", "MKV"))
        }
        with pytest.raises(ValueError, match="different canonical sequences"):
            cbioportal.return_gene2seq({"X": ["X_1", "X_2"]}, dict_kinase)


def test_return_hgvsg_list():
    df = pd.DataFrame(
        {
            "chr": ["7", "7", "7", "14", "14", "14"],
            "startPosition": [140453136] * 3 + [105246551] * 3,
            "referenceAllele": ["A", "A", "AT", "C", "C", "C"],
            "variantAllele": ["T", "T", "A", "T", "T", "T"],
            "ncbiBuild": ["GRCh37", "GRCh38", "GRCh37", "NA", "37", "hg19"],
        }
    )
    assert cbioportal.return_hgvsg_list(df, "GRCh37") == [
        "7:g.140453136A>T",
        None,
        None,
        None,
        "14:g.105246551C>T",
        "14:g.105246551C>T",
    ]
    assert cbioportal.return_hgvsg_list(df.drop(columns="chr"), "GRCh37") == [None] * 6


def test_assign_mkt_name_resolves_jak2_v617f_to_jh2():
    """JAK2 V617F lies in the JH2 pseudokinase domain (JAK2_2); gaps keep the base name."""
    dict_kinase = _kinase("JAK2_1", "JAK2_2", "BRAF")
    idx_gap = dict_kinase["JAK2_2"].adjudicate_kd_end() + 1
    assert idx_gap < dict_kinase["JAK2_1"].adjudicate_kd_start()
    df = pd.DataFrame(
        {
            "gene_hugoGeneSymbol": ["JAK2", "JAK2", "BRAF", "JAK2", "TP53"],
            "uniprot_idx": pd.array([617, idx_gap, 600, None, 175], dtype="Int64"),
        }
    )
    df_out = cbioportal.assign_mkt_name(
        df, {"JAK2": ["JAK2_1", "JAK2_2"], "BRAF": ["BRAF"]}, dict_kinase
    )
    assert df_out["mkt_name"].tolist() == ["JAK2_2", "JAK2", "BRAF", "JAK2", None]
    assert df_out["in_kinase_domain"].tolist() == [True, False, True, None, None]


class TestPositionRoutes:
    def test_legacy_filter_drops_whole_gene(self):
        df, dict_gene2seq = _braf_rows()
        df_out = KMM.remove_mismatched_uniprot_mutations(_stub(), df, dict_gene2seq)
        assert df_out["gene_hugoGeneSymbol"].tolist() == ["KIN"]
        assert df_out["uniprot_idx"].tolist() == [2]
        assert df_out["reconcile_source"].tolist() == ["direct"]

    def test_reconciliation_drops_rows_individually(self):
        df, dict_gene2seq = _braf_rows()
        df_out = KMM.reconcile_uniprot_positions(_stub(), df, dict_gene2seq)
        assert df_out["proteinChange"].tolist() == [df["proteinChange"][0], "K2A"]
        assert df_out["uniprot_idx"].tolist() == [600, 2]
        assert df_out["reconcile_source"].tolist() == ["direct", "direct"]

    def test_unreconciled_rows_kept_on_request(self):
        df, dict_gene2seq = _braf_rows()
        df_out = KMM.reconcile_uniprot_positions(
            _stub(bool_drop_unreconciled=False), df, dict_gene2seq
        )
        assert len(df_out) == 3
        assert df_out["reconcile_source"].tolist() == ["direct", None, "direct"]

    def test_both_routes_emit_the_same_columns(self):
        df, dict_gene2seq = _braf_rows()
        df_legacy = KMM.remove_mismatched_uniprot_mutations(_stub(), df, dict_gene2seq)
        df_reconciled = KMM.reconcile_uniprot_positions(_stub(), df, dict_gene2seq)
        assert list(df_legacy.columns) == list(df_reconciled.columns)


def test_annotate_kinase_regions_skips_bare_symbols():
    """A bare multi-domain symbol (outside every domain) gets no region annotation."""
    dict_kinase = _kinase("BRAF", "JAK2_1", "JAK2_2")
    braf = dict_kinase["BRAF"]
    region, idx_klifs = next(
        (key, idx) for key, idx in braf.KLIFS2UniProtIdx.items() if idx is not None
    )
    aa_ref = braf.uniprot.canonical_seq[idx_klifs - 1]
    df = pd.DataFrame(
        {
            "mkt_name": ["BRAF", "JAK2"],
            "uniprot_idx": pd.array([idx_klifs, 830], dtype="Int64"),
            "proteinChange": [f"{aa_ref}{idx_klifs}A", "E830K"],
        }
    )
    df_out = KMM.annotate_kinase_regions(_stub(), df, dict_kinase)
    assert df_out["klifs_region"].tolist() == [region, None]
    assert df_out["kincore_kd"].tolist()[1] is None
