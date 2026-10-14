import logging

import pytest


def test_adjudicate_group(mutable_kinase, caplog):
    """Test kinase group adjudication across data-source priorities.

    Uses ``mutable_kinase`` because the final case sets ``PI4KA.klifs = None``
    to exercise the no-group-found path.
    """
    caplog.set_level(logging.INFO)

    assert mutable_kinase("ABL1").adjudicate_group() == "TK"  # Kincore
    assert mutable_kinase("ADCK1").adjudicate_group() == "Atypical"  # KinHub

    obj_pi4ka = mutable_kinase("PI4KA")
    assert obj_pi4ka.adjudicate_group() == "Atypical"  # KLIFS

    # remove KLIFS so no source can supply a group
    obj_pi4ka.klifs = None
    caplog.clear()
    assert obj_pi4ka.adjudicate_group(bool_verbose=True) is None
    assert "No group found for PI4KA" in caplog.text


def test_adjudicate_kd_clean_bounds(dict_kinase):
    """A kinase whose KLIFS pocket falls inside the KD is returned unchanged."""
    obj = dict_kinase["ABL1"]
    # KLIFS indices 246-385 fall within the adjudicated KD 234-503
    assert obj.adjudicate_kd_start() == 234
    assert obj.adjudicate_kd_end() == 503


def test_adjudicate_kd_no_klifs(mutable_kinase):
    """Missing KLIFS2UniProtIdx leaves the adjudicated bounds untouched."""
    obj = mutable_kinase("ABL1")
    obj.KLIFS2UniProtIdx = None
    assert obj.adjudicate_kd_start() == 234
    assert obj.adjudicate_kd_end() == 503


@pytest.mark.parametrize(
    "hgnc_name, expected_start, expected_end",
    [
        ("NPR1", 532, 803),  # start gap of 1 -> expand start
        ("NPR2", 516, 788),  # start gap of 1 -> expand start
        ("RPS6KL1", 149, 539),  # start gap of 1 -> expand start
        ("BUB1B", 759, 1021),  # start gap of 7 -> expand start
        ("GUCY2F", 536, 811),  # start gap of 11 -> expand start
    ],
)
def test_adjudicate_kd_small_gap_expands(
    dict_kinase, hgnc_name, expected_start, expected_end
):
    """A KLIFS pocket extending past the bound expands it to the KLIFS index."""
    obj = dict_kinase[hgnc_name]
    assert obj.adjudicate_kd_start() == expected_start
    assert obj.adjudicate_kd_end() == expected_end


def test_adjudicate_kd_large_gap_expands_by_default(dict_kinase):
    """With the default infinite cut-off, large gaps expand to the KLIFS index.

    These bounds returned None under the historical finite cut-off; the KLIFS
    pocket is now trusted as the better-annotated bound (large kinase-domain
    inserts missed by Pfam but present in KLIFS).
    """
    # PIK3CA Pfam start (798) gap to the KLIFS minimum (769) expands to the KLIFS index
    # (a lipid kinase, so no KinCoRe/MSA overrides Pfam here)
    assert dict_kinase["PIK3CA"].adjudicate_kd_start() == 769


def test_adjudicate_kd_fallbacks(dict_kinase, mutable_kinase):
    """Without KinCoRe, Pfam bounds single-domain entries and the KLIFS pocket is last."""
    # ADCK2 has neither KinCoRe nor an intersecting Pfam hit, so KLIFS bounds the KD
    obj = mutable_kinase("ADCK2")
    obj.pfam = None
    assert (obj.adjudicate_kd_start(), obj.adjudicate_kd_end()) == (285, 497)

    # multi-domain entries skip Pfam (one span per protein) and fall back to KLIFS
    obj = mutable_kinase("JAK1_2")
    assert obj.pfam is not None
    obj.kincore = None
    bounds = obj._klifs_uniprot_idx_bounds()
    assert (obj.adjudicate_kd_start(), obj.adjudicate_kd_end()) == bounds

    # TEX14_2 has no source for its own domain, so it no longer overlaps TEX14_1
    assert dict_kinase["TEX14_2"].adjudicate_kd_start() is None
    assert dict_kinase["TEX14_2"].adjudicate_kd_end() is None
    assert dict_kinase["TEX14_1"].adjudicate_kd_start() == 227
    assert dict_kinase["TEX14_1"].adjudicate_kd_end() == 512


def test_adjudicate_kd_finite_cutoff_returns_none(dict_kinase, caplog):
    """An explicit finite int_max_gap still returns None and warns."""
    caplog.set_level(logging.WARNING)

    # PIK3CA start gap is 29 (> explicit cut-off of 15)
    assert dict_kinase["PIK3CA"].adjudicate_kd_start(int_max_gap=15) is None
    assert "Kinase domain start found for PIK3CA" in caplog.text
    assert "larger than cut-off 15" in caplog.text

    # a large-but-finite cut-off still expands the PIK3CA start bound
    assert dict_kinase["PIK3CA"].adjudicate_kd_start(int_max_gap=2000) == 769


def test_adjudicate_kd_verbose_logs_expansion(dict_kinase, caplog):
    """Verbose mode logs an info message when a bound is expanded."""
    caplog.set_level(logging.INFO)
    assert dict_kinase["BUB1B"].adjudicate_kd_start(bool_verbose=True) == 759
    assert "expanding start to 759" in caplog.text


def test_molecular_brake_residues(dict_kinase):
    """Brake residues are read via KLIFS2UniProtIdx with the VIII:79 -1 offset.

    FGFR2 carries the full canonical N-E-K triad; EGFR keeps only the conserved
    VIII:79 lysine. The brake lysine sits one residue N-terminal to its VIII:79
    KLIFS-aligned index, so the -1 offset must be applied -- the raw mapped index
    resolves to a non-conserved residue.
    """
    obj = dict_kinase["FGFR2"]
    assert obj.return_molecular_brake_residues() == {
        "b.l:37": "N",
        "hinge:46": "E",
        "VIII:79-1": "K",
    }
    # the -1 offset recovers the lysine; the raw mapped index does not
    idx = obj.KLIFS2UniProtIdx["VIII:79"]
    seq = obj.uniprot.canonical_seq
    assert seq[idx - 1] != "K" and seq[idx - 2] == "K"

    assert dict_kinase["EGFR"].return_molecular_brake_residues() == {
        "b.l:37": "R",
        "hinge:46": "Q",
        "VIII:79-1": "K",
    }


@pytest.mark.parametrize(
    "hgnc_name, expected",
    [
        ("FGFR2", (True, True, True)),  # canonical FGFR brake
        ("FGFR1", (True, True, True)),
        ("KIT", (True, True, True)),
        ("PDGFRA", (True, True, True)),
        ("EGFR", (False, False, True)),  # only the VIII:79 lysine is conserved
    ],
)
def test_molecular_brake_against_canonical(dict_kinase, hgnc_name, expected):
    """Brake residues are compared position-wise against the canonical N-E-K."""
    assert dict_kinase[hgnc_name].check_molecular_brake_against_canonical() == expected


def test_molecular_brake_missing_mapping(mutable_kinase):
    """Absent or unmapped KLIFS positions yield None rather than raising."""
    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx = None
    assert obj.return_molecular_brake_residues() is None
    assert obj.check_molecular_brake_against_canonical() is None

    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx["b.l:37"] = None
    assert obj.return_molecular_brake_residues()["b.l:37"] is None
    # the unmapped position no longer matches; the other two still do
    assert obj.check_molecular_brake_against_canonical() == (False, True, True)


def test_adjudicate_kd_sequence_one_to_one_with_bounds(dict_kinase):
    """adjudicate_kd_sequence is exactly the canonical UniProt slice over the KD bounds.

    Guards the invariant that the KD sequence, the KD start/end, and (by construction) the
    KD-sliced AlphaFold structure are 1-to-1: ``seq == canonical_seq[start - 1 : end]`` and
    ``len(seq) == end - start + 1``.
    """
    for obj in dict_kinase.values():
        seq = obj.adjudicate_kd_sequence()
        start, end = obj.adjudicate_kd_start(), obj.adjudicate_kd_end()
        if seq is None:
            assert start is None or end is None
            continue
        assert len(seq) == end - start + 1
        assert seq == obj.uniprot.canonical_seq[start - 1 : end]


def test_return_catalytic_residues(dict_kinase, mutable_kinase):
    """Catalytic residues are read from the KLIFS pocket at the canonical positions."""
    from mkt.schema.constants import (
        STR_KLIFS_BETA3_LYSINE,
        STR_KLIFS_CATALYTIC_ASP,
        STR_KLIFS_DFG_ASP,
    )

    assert dict_kinase["ABL1"].return_catalytic_residue_source() == "klifs"
    res = dict_kinase["ABL1"].return_catalytic_residues()
    assert res[STR_KLIFS_BETA3_LYSINE] == "K"  # VAIK beta3 lysine
    assert res[STR_KLIFS_CATALYTIC_ASP] == "D"  # HRD catalytic aspartate
    assert res[STR_KLIFS_DFG_ASP] == "D"  # DFG aspartate

    # without a KLIFS pocket the MSA fallback reads the same residues
    obj = mutable_kinase("ABL1")
    obj.klifs.pocket_seq = None
    assert obj.return_catalytic_residue_source() == "msa"
    assert obj.return_catalytic_residues() == res

    # with neither alignment there are no catalytic residues to read
    obj.kincore.msa = None
    assert obj.return_catalytic_residue_source() is None
    assert obj.return_catalytic_residues() is None


def test_return_catalytic_residues_uniprot_idx(dict_kinase):
    """bool_uniprot_idx appends the UniProt index from KLIFS, or the MSA without KLIFS."""
    abl1 = dict_kinase["ABL1"]
    res_idx = abl1.return_catalytic_residues(bool_uniprot_idx=True)
    assert res_idx["III:17"] == "K271"
    assert [res_idx[i] for i in ("c.l:68", "c.l:69", "c.l:70")] == [
        "H361",
        "R362",
        "D363",
    ]
    assert [res_idx[i] for i in ("xDFG:81", "xDFG:82", "xDFG:83")] == [
        "D381",
        "F382",
        "G383",
    ]
    # the flag only appends the index; residues match the default letters
    dict_letters = {k: v and v[0] for k, v in res_idx.items()}
    assert dict_letters == abl1.return_catalytic_residues()

    # PEAK3 has no KLIFS pocket, so indices come from the MSA (HRD reads LVE)
    peak3 = dict_kinase["PEAK3"]
    assert peak3.return_catalytic_residue_source() == "msa"
    res_idx = peak3.return_catalytic_residues(bool_uniprot_idx=True)
    assert res_idx["III:17"] == "K204"
    assert [res_idx[i] for i in ("c.l:68", "c.l:69", "c.l:70")] == [
        "L302",
        "V303",
        "E304",
    ]
    assert [res_idx[i] for i in ("xDFG:81", "xDFG:82", "xDFG:83")] == [
        "D330",
        "F331",
        "G332",
    ]


def test_is_pseudokinase_tristate(dict_kinase):
    """is_pseudokinase is tri-state; the MSA fallback rescues KLIFS-less kinases."""
    # KLIFS-less but MSA-mapped: PEAK3 lacks the HRD aspartate, SIK1B is intact
    for name in ("CDK11A", "PEAK3", "SIK1B"):
        assert (
            dict_kinase[name].klifs is None
            or dict_kinase[name].klifs.pocket_seq is None
        )
        assert dict_kinase[name].return_catalytic_residue_source() == "msa"
    assert dict_kinase["PEAK3"].is_pseudokinase() is True
    assert dict_kinase["SIK1B"].is_pseudokinase() is False

    # a gap within an available alignment is a missing residue, not an unassessable one:
    # PLK5's MSA row is gapped across the whole triad, so it is a predicted pseudokinase
    assert dict_kinase["PLK5"].return_catalytic_residue_source() == "msa"
    assert all(
        v is None for v in dict_kinase["PLK5"].return_catalytic_residues().values()
    )
    assert dict_kinase["PLK5"].is_pseudokinase() is True

    # atypical kinases with neither alignment are unassessable, not "not a pseudokinase"
    for name in ("ALPK1", "PDK1", "PRKDC", "TRPM7", "PIP5K1A"):
        assert dict_kinase[name].return_catalytic_residues() is None
        assert dict_kinase[name].is_pseudokinase() is None


def test_msa_fallback_does_not_load_corpus(dict_kinase, monkeypatch):
    """The MSA fallback reads a constant map, so one KinaseInfo never loads the corpus.

    Regression: the Streamlit app deserializes a single kinase at a time; loading all of
    ``DICT_KINASE`` here (~7 GB) OOM-killed it on KLIFS-less kinases such as PEAK3.
    """
    from mkt.schema import io_utils

    def _raise(*args, **kwargs):
        raise AssertionError("deserialize_kinase_dict called by the MSA fallback")

    monkeypatch.setattr(io_utils, "deserialize_kinase_dict", _raise)

    obj = dict_kinase["PEAK3"]
    assert obj.return_catalytic_residue_source() == "msa"
    assert obj.is_pseudokinase() is True


def test_return_hrd_motif_labels(dict_kinase):
    """PIK/PIKK kinases read the catalytic loop D-R-H, so their HRD sits at c.l:72-71-70."""
    from mkt.schema.constants import LIST_KLIFS_HRD_MOTIF, LIST_KLIFS_HRD_MOTIF_REVERSED

    assert dict_kinase["ABL1"].return_hrd_motif_labels() == LIST_KLIFS_HRD_MOTIF
    for name in ["ATM", "PIK3CA", "PI4K2A"]:
        assert (
            dict_kinase[name].return_hrd_motif_labels() == LIST_KLIFS_HRD_MOTIF_REVERSED
        )

    residues = dict_kinase["ATM"].return_catalytic_residues(bool_uniprot_idx=True)
    assert [residues[i] for i in LIST_KLIFS_HRD_MOTIF_REVERSED] == [
        "H2872",
        "R2871",
        "D2870",
    ]


def _return_klifs_span(obj) -> tuple[int, int]:
    """First and last UniProt index of a kinase's KLIFS pocket."""
    list_idx = [i for i in obj.KLIFS2UniProtIdx.values() if i is not None]
    return min(list_idx), max(list_idx)


@pytest.mark.parametrize(
    "hgnc_name, expected",
    [
        ("ABL1", "ABL1"),
        ("JAK1_1", "JAK1 JH1"),
        ("JAK1_2", "JAK1 JH2"),
        ("JAK2_1", "JAK2 JH1"),
        ("JAK2_2", "JAK2 JH2"),
        ("JAK3_1", "JAK3 JH1"),
        ("JAK3_2", "JAK3 JH2"),
        ("TYK2_1", "TYK2 JH1"),
        ("TYK2_2", "TYK2 JH2"),
        ("RPS6KA1_1", "RPS6KA1 NTKD"),
        ("RPS6KA1_2", "RPS6KA1 CTKD"),
        ("RPS6KA4_1", "RPS6KA4 NTKD"),
        ("RPS6KA4_2", "RPS6KA4 CTKD"),
        ("EIF2AK4_1", "EIF2AK4 KD"),
        ("EIF2AK4_2", "EIF2AK4 ΨKD"),
        ("OBSCN_1", "OBSCN SK1"),
        ("OBSCN_2", "OBSCN SK2"),
        ("SPEG_1", "SPEG SK1"),
        ("SPEG_2", "SPEG SK2"),
        ("TEX14_1", "TEX14 (SgK307)"),
        ("TEX14_2", "TEX14 (SgK424)"),
    ],
)
def test_adjudicate_name(dict_kinase, hgnc_name, expected):
    """A single-domain kinase keeps its HGNC name; a domain entry names its domain."""
    assert dict_kinase[hgnc_name].adjudicate_name() == expected


def test_adjudicate_name_resolves_every_multi_domain_entry(dict_kinase):
    """Every suffixed entry gets a name without the suffix, distinct from its pair's."""
    from mkt.schema.utils import split_domain_suffix

    dict_names: dict[str, set[str]] = {}
    for hgnc_name, obj in dict_kinase.items():
        str_gene, str_suffix = split_domain_suffix(hgnc_name)
        if not str_suffix:
            assert obj.adjudicate_name() == hgnc_name
            continue
        str_name = obj.adjudicate_name()
        assert str_name.startswith(f"{str_gene} ")
        assert not str_name.endswith(str_suffix)
        dict_names.setdefault(str_gene, set()).add(str_name)
    # 14 multi-domain proteins, each with two differently named domains
    assert len(dict_names) == 14
    assert all(len(set_names) == 2 for set_names in dict_names.values())


@pytest.mark.parametrize("str_gene", ["JAK1", "JAK2", "JAK3", "TYK2"])
def test_adjudicate_name_jak_jh1_follows_jh2(dict_kinase, str_gene):
    """JH1 is the C-terminal catalytic domain: its KLIFS pocket starts after JH2's
    pocket ends, and only JH2 is a pseudokinase."""
    obj_jh1, obj_jh2 = dict_kinase[f"{str_gene}_1"], dict_kinase[f"{str_gene}_2"]
    assert obj_jh1.adjudicate_name().endswith("JH1")
    assert _return_klifs_span(obj_jh1)[0] > _return_klifs_span(obj_jh2)[1]
    assert obj_jh2.is_pseudokinase() is True
    assert obj_jh1.is_pseudokinase() is False


@pytest.mark.parametrize(
    "str_gene", ["RPS6KA1", "RPS6KA2", "RPS6KA3", "RPS6KA4", "RPS6KA5", "RPS6KA6"]
)
def test_adjudicate_name_rsk_ntkd_precedes_ctkd(dict_kinase, str_gene):
    """NTKD is the N-terminal AGC domain, CTKD the C-terminal CAMK domain."""
    obj_ntkd, obj_ctkd = dict_kinase[f"{str_gene}_1"], dict_kinase[f"{str_gene}_2"]
    assert obj_ntkd.adjudicate_name().endswith("NTKD")
    assert _return_klifs_span(obj_ntkd)[1] < _return_klifs_span(obj_ctkd)[0]
    assert obj_ntkd.adjudicate_group() == "AGC"
    assert obj_ctkd.adjudicate_group() == "CAMK"


def test_adjudicate_name_gcn2_pseudokinase_precedes_kd(dict_kinase):
    """GCN2's pseudokinase domain (ΨKD, _2) lies N-terminal to its kinase domain."""
    obj_kd, obj_pkd = dict_kinase["EIF2AK4_1"], dict_kinase["EIF2AK4_2"]
    assert _return_klifs_span(obj_pkd)[1] < _return_klifs_span(obj_kd)[0]
    assert obj_pkd.is_pseudokinase() is True
    assert obj_kd.is_pseudokinase() is False


@pytest.mark.parametrize("str_gene", ["OBSCN", "SPEG"])
def test_adjudicate_name_sk1_precedes_sk2(dict_kinase, str_gene):
    """SK1 is the first of the two kinase domains in the sequence."""
    obj_sk1, obj_sk2 = dict_kinase[f"{str_gene}_1"], dict_kinase[f"{str_gene}_2"]
    assert _return_klifs_span(obj_sk1)[1] < _return_klifs_span(obj_sk2)[0]


def test_adjudicate_name_unknown_multi_domain_kinase_raises(mutable_kinase):
    """A suffixed kinase outside the known families raises rather than guessing."""
    obj = mutable_kinase("ABL1")
    obj.hgnc_name = "ABL1_1"
    with pytest.raises(ValueError, match="not in the known list"):
        obj.adjudicate_name()
