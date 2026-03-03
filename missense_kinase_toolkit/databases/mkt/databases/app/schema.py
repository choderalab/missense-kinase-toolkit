import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.colors import (
    DICT_QUARTILE_HEATMAP_COLORMAP_PLASMA,
    percentile_colormap,
)

logger = logging.getLogger(__name__)


@dataclass
class StructureConfig(ABC):
    """Configuration for generating PyMOL files.

    Uses dependency injection: receives a SequenceAlignment object
    which provides access to kinase info and aligned sequences.
    """

    seq_align: SequenceAlignment
    """SequenceAlignment object providing kinase info and dict_align."""
    str_attr: str
    """Attribute to highlight in the structure."""
    list_idx: list[int] = field(init=False)
    """List of 1-indexed residue positions to be highlighted, generated in __post_init__."""
    list_color: list[str] = field(init=False)
    """List of colors for the residues to be highlighted, generated in __post_init__."""
    list_style: list[str] = field(init=False)
    """List of styles for the residues to be highlighted, generated in __post_init__."""
    list_label: list[str | None] = field(init=False)
    """List of labels for the residues to be highlighted (None for no label)."""

    def __post_init__(self):
        list_idx, list_color, list_style = self.return_list_intersect_color_style()
        self.list_idx = list_idx
        self.list_color = list_color
        self.list_style = list_style
        self.list_label = self.generate_labels(
            [i - 1 for i in list_idx]  # convert back to 0-indexed for label generation
        )

    @abstractmethod
    def generate_style_color_lists(
        self, list_idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """Generate style and color lists for StructureVisualizer based on the configuration.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions (before +1 conversion).

        Returns
        -------
        tuple[list[str], list[str]]
            Tuple of (list_style, list_color).
        """
        ...

    @abstractmethod
    def generate_list_idx(self) -> list[int]:
        """Generate list of 0-indexed positions for the residues to be highlighted.

        Returns
        -------
        list[int]
            List of 0-indexed residue positions.
        """
        ...

    def generate_labels(self, list_idx: list[int]) -> list[str | None]:
        """Generate labels for highlighted residues.

        Override in subclasses to provide specific label formats.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        list[str | None]
            List of labels (None for residues that should not be labeled).
        """
        return [None] * len(list_idx)

    def return_list_cif_idx(self) -> list[int]:
        """Return list of 0-indexed positions corresponding to the CIF sequence.

        Returns
        -------
        list[int]
            List of 0-indexed positions where CIF sequence has residues (not gaps).
        """
        str_seq_cif = self.seq_align.dict_align["KinCore, CIF"]["str_seq"]
        list_cif_idx = [idx for idx, i in enumerate(str_seq_cif) if i != "-"]
        return list_cif_idx

    def _generate_list_idx_from_dict_align_with_attr(self) -> list[int] | None:
        """Generate list of 0-indexed attribute positions from dict_align.

        Returns
        -------
        list[int] | None
            List of 0-indexed positions where attribute sequence has residues,
            or None if attribute not found in dict_align.
        """
        try:
            str_seq_attr = self.seq_align.dict_align[self.str_attr]["str_seq"]
            list_attr_idx = [idx for idx, i in enumerate(str_seq_attr) if i != "-"]
            return list_attr_idx
        except KeyError:
            logger.error(
                f"{self.str_attr} not found in {self.seq_align.dict_align.keys()}"
            )
            return None

    def return_list_idx_intersect(self) -> list[int]:
        """Return list of 0-indexed positions at intersection of CIF and attribute sequences.

        Returns
        -------
        list[int]
            Sorted list of 0-indexed positions present in both CIF and attribute sequences.

        Raises
        ------
        ValueError
            If attribute sequence cannot be generated from dict_align.
        """
        list_cif_idx = self.return_list_cif_idx()
        list_attr_idx = self._generate_list_idx_from_dict_align_with_attr()

        if list_attr_idx is None:
            raise ValueError(
                f"Cannot generate intersecting indices without valid attribute sequence for {self.str_attr}."
            )

        # get intersecting indices between cif and attribute sequences
        set_cif_idx = set(list_cif_idx)
        set_attr_idx = set(list_attr_idx)
        list_intersect = sorted(set_cif_idx.intersection(set_attr_idx))

        return list_intersect

    def return_list_intersect_color_style(
        self,
    ) -> tuple[list[int], list[str], list[str]]:
        """Generate the indices, styles, and colors for highlighting.

        Returns
        -------
        tuple[list[int], list[str], list[str]]
            Tuple of (list_idx, list_color, list_style) where list_idx is 1-indexed.
        """
        list_idx_0based = self.generate_list_idx()
        list_style, list_color = self.generate_style_color_lists(list_idx_0based)

        # convert to 1-based indexing for structure visualization
        list_idx_1based = [i + 1 for i in list_idx_0based]

        return list_idx_1based, list_color, list_style


@dataclass(kw_only=True)
class DefaultConfig(StructureConfig):
    """Default configuration rendering the whole protein as cartoon with spectrum coloring."""

    str_attr: str = "KinCore, CIF"
    """Attribute to highlight in the structure (default: 'KinCore, CIF')."""

    def generate_list_idx(self) -> list[int]:
        """Generate list of 0-indexed positions for all CIF residues.

        Returns
        -------
        list[int]
            List of 0-indexed positions for all CIF residues.
        """
        return self.return_list_cif_idx()

    def generate_style_color_lists(
        self, list_idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """Generate style and color lists for default spectrum coloring.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        tuple[list[str], list[str]]
            All residues get 'cartoon' style and 'spectrum' color.
        """
        list_style = ["cartoon" for _ in list_idx]
        list_color = ["spectrum" for _ in list_idx]
        return list_style, list_color


@dataclass(kw_only=True)
class PhosphositesConfig(StructureConfig):
    """Configuration for highlighting phosphosites in PyMOL."""

    str_attr: str = "Phosphosites"
    """Attribute to highlight in the structure (default: 'Phosphosites')."""

    def generate_list_idx(self) -> list[int]:
        """Generate list of 0-indexed positions for phosphosite residues.

        Returns
        -------
        list[int]
            List of 0-indexed positions for the residues to be highlighted.
        """
        return self.return_list_idx_intersect()

    def generate_style_color_lists(
        self, list_idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """Generate style and color lists for phosphosites.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        tuple[list[str], list[str]]
            All residues get 'stick' style and 'red' color.
        """
        list_style = ["stick" for _ in list_idx]
        list_color = ["red" for _ in list_idx]
        return list_style, list_color


@dataclass(kw_only=True)
class KLIFSConfig(StructureConfig):
    """Configuration for highlighting KLIFS residues in PyMOL."""

    str_attr: str = "KLIFS"
    """Attribute to highlight in the structure (default: 'KLIFS')."""
    list_stick_positions: list[int] = field(default_factory=list)
    """List of 0-indexed positions in KLIFS sequence for stick representation."""

    def generate_list_idx(self) -> list[int]:
        """Generate list of 0-indexed positions for KLIFS residues.

        Returns
        -------
        list[int]
            List of 0-indexed positions for the residues to be highlighted.
        """
        return self.return_list_idx_intersect()

    def generate_style_color_lists(
        self, list_idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """Generate style and color lists for KLIFS residues.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        tuple[list[str], list[str]]
            Styles based on stick positions, colors from KLIFS pocket colors.
        """
        list_attr_idx = self._generate_list_idx_from_dict_align_with_attr()

        list_idx_keep = [idx for idx, i in enumerate(list_idx) if i in list_attr_idx]
        list_idx_stick = self._index_stick_region()
        list_style = [
            "stick" if idx in list_idx_stick and idx in list_idx_keep else "cartoon"
            for idx, _ in enumerate(list_attr_idx)
        ]
        list_color = [
            self.seq_align.dict_align[self.str_attr]["list_colors"][i] for i in list_idx
        ]

        return list_style, list_color

    def _index_stick_region(self) -> list[int]:
        """Extract the indices of the residues in the KLIFS pocket region by removing gaps.

        Returns
        -------
        list[int]
            List of indices for the residues in the KLIFS pocket region.
        """
        i = 0
        list_idx_stick = []
        for idx, res in enumerate(self.seq_align.obj_kinase.klifs.pocket_seq):
            if idx in self.list_stick_positions and res != "-":
                list_idx_stick.append(idx - i)
            if res == "-":
                i += 1
        return list_idx_stick


@dataclass(kw_only=True)
class KLIFSConservedConfig(KLIFSConfig):
    """Configuration for highlighting KLIFS conserved residues in PyMOL."""

    list_stick_positions: list[int] = field(
        default_factory=lambda: [
            3,  # g.l:4
            5,  # g.l:6
            8,  # g.l:9
            68,  # c.l:69 (R in HRD)
            80,  # xDFG:81 (D in xDFG)
        ]
    )
    """List of 0-indexed positions in KLIFS sequence for stick representation (conserved positions)."""


@dataclass(kw_only=True)
class KLIFSImportantConfig(KLIFSConfig):
    """Configuration for highlighting important KLIFS residues in PyMOL."""

    list_stick_positions: list[int] = field(
        default_factory=lambda: [
            44,  # gatekeeper
            45,  # hinge region
            46,
            47,
            67,  # HRD motif
            68,
            69,
            79,  # xDFG motif
            80,
            81,
            82,
        ]
    )
    """List of 0-indexed positions in KLIFS sequence for stick representation (important positions)."""


@dataclass(kw_only=True)
class MutationsConfig(StructureConfig):
    """Configuration for generating PyMOL files with mutation data."""

    str_attr: str = "Mutations"
    """Attribute to highlight in the structure (default: 'Mutations')."""
    bool_mutations_by_group: bool
    """Whether to average mutation counts by kinase group."""
    bool_klifs_only: bool
    """Whether to only highlight KLIFS pocket residues."""
    bool_show_sticks: bool = True
    """Whether to show top mutations as sticks (default: True)."""
    str_filepath_json: str
    """File path to the JSON file containing the mutation dictionary."""
    dict_mutations: dict[int, float] = field(init=False)
    """Dictionary of mutation positions (1-indexed) with corresponding normalized counts."""

    def __post_init__(self):
        self.dict_mutations = self._preprocess_mutation_dict()
        if self.bool_klifs_only:
            self.dict_mutations = self._filter_mutations_to_klifs()
        super().__post_init__()

    def generate_list_idx(self) -> list[int]:
        """Generate list of 0-indexed positions for mutation residues.

        Returns
        -------
        list[int]
            List of 0-indexed positions for the residues to be highlighted.
        """
        # dict_mutations keys are 1-indexed, convert to 0-indexed
        list_attr_idx = [i - 1 for i in self.dict_mutations.keys()]
        list_cif_idx = self.return_list_cif_idx()
        assert all(
            i in list_cif_idx for i in list_attr_idx
        ), "Some mutation indices are not present in the KinCore CIF sequence."
        return list_attr_idx

    @staticmethod
    def _get_klifs_uniprot_positions(klifs_mapping: dict | None) -> set[int]:
        """Get the set of UniProt positions (1-indexed) in the KLIFS pocket.

        Parameters
        ----------
        klifs_mapping : dict | None
            KLIFS2UniProtIdx mapping from kinase object.

        Returns
        -------
        set[int]
            Set of 1-indexed UniProt positions in the KLIFS pocket.
        """
        if klifs_mapping is None:
            return set()
        return {v for v in klifs_mapping.values() if v is not None}

    @staticmethod
    def _get_klifs_uniprot_mapping(
        klifs_mapping: dict | None,
        bool_uniprot_to_klifs: bool = True,
    ) -> dict[int, int]:
        """Create mapping between UniProt positions and KLIFS indices.

        Parameters
        ----------
        klifs_mapping : dict | None
            KLIFS2UniProtIdx mapping from kinase object (ordered dict).
        bool_uniprot_to_klifs : bool, optional
            If True, return UniProt position (1-indexed) -> KLIFS index (0-84).
            If False, return KLIFS index (0-84) -> UniProt position (1-indexed).
            Default is True.

        Returns
        -------
        dict[int, int]
            Mapping between UniProt positions and KLIFS indices.
        """
        if klifs_mapping is None:
            return {}
        if bool_uniprot_to_klifs:
            return {
                uniprot_pos: klifs_idx
                for klifs_idx, uniprot_pos in enumerate(klifs_mapping.values())
                if uniprot_pos is not None
            }
        else:
            return {
                klifs_idx: uniprot_pos
                for klifs_idx, uniprot_pos in enumerate(klifs_mapping.values())
                if uniprot_pos is not None
            }

    def _filter_mutations_to_klifs(self) -> dict[int, float]:
        """Filter dict_mutations to only include residues in the 85 KLIFS pocket positions.

        Returns
        -------
        dict[int, float]
            Filtered dictionary with only KLIFS pocket mutations.
        """
        klifs_mapping = getattr(self.seq_align.obj_kinase, "KLIFS2UniProtIdx", None)
        klifs_positions = self._get_klifs_uniprot_positions(klifs_mapping)

        if not klifs_positions:
            logger.warning(
                f"KLIFS2UniProtIdx not available for {self.seq_align.obj_kinase.hgnc_name}, "
                "returning unfiltered mutations."
            )
            return self.dict_mutations

        # filter mutations to only include KLIFS positions (keys are 1-indexed)
        filtered = {
            k: v for k, v in self.dict_mutations.items() if k in klifs_positions
        }

        logger.info(
            f"Filtered mutations from {len(self.dict_mutations)} to {len(filtered)} "
            f"KLIFS pocket positions for {self.seq_align.obj_kinase.hgnc_name}."
        )

        return filtered

    def _map_group_klifs_to_uniprot(
        self,
        dict_group_data: dict[int, float],
    ) -> dict[int, float]:
        """Map group-averaged KLIFS-indexed data to target kinase's UniProt positions.

        Parameters
        ----------
        dict_group_data : dict[int, float]
            Group-averaged normalized counts keyed by KLIFS index (0-84).

        Returns
        -------
        dict[int, float]
            Normalized counts mapped to target kinase's UniProt positions (1-indexed).
        """
        str_kinase = self.seq_align.str_kinase
        klifs_mapping = getattr(self.seq_align.obj_kinase, "KLIFS2UniProtIdx", None)
        if klifs_mapping is None:
            raise ValueError(
                f"KLIFS2UniProtIdx not available for {str_kinase}, "
                "cannot map group-averaged KLIFS positions to UniProt positions."
            )

        # KLIFS index (0-84) -> UniProt position (1-indexed)
        klifs_idx_to_uniprot = self._get_klifs_uniprot_mapping(
            klifs_mapping, bool_uniprot_to_klifs=False
        )

        return {
            klifs_idx_to_uniprot[klifs_idx]: norm_count
            for klifs_idx, norm_count in dict_group_data.items()
            if klifs_idx in klifs_idx_to_uniprot
        }

    def _preprocess_mutation_dict(self) -> dict[int, float]:
        """Preprocess mutation dictionary from JSON file.

        Handles two JSON formats:
        - New format: ``{"kinases": {...}, "kinase_groups": {...}}``
        - Legacy format: ``{gene_name: {pos: count, ...}, ...}``

        For group-averaged mode, reads pre-computed group data (keyed by KLIFS index)
        and maps it to the target kinase's UniProt positions.

        Returns
        -------
        dict[int, float]
            Mutation dictionary with UniProt position keys (1-indexed) and normalized counts.
        """
        with open(self.str_filepath_json) as f:
            dict_mutations_total = json.load(f)

        str_kinase = self.seq_align.str_kinase

        if self.bool_mutations_by_group:
            # use pre-computed group-averaged data (keyed by KLIFS index 0-84)
            kinase_groups = dict_mutations_total.get("kinase_groups")
            if kinase_groups is None:
                raise ValueError(
                    "JSON file does not contain 'kinase_groups' key. "
                    "Please regenerate the structure heatmap data."
                )

            # find this kinase's group
            target_group = self.seq_align.obj_kinase.adjudicate_group()
            if target_group not in kinase_groups:
                raise ValueError(
                    f"No group-averaged mutations found for group '{target_group}' "
                    f"(kinase {str_kinase}) in provided JSON file."
                )

            # convert JSON string keys to int and map KLIFS indices to UniProt positions
            group_data = {
                int(k): float(v) for k, v in kinase_groups[target_group].items()
            }
            return self._map_group_klifs_to_uniprot(group_data)
        else:
            # use per-kinase data; support both new and legacy JSON formats
            kinases = dict_mutations_total.get("kinases", dict_mutations_total)

            if str_kinase not in kinases:
                raise ValueError(
                    f"No mutations found for gene {str_kinase} in provided JSON file."
                )

            return {int(k): float(v) for k, v in kinases[str_kinase].items()}

    def _index_mutation_region(self) -> list[int]:
        """Extract the indices of the top mutation residues for stick highlighting.

        Returns
        -------
        list[int]
            List of 0-indexed positions for the top mutation residues.
        """
        if self.dict_mutations is None:
            logger.warning(
                "dict_mutations is None, cannot extract mutation region indices."
            )
            return []

        # extract indices of top 10 mutations (allowing for more in case of ties)
        dict_temp = dict(
            sorted(self.dict_mutations.items(), key=lambda item: item[1], reverse=True)
        )
        values = list(dict_temp.values())
        threshold = values[min(9, len(values) - 1)] if values else float("-inf")
        # keys are 1-indexed, convert to 0-indexed
        list_idx_mutation = [k - 1 for k, v in dict_temp.items() if v >= threshold]

        return list_idx_mutation

    def _get_uniprot_label(self, idx_0based: int) -> str:
        """Get amino acid + UniProt position label (e.g., 'T790').

        Parameters
        ----------
        idx_0based : int
            0-indexed position in the alignment.

        Returns
        -------
        str
            Label in format 'X###' where X is single-letter amino acid code.
        """
        uniprot_pos = idx_0based + 1  # convert to 1-indexed
        aa = self.seq_align.obj_kinase.uniprot.canonical_seq[idx_0based]
        return f"{aa}{uniprot_pos}"

    def _get_klifs_label(self, idx_0based: int) -> str | None:
        """Get KLIFS pocket label for a position (e.g., 'GK:45').

        Parameters
        ----------
        idx_0based : int
            0-indexed position in the alignment.

        Returns
        -------
        str | None
            KLIFS pocket label or None if not in KLIFS pocket.
        """
        klifs_mapping = getattr(self.seq_align.obj_kinase, "KLIFS2UniProtIdx", None)
        if klifs_mapping is None:
            return None
        # reverse map: UniProt position (1-indexed) -> KLIFS label
        uniprot_pos = idx_0based + 1
        reverse_map = {v: k for k, v in klifs_mapping.items() if v is not None}
        return reverse_map.get(uniprot_pos)

    @abstractmethod
    def generate_labels(self, list_idx: list[int]) -> list[str | None]:
        """Generate labels for top mutation residues.

        Must be implemented by subclasses to provide config-specific label formats.

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        list[str | None]
            List of labels (None for residues that should not be labeled).
        """
        ...

    def generate_style_color_lists(
        self, list_idx: list[int]
    ) -> tuple[list[str], list[str]]:
        """Generate style and color lists for mutation residues.

        Uses piecewise red gradient coloring:
        - Zero counts get lightgray
        - Non-zero counts are scaled piecewise and interpolated from light red to red

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        tuple[list[str], list[str]]
            Tuple of (list_style, list_color).
        """
        list_norm_count = list(self.dict_mutations.values())

        assert len(list_idx) == len(
            list_norm_count
        ), "Length of indices and normalized counts must be the same."

        # style: sticks for top mutations (if enabled), cartoon otherwise
        if self.bool_show_sticks:
            list_idx_stick = self._index_mutation_region()
            list_style = [
                "stick" if i in list_idx_stick else "cartoon" for i in list_idx
            ]
        else:
            list_style = ["cartoon" for _ in list_idx]

        list_color = percentile_colormap(
            list_norm_count, DICT_QUARTILE_HEATMAP_COLORMAP_PLASMA
        )

        return list_style, list_color


@dataclass(kw_only=True)
class MutationsDefaultConfig(MutationsConfig):
    """Default configuration for mutation visualization in PyMOL."""

    bool_mutations_by_group: bool = False
    """Whether to average mutation counts by kinase group (default: False)."""
    bool_klifs_only: bool = False
    """Whether to only highlight KLIFS pocket residues (default: False)."""

    def generate_labels(self, list_idx: list[int]) -> list[str | None]:
        """Generate labels for top mutations: amino acid + UniProt position (e.g., 'T790').

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        list[str | None]
            Labels for top mutation positions, None for others.
        """
        top_mutations = set(self._index_mutation_region())
        return [
            self._get_uniprot_label(idx) if idx in top_mutations else None
            for idx in list_idx
        ]


@dataclass(kw_only=True)
class MutationsGroupConfig(MutationsConfig):
    """Group-averaged configuration for mutation visualization in PyMOL."""

    bool_mutations_by_group: bool = True
    """Whether to average mutation counts by kinase group (default: True)."""
    bool_klifs_only: bool = True
    """Whether to only highlight KLIFS pocket residues (default: True)."""
    bool_show_sticks: bool = False
    """Whether to show top mutations as sticks (default: False for group-averaged)."""

    def generate_labels(self, list_idx: list[int]) -> list[str | None]:
        """Generate labels for top mutations: KLIFS pocket label (e.g., 'GK:45').

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        list[str | None]
            KLIFS labels for top mutation positions, None for others.
        """
        top_mutations = set(self._index_mutation_region())
        return [
            self._get_klifs_label(idx) if idx in top_mutations else None
            for idx in list_idx
        ]


@dataclass(kw_only=True)
class MutationsKLIFSConfig(MutationsConfig):
    """KLIFS-focused configuration for mutation visualization in PyMOL."""

    bool_mutations_by_group: bool = False
    """Whether to average mutation counts by kinase group (default: False)."""
    bool_klifs_only: bool = True
    """Whether to only highlight KLIFS pocket residues (default: True)."""

    def generate_labels(self, list_idx: list[int]) -> list[str | None]:
        """Generate labels for top mutations: KLIFS label + amino acid + UniProt position.

        Example: 'GK:45 T790'

        Parameters
        ----------
        list_idx : list[int]
            List of 0-indexed residue positions.

        Returns
        -------
        list[str | None]
            Combined KLIFS + UniProt labels for top mutation positions, None for others.
        """
        top_mutations = set(self._index_mutation_region())
        labels = []
        for idx in list_idx:
            if idx not in top_mutations:
                labels.append(None)
                continue
            klifs_label = self._get_klifs_label(idx)
            uniprot_label = self._get_uniprot_label(idx)
            if klifs_label:
                labels.append(f"{klifs_label} {uniprot_label}")
            else:
                labels.append(uniprot_label)
        return labels


class StandardConfigChoice(str, Enum):
    """String-based enum for CLI choices (dataclass not hashable)."""

    DEFAULT = "DEFAULT"
    PHOSPHOSITES = "PHOSPHOSITES"
    KLIFS_CONSERVED = "KLIFS_CONSERVED"
    KLIFS_IMPORTANT = "KLIFS_IMPORTANT"
    MUTATIONS_DEFAULT = "MUTATIONS_DEFAULT"
    MUTATIONS_GROUP = "MUTATIONS_GROUP"
    MUTATIONS_KLIFS = "MUTATIONS_KLIFS"
    MUTATIONS_KLIFS_GROUP = "MUTATIONS_KLIFS_GROUP"


class StandardConfig(Enum):
    """Enumeration of standard configurations for PyMOL visualization."""

    DEFAULT = DefaultConfig
    PHOSPHOSITES = PhosphositesConfig
    KLIFS_CONSERVED = KLIFSConservedConfig
    KLIFS_IMPORTANT = KLIFSImportantConfig
    MUTATIONS_DEFAULT = MutationsDefaultConfig
    MUTATIONS_GROUP = MutationsGroupConfig
    MUTATIONS_KLIFS = MutationsKLIFSConfig
