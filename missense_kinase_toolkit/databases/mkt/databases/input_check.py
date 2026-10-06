"""Record and check the inputs a derived structure value was computed from.

Derived values (SASA, superposition, the AlphaFold kinase-domain slice) store named SHA-256s
of their large inputs in ``input_sha256`` (structure, residue maps, reference, sequence, and
the code that computed them); their small inputs are ordinary readable fields.
:class:`InputCheck` compares both with the current inputs and logs which changed, so a
rebuild recomputes only stale values and says why.
"""

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

from mkt.schema.utils import return_json_sha256, return_sha256

logger = logging.getLogger(__name__)

_DICT_CODE_SHA256: dict[str, str] = {}
"""dict[str, str]: Code-hash cache (``code:`` key -> SHA-256), one hash per process."""


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove module, class, and function docstrings from a parsed syntax tree."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def return_code_sha256(obj: Any) -> dict[str, str]:
    """Return the ``code:`` input hash of a module, function, or class.

    Hashes the syntax tree with docstrings stripped, so comments, docstrings, and
    formatting don't change it but logic does.

    Parameters
    ----------
    obj : module | function | class
        Code whose logic a derived value depends on: the whole computing module, or a
        single helper from another module.

    Returns
    -------
    dict[str, str]
        ``{"code:<qualified name>": <SHA-256>}``, ready to merge into ``input_sha256``.
    """
    str_name = obj.__name__
    if not inspect.ismodule(obj):
        str_name = f"{obj.__module__}.{obj.__qualname__}"
    str_key = f"code:{str_name}"
    if str_key not in _DICT_CODE_SHA256:
        tree = _strip_docstrings(ast.parse(textwrap.dedent(inspect.getsource(obj))))
        _DICT_CODE_SHA256[str_key] = return_sha256(ast.dump(tree).encode("utf-8"))
    return {str_key: _DICT_CODE_SHA256[str_key]}


def return_structure_sha256(structure: Any) -> str:
    """Return a structure's SHA-256, computing and storing it if the archive lacks it.

    Parameters
    ----------
    structure : KinCoReCIF | AlphaFold
        Structure model with a ``cif`` dict and a ``sha256`` field.

    Returns
    -------
    str
        SHA-256 of the structure's canonical-JSON ``cif`` (see ``return_json_sha256``).
    """
    if structure.sha256 is None:
        structure.sha256 = return_json_sha256(structure.cif)
    return structure.sha256


def return_inputs_sha256(
    dict_inputs: dict[str, Any],
    list_code: list[Any] | None = None,
) -> dict[str, str]:
    """Return named SHA-256s for a derived value's inputs.

    Parameters
    ----------
    dict_inputs : dict[str, Any]
        Input name -> value; strings are taken as precomputed SHA-256s (e.g. a structure's
        ``sha256``), anything else is hashed as canonical JSON (e.g. a residue map).
    list_code : list[Any] | None, optional
        Modules/functions/classes whose code the value depends on, by default None.

    Returns
    -------
    dict[str, str]
        Input name -> SHA-256, including ``code:`` entries.
    """
    dict_sha256 = {
        str_name: val if isinstance(val, str) else return_json_sha256(val)
        for str_name, val in dict_inputs.items()
    }
    for obj in list_code or []:
        dict_sha256.update(return_code_sha256(obj))
    return dict_sha256


@dataclass
class InputCheck:
    """The inputs a derived value would be computed from now, checked against stored ones."""

    str_label: str
    """What is checked, for log messages (e.g. ``"ABL1 kincore SASA"``)."""
    dict_sha256: dict[str, str]
    """Input name -> current SHA-256 (see :func:`return_inputs_sha256`)."""
    dict_readable: dict[str, Any] = field(default_factory=dict)
    """Readable input field name -> current value (e.g. ``probe_radius``, ``start``), by
    default empty."""

    def return_changed(
        self,
        dict_stored_sha256: dict[str, str] | None,
        dict_stored_readable: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return the names of the inputs that differ from the stored ones.

        Parameters
        ----------
        dict_stored_sha256 : dict[str, str] | None
            The stored value's ``input_sha256`` (None for values computed before inputs
            were recorded).
        dict_stored_readable : dict[str, Any] | None, optional
            The stored value's readable input fields, by default None.

        Returns
        -------
        list[str]
            Changed input names (added or removed hashes included); ``["no recorded
            inputs"]`` if nothing was stored.
        """
        if dict_stored_sha256 is None:
            return ["no recorded inputs"]
        list_changed = sorted(
            str_name
            for str_name in set(self.dict_sha256) | set(dict_stored_sha256)
            if self.dict_sha256.get(str_name) != dict_stored_sha256.get(str_name)
        )
        dict_stored_readable = dict_stored_readable or {}
        list_changed.extend(
            str_name
            for str_name, val in self.dict_readable.items()
            if dict_stored_readable.get(str_name) != val
        )
        return list_changed

    def is_stale(
        self,
        stored: Any,
        force: bool = False,
        str_action: str = "recomputing",
    ) -> bool:
        """Return True if the stored value must be recomputed, logging which inputs changed.

        Parameters
        ----------
        stored : BaseModel | None
            The stored derived value (with ``input_sha256`` and the readable fields), or None.
        force : bool, optional
            Recompute regardless (``--recompute``), by default False.
        str_action : str, optional
            Verb for the log message, by default "recomputing".

        Returns
        -------
        bool
            True if absent, forced, or any recorded input changed.
        """
        if stored is None or force:
            return True
        list_changed = self.return_changed(
            stored.input_sha256,
            {
                str_name: getattr(stored, str_name, None)
                for str_name in self.dict_readable
            },
        )
        if not list_changed:
            return False
        str_changed = ", ".join(list_changed)
        str_msg = f"{self.str_label} stale: {str_changed}"
        if list_changed != ["no recorded inputs"]:
            str_msg += " changed"
        # values from pre-Phase-2 archives are expected; keep them out of INFO
        log = logger.debug if list_changed == ["no recorded inputs"] else logger.info
        log(f"{str_msg}; {str_action}")
        return True
