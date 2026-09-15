"""Tests for the shared ``--data/--no-data`` and ``--figs/--no-figs`` flags of the figure CLIs.

Covers :meth:`mkt.databases.plot_config.DataFiguresConfig.resolve_data`, the task-config
hierarchy, and how the kinaseinfo, conservation, and dataset CLIs parse and apply the flags,
with the build and figure work stubbed so no data is fetched or written.
"""

import pytest
from mkt.databases.cli import generate_conservation_data as cli_conservation
from mkt.databases.cli import generate_dataset_csv_files as cli_dataset
from mkt.databases.cli import generate_kinaseinfo_objects as cli_kinaseinfo
from mkt.databases.plot_config import (
    ArgumentError,
    ConservationFiguresConfig,
    DataFiguresConfig,
    DatasetFiguresConfig,
    KinaseInfoFiguresConfig,
    PymolConfig,
    TaskConfig,
    load_task_config,
)
from omegaconf.errors import ConfigKeyError
from typer.testing import CliRunner

RUNNER = CliRunner()
"""CliRunner: Shared Typer test runner."""


@pytest.fixture(autouse=True)
def _isolate_cli_logging(monkeypatch, caplog):
    """Route CLI logs to ``caplog`` only, keeping them away from pytest's live-log handler.

    With ``log_cli = true``, a WARNING+ record emitted inside ``CliRunner.invoke`` makes pytest
    suspend/resume capture, which swaps out the runner's ``sys.stdout`` wrapper; the orphaned
    wrapper is garbage-collected and closes the runner's buffer ("I/O operation on closed file").
    """
    for cli in (cli_conservation, cli_dataset, cli_kinaseinfo):
        monkeypatch.setattr(cli, "configure_logging", lambda **kwargs: None)
        monkeypatch.setattr(cli.logger, "propagate", False)
        cli.logger.addHandler(caplog.handler)
    yield
    for cli in (cli_conservation, cli_dataset, cli_kinaseinfo):
        cli.logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    "bool_data,bool_config_data,expected",
    [
        (None, True, True),
        (None, False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_resolve_data(bool_data, bool_config_data, expected):
    """An explicit flag wins over the config's ``data``."""
    cfg = DataFiguresConfig(data=bool_config_data)
    assert cfg.resolve_data(bool_data, bool_figs=True) is expected


def test_resolve_data_nothing_to_do():
    """Data off plus figures off raises."""
    with pytest.raises(ArgumentError, match="nothing to do"):
        DataFiguresConfig(data=False).resolve_data(None, bool_figs=False)


def test_task_config_hierarchy():
    """Data-bearing tasks share ``data``/``matplotlib_rc``; PyMOL shares only ``output``."""
    for cls in (
        KinaseInfoFiguresConfig,
        ConservationFiguresConfig,
        DatasetFiguresConfig,
    ):
        assert issubclass(cls, DataFiguresConfig)
    assert issubclass(PymolConfig, TaskConfig)
    assert not issubclass(PymolConfig, DataFiguresConfig)


def test_pymol_config_rejects_data_key(tmp_path):
    """PyMOL has no data step, so a ``pymol.data`` key is a config error."""
    path_config = tmp_path / "study.yaml"
    path_config.write_text("pymol:\n  data: true\n")
    with pytest.raises(ConfigKeyError):
        load_task_config(PymolConfig, str(path_config), "pymol")


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], {"bool_data": None, "bool_figs": True, "force": False}),
        (["--data"], {"bool_data": True, "bool_figs": True, "force": False}),
        (["--no-data"], {"bool_data": False, "bool_figs": True, "force": False}),
        (["--no-figs"], {"bool_data": None, "bool_figs": False, "force": False}),
        (["--recompute"], {"bool_data": None, "bool_figs": True, "force": True}),
    ],
)
def test_kinaseinfo_cli_flags(monkeypatch, args, expected):
    """The kinaseinfo CLI passes an unset ``--data`` through as None for the config to decide."""
    calls = []
    monkeypatch.setattr(cli_kinaseinfo.pipeline, "run", lambda **kw: calls.append(kw))
    result = RUNNER.invoke(cli_kinaseinfo.app, args)
    assert result.exit_code == 0, result.output
    assert {key: calls[0][key] for key in expected} == expected


def test_kinaseinfo_cli_catches_only_argument_errors(monkeypatch, caplog):
    """Argument errors exit 1 with a message; other ValueErrors keep their traceback."""

    def _raising(exc):
        def _run(**kwargs):
            raise exc

        return _run

    monkeypatch.setattr(
        cli_kinaseinfo.pipeline, "run", _raising(ArgumentError("bad flag combo"))
    )
    result = RUNNER.invoke(cli_kinaseinfo.app, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "bad flag combo" in caplog.text

    monkeypatch.setattr(
        cli_kinaseinfo.pipeline, "run", _raising(ValueError("data validation bug"))
    )
    result = RUNNER.invoke(cli_kinaseinfo.app, [])
    assert type(result.exception) is ValueError
    assert "data validation bug" in str(result.exception)


def _write_config(tmp_path, str_task, bool_data):
    """Write a study YAML setting ``<str_task>.data`` and return its path."""
    path_config = tmp_path / "study.yaml"
    path_config.write_text(f"{str_task}:\n  data: {str(bool_data).lower()}\n")
    return str(path_config)


@pytest.mark.parametrize(
    "args,config_data,expected",
    [
        ([], None, ["build", "figures"]),
        (["--no-figs"], None, ["build"]),
        (["--no-data"], None, ["figures"]),
        ([], False, ["figures"]),
        (["--data"], False, ["build", "figures"]),
    ],
)
def test_dataset_cli_flags(tmp_path, monkeypatch, args, config_data, expected):
    """The dataset CLI builds and/or renders per the resolved flags."""
    calls = []
    monkeypatch.setattr(cli_dataset, "_build_datasets", lambda: calls.append("build"))
    monkeypatch.setattr(
        cli_dataset, "_run_figures", lambda config_path: calls.append("figures")
    )
    if config_data is not None:
        args = args + ["--config", _write_config(tmp_path, "dataset", config_data)]
    result = RUNNER.invoke(cli_dataset.app, args)
    assert result.exit_code == 0, result.output
    assert calls == expected


@pytest.mark.parametrize(
    "args,config_data",
    [(["--no-data"], None), ([], False)],
)
def test_conservation_cli_no_data_renders_figures(
    tmp_path, monkeypatch, args, config_data
):
    """With data off, the conservation CLI renders figures without building the engine."""
    calls = []
    monkeypatch.setattr(
        cli_conservation, "_run_figures", lambda config_path: calls.append("figures")
    )
    if config_data is not None:
        args = args + ["--config", _write_config(tmp_path, "conservation", config_data)]
    result = RUNNER.invoke(cli_conservation.app, args)
    assert result.exit_code == 0, result.output
    assert calls == ["figures"]


@pytest.mark.parametrize("cli", [cli_conservation, cli_dataset])
def test_cli_no_data_no_figs_exits(monkeypatch, caplog, cli):
    """``--no-data --no-figs`` exits non-zero with a message, before doing any work."""
    calls = []
    monkeypatch.setattr(
        cli, "_run_figures", lambda config_path: calls.append("figures")
    )
    result = RUNNER.invoke(cli.app, ["--no-data", "--no-figs"])
    assert result.exit_code == 1
    assert calls == []
    assert "nothing to do" in caplog.text
