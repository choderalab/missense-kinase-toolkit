"""Headless smoke test mirroring the Streamlit Cloud deployment.

Runs ``app.py`` end-to-end through streamlit's ``AppTest`` so CI catches import
errors, render exceptions, and native-dependency regressions before they reach
https://mkt-app.streamlit.app.

This is the guard that would have caught the pyarrow 25.0.0 mimalloc segfault:
``st.table`` forces Arrow serialization of the property tables, so a bad wheel
crashes the test process (exit 139) and turns the job red.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_default_kinase():
    """The default kinase renders without raising, exercising the Arrow path."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()

    assert not at.exception, f"app raised on first render: {at.exception}"
    # st.title -> at.title; st.subheader -> at.subheader (the selected kinase)
    assert any("KinaseInfo Dashboard" in t.value for t in at.title)
    assert at.subheader, "expected a 'Selected Kinase' subheader"

    # the original crash surfaced on a rerun (streamlit reruns on interaction),
    # so exercise a second pass through the script as well
    at.run()
    assert not at.exception, f"app raised on rerun: {at.exception}"
