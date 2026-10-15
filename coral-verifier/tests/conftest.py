import os, sys
here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(here, "..", "verifier"), "/app"):
    if os.path.isdir(p):
        sys.path.insert(0, p)

import pytest


@pytest.fixture(autouse=True)
def _no_auto_validation(monkeypatch):
    """The background auto-validation is off unless a test turns it on (Journal(..., auto_validate=True)): episodes
    closed as vetoed would otherwise wait for it, and the existing tests would race it."""
    import journal
    monkeypatch.setitem(journal.J, "auto_validate", False)
