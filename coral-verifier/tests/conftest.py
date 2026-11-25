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


@pytest.fixture(autouse=True)
def _no_frozen_objects(request, monkeypatch):
    """0.3.65 frozen objects (defaults.FROZEN) are off in the tests written before it: their synthetic people stand on pixel-identical boxes from
    the first frame, which the rule reads as a still object. tests/test_frozen.py tests it (and its interplay with the trackers) with it on."""
    import defaults
    if request.module.__name__.split(".")[-1] != "test_frozen":
        monkeypatch.setitem(defaults.FROZEN, "enabled", False)
