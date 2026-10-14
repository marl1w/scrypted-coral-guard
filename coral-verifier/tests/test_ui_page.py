"""The review page's script is EXECUTED (node, fake DOM, stubbed fetch): Load more / empty messages for Events, To review and Reviewed."""
import json, os, re, shutil, subprocess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "..", "verifier", "ui.html")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


@pytest.fixture(scope="module")
def run():
    p = subprocess.run(["node", os.path.join(HERE, "ui_harness.js"), UI], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_every_load_more_button_and_empty_message_starts_hidden(run):
    assert run["initial"] == {"more": True, "rToMore": True, "rDoMore": True, "empty": True, "rToEmpty": True, "rDoEmpty": True}


@pytest.mark.parametrize("name", ["events", "to_review", "reviewed"])
def test_load_more_follows_loaded_versus_total(run, name):
    s = run["scenarios"][name]
    assert s["empty"] == {"hidden": True, "emptyShown": True}                         # empty list: no button, ONE empty message
    assert s["firstPage"] == {"hidden": False, "emptyShown": False}                   # a full page with more to come: visible
    assert s["middlePage"]["hidden"] is False
    assert s["lastPage"]["hidden"] is True                                            # final (short) page: hidden
    assert s["exactlyOnePage"]["hidden"] is True                                      # loaded == total: hidden
    assert s["emptyThenMoreClicks"] == {"hidden": True, "emptyShown": True}           # clicking on an empty list changes nothing


def test_empty_list_never_adds_cards_to_the_grid(run):
    assert run["toReviewGridChildren"] == 0


def test_hidden_attribute_cannot_be_overridden_by_a_display_rule():
    css = re.search(r"<style>(.*?)</style>", open(UI).read(), re.S).group(1)
    assert "[hidden]{display:none!important}" in css                                    # `.more{display:block}` used to keep the button visible although hidden
    assert ".more{display:block" in css or ".more{" in css                              # the rule that caused it is still there, so the override is what protects us


# ---------------- the Filter menu (executed) ----------------

def test_filter_menu_open_close_and_keyboard(run):
    m = run["menu"]
    assert m["initial"] == {"menuHidden": True, "expanded": "false", "stateHidden": True, "text": "Filter"}          # one outlined Filter button, no dot at the defaults
    assert m["open"]["menuHidden"] is False and m["open"]["expanded"] == "true" and m["open"]["focusOnChecked"] is True
    assert m["keyboard"] == {"down": True, "end": True, "wrapDown": True, "home": True, "wrapUp": True, "items": m["keyboard"]["items"]} and m["keyboard"]["items"] > 10
    assert m["escape"] == {"menuHidden": True, "focusOnButton": True} and m["tab"]["menuHidden"] is True and m["outside"]["menuHidden"] is True


def test_filter_menu_groups_per_tab_and_camera_names(run):
    m = run["menu"]
    assert m["groupsEvents"] == ["camera", "category", "outcome", "period", "from", "to"]
    assert m["groupsValidate"] == m["groupsEvents"]
    assert m["groupsReviewed"] == ["camera", "category", "outcome", "label", "period", "from", "to"]               # Label only in Reviewed
    assert m["groupsSummary"] == ["camera", "category", "period", "from", "to"]
    assert m["cameraOptions"] == [["", "All"], ["door_2", "Door"], ["garden_1", "Garden"]]                          # display names, sorted
    labels = m["open"]["labels"]
    for needle in ("person", "vehicle", "animal", "confirmed", "vetoed", "unflagged", "24 h", "7 days", "14 days", "All time"):
        assert needle in labels


def test_filter_choices_change_filters_requests_words_and_hash(run):
    m = run["menu"]
    c = m["camera"]
    assert c["filter"] == "garden_1" and c["menuHidden"] is True and c["request"] is True and c["hash"] == "#t=ev&ev.camera=garden_1" and c["summary"] == "Garden"
    w = m["words"]
    assert w["summary"] == "Garden \u00b7 person \u00b7 7 days" and w["stateHidden"] is False and w["dotShown"] is True and w["requestHasAll"] is True
    assert w["hash"] == "#t=ev&ev.camera=garden_1&ev.category=person&ev.period=7d"
    assert m["outcome"] == {"filter": "confirmed", "request": True}
    assert m["dates"] == {"filter": "2026-10-01", "summary": "2026-10-01 \u2013 \u2026"}                              # dates replace the period words
    assert m["summaryDefault"] == "" and m["summaryChoice"] == "14 days"                                          # Summary's default period (7 days) is not shown


def test_filter_reset_restores_defaults(run):
    m = run["menu"]
    assert m["hasReset"] is True
    r = m["reset"]
    assert json.loads(r["filter"]) == {"camera": "", "category": "", "outcome": "", "label": "", "period": "all", "from": "", "to": ""}
    assert r["stateHidden"] is True and r["dotHidden"] is True and r["summary"] == "" and r["hash"] == "#t=ev"


def test_filters_per_tab_reach_their_own_requests_and_counts_are_one_line(run):
    m = run["menu"]
    assert m["validateRequest"] == "v1/journal/next?camera=door_2&group=vehicle"
    assert m["countsLine"] == "6 reviewed \u00b7 2 to review \u00b7 1 in progress"                                  # a single compact line, no chips


def test_filters_and_tab_persist_in_the_url_hash(run):
    h = run["hash"]
    assert h["tab"] == "re" and h["reviewedShown"] is True and h["eventsShown"] is False
    assert h["re"] == {"camera": "door_2", "category": "", "outcome": "", "label": "missed", "period": "14d", "from": "", "to": ""}
    assert h["su"]["category"] == "animal" and h["su"]["period"] == "7d" and h["ev"]["from"] == "2026-10-01"
    assert h["ev"]["period"] == "all" and h["va"]["from"] == ""                                                    # invalid values in the hash are ignored
    assert h["summaryText"] == "Door \u00b7 missed \u00b7 14 days" and h["request"] is True                       # and they drive the first requests


def test_old_selects_and_chip_row_are_gone():
    html = open(UI).read()
    for gone in ('id="fCam"', 'id="vCam"', 'id="rCam"', 'id="suCam"', 'id="fGo"', 'id="suPer"', 'class="chips"></span>'):
        assert gone not in html
