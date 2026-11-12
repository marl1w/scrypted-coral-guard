"""0.3.60: the journal stores each vehicle detection with the tracker's track under it; auto-validation's "the car moved" uses
the same rule as the tracker (one side redrawn by the detector is not a move)."""
from journal import boxes_moved, tag_vehicle_tracks

LOW_IN = [0.147, 0.558, 0.568, 0.985]
LOW_EDGE = [0.0005, 0.558, 0.569, 0.994]


def test_vehicle_detections_carry_their_track_and_others_are_untouched():
    dets = [{"cls": "car", "group": "vehicle", "score": 0.8, "box": LOW_EDGE},
            {"cls": "person", "group": "person", "score": 0.9, "box": [0.7, 0.4, 0.8, 0.9]},
            {"cls": "car", "group": "vehicle", "score": 0.5, "box": [0.8, 0.0, 0.9, 0.1]}]
    tracks = [{"id": 7, "state": "parked", "box": LOW_IN}, {"id": 9, "state": "moving", "box": [0.3, 0.2, 0.6, 0.6]}]
    out = tag_vehicle_tracks(dets, tracks)
    assert out[0]["track"] == [7, "parked"]
    assert "track" not in out[1] and "track" not in out[2]       # a person; a vehicle under no track
    assert "track" not in dets[0]                                 # copies: the reply's detections are untouched
    assert tag_vehicle_tracks(dets, None) == dets


def test_a_box_flipping_to_the_frame_edge_did_not_move_but_a_car_that_drove_off_did():
    assert not boxes_moved([[LOW_IN], [LOW_EDGE], [LOW_IN]])
    assert boxes_moved([[LOW_IN], [[0.45, 0.558, 0.87, 0.985]]])
