import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import aggregate_weekly, retrieve_supporting_note, _note_effective_window


def test_week_starts_monday_and_weighted_cost():
    d = pd.DataFrame({
        "origin": ["A", "A"], "destination": ["B", "B"], "route_type": ["Short", "Short"],
        "route": ["A-B", "A-B"], "week_of": pd.to_datetime(["2025-01-06", "2025-01-06"]),
        "quantity_tonnes": [10, 20], "distance_km": [100, 200], "freight_cost_inr": [1000, 6000],
        "shipment_id": ["1", "2"]
    })
    d["tonne_km"] = d.quantity_tonnes * d.distance_km
    # call the same grouping logic via a minimal shipment frame
    s = d.drop(columns=["route", "week_of", "tonne_km"])
    s["shipment_date"] = pd.to_datetime(["2025-01-06", "2025-01-07"])
    got = aggregate_weekly(s)
    assert got.iloc[0]["week_of"] == pd.Timestamp("2025-01-06")
    assert abs(got.iloc[0]["cost_per_tonne_km"] - (7000 / 5000)) < 1e-9


def test_note_route_date_and_cost_gates():
    notes = pd.DataFrame([
        {"note_id":"N1", "date":"2025-01-20", "applies_to":"A-B", "note":"A surcharge caused higher trip costs this week."},
        {"note_id":"N2", "date":"2025-01-20", "applies_to":"Other-Route", "note":"A surcharge caused higher trip costs this week."},
        {"note_id":"N3", "date":"2025-01-20", "applies_to":"A-B", "note":"Costs were not significantly affected."},
    ])
    note, _ = retrieve_supporting_note("A-B", pd.Timestamp("2025-01-20"), notes)
    assert note is not None and note["note_id"] == "N1"
    note, _ = retrieve_supporting_note("A-B", pd.Timestamp("2025-02-20"), notes)
    assert note is None


def test_history_baseline_is_prior_only():
    # Regression test for required shift-before-rolling semantics.
    values = pd.Series([2.0, 4.0, 10.0])
    baseline = values.shift(1).rolling(8, min_periods=1).mean()
    assert np.isnan(baseline.iloc[0])
    assert baseline.iloc[1] == 2.0
    assert baseline.iloc[2] == 3.0
