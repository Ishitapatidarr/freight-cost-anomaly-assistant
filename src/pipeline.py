"""Deterministic freight cost anomaly pipeline with evidence-gated note retrieval."""
from __future__ import annotations
import argparse, json, re, hashlib
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id", "reason"]
# Transparent policy parameters; chosen defaults, not supplied by the case prompt.
ANOMALY_THRESHOLD = 0.20  # at least 20% above either available baseline
MIN_ABSOLUTE_RISE = 0.05  # also require a positive rise vs previous observed week


def normalize_route(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace("–", "-").replace("—", "-").casefold()


def load_inputs(data_dir: Path = INPUT):
    shipments = pd.read_csv(data_dir / "shipment_records.csv")
    notes = pd.read_csv(data_dir / "context_notes.csv").fillna("")
    template = pd.read_csv(data_dir / "sample_output_format_v2.csv", dtype=str, nrows=0)
    required = {"origin", "destination", "route_type", "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date"}
    missing = required - set(shipments.columns)
    if missing:
        raise ValueError(f"shipment_records.csv missing required columns: {sorted(missing)}")
    if not set(OUTPUT_COLUMNS).issubset(template.columns):
        raise ValueError(f"Output template must include: {OUTPUT_COLUMNS}")
    shipments["shipment_date"] = pd.to_datetime(shipments["shipment_date"], errors="coerce")
    for c in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        shipments[c] = pd.to_numeric(shipments[c], errors="coerce")
    bad = shipments["shipment_date"].isna() | shipments[["quantity_tonnes", "distance_km", "freight_cost_inr"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"Found {int(bad.sum())} rows with invalid date/quantity/distance/cost; fix inputs rather than silently dropping them.")
    if (shipments["quantity_tonnes"] <= 0).any() or (shipments["distance_km"] <= 0).any() or (shipments["freight_cost_inr"] < 0).any():
        raise ValueError("Quantity and distance must be positive; freight cost must be non-negative.")
    shipments["week_of"] = shipments["shipment_date"].dt.to_period("W-SUN").dt.start_time
    shipments["route"] = shipments["origin"].astype(str).str.strip() + "-" + shipments["destination"].astype(str).str.strip()
    return shipments, notes, template


def aggregate_weekly(shipments: pd.DataFrame) -> pd.DataFrame:
    # Correct weighted formula: sum(cost) / sum(quantity * distance). This also handles
    # shipment-level distance variation without assuming every row has identical km.
    s = shipments.copy()
    if "week_of" not in s.columns:
        s["shipment_date"] = pd.to_datetime(s["shipment_date"], errors="raise")
        s["week_of"] = s["shipment_date"].dt.to_period("W-SUN").dt.start_time
    if "route" not in s.columns:
        s["route"] = s["origin"].astype(str).str.strip() + "-" + s["destination"].astype(str).str.strip()
    s["tonne_km"] = s["quantity_tonnes"] * s["distance_km"]
    keys = ["origin", "destination", "route_type", "route", "week_of"]
    weekly = s.groupby(keys, as_index=False, sort=True).agg(
        total_cost=("freight_cost_inr", "sum"), total_tonne_km=("tonne_km", "sum"),
        shipment_count=("shipment_id", "count") if "shipment_id" in s.columns else ("route", "count")
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    weekly = weekly.sort_values(["origin", "destination", "route_type", "week_of"]).reset_index(drop=True)
    route_keys = ["origin", "destination", "route_type"]
    weekly["previous_week_cost"] = weekly.groupby(route_keys)["cost_per_tonne_km"].shift(1)
    weekly["vs_own_history_value"] = weekly.groupby(route_keys)["cost_per_tonne_km"].transform(
        lambda x: x.shift(1).rolling(window=8, min_periods=1).mean()
    )
    # Every route-week contributes one weekly route rate to its type's peer mean.
    def peer_mean(group):
        n = len(group)
        return (group.sum() - group) / (n - 1) if n > 1 else pd.Series(np.nan, index=group.index)
    weekly["vs_similar_value"] = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"].transform(peer_mean)
    return weekly


def _note_effective_window(note):
    """Conservative temporal interpretation; avoid treating dated one-off notes as evergreen."""
    d = pd.to_datetime(note["date"], errors="coerce")
    if pd.isna(d):
        return None, None
    text = str(note["note"]).casefold()
    # Explicit flood window in N001 (Feb 24 through Mar 8).
    m = re.search(r"from\s+([a-z]{3,9}\s+\d{1,2})\s+to\s+([a-z]{3,9}\s+\d{1,2})", text)
    if m:
        year = d.year
        start = pd.to_datetime(f"{m.group(1)} {year}", errors="coerce")
        end = pd.to_datetime(f"{m.group(2)} {year}", errors="coerce")
        if pd.notna(start) and pd.notna(end):
            if start < d - pd.Timedelta(days=180): start += pd.DateOffset(years=1)
            if end < start: end += pd.DateOffset(years=1)
            return start.normalize(), end.normalize()
    # Nationwide diesel note says the increase starts that week; conservatively allow 14 days.
    if "starting this week" in text:
        return d.normalize(), (d + pd.Timedelta(days=13)).normalize()
    # Other event notes without duration are treated as a one-week event, not evergreen.
    return d.normalize(), (d + pd.Timedelta(days=6)).normalize()


def _explicitly_supports_cost_increase(note_text: str) -> bool:
    t = note_text.casefold()
    positive = any(k in t for k in ["higher trip costs", "surcharge", "pushing up transportation costs", "costs increased", "increased costs", "raised costs", "higher costs", "rate increase"])
    negated = any(k in t for k in ["costs were not significantly affected", "without a rate change", "no significant disruptions", "stable demand", "improved road conditions", "returned to normal", "not part of this dataset"])
    return positive and not negated


def retrieve_supporting_note(route: str, week_of: pd.Timestamp, notes: pd.DataFrame):
    """TF-IDF retrieves candidates; hard route/date/causality gates decide eligibility.

    This is deliberately evidence-first: similarity never overrides the hard checks.
    Returns (note row or None, retrieval diagnostics).
    """
    if notes.empty:
        return None, {"candidate_count": 0, "reason": "no notes"}
    docs = (notes["applies_to"].astype(str) + " " + notes["note"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform(docs + [route.replace("-", " ") + " freight cost increase"])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        sims = np.zeros(len(notes))
    candidates = sorted(range(len(notes)), key=lambda i: (-float(sims[i]), str(notes.iloc[i]["note_id"])))
    route_norm = normalize_route(route)
    eligible = []
    for i in candidates:
        note = notes.iloc[i]
        applies = str(note["applies_to"]).strip()
        route_match = applies.casefold() == "all routes" or normalize_route(applies) == route_norm
        if not route_match:
            continue
        start, end = _note_effective_window(note)
        if start is None or not (start <= week_of.normalize() <= end):
            continue
        if not _explicitly_supports_cost_increase(str(note["note"])):
            continue
        eligible.append((float(sims[i]), str(note["note_id"]), note))
    if not eligible:
        return None, {"candidate_count": len(candidates), "reason": "no note passed route/date/cost-impact checks"}
    eligible.sort(key=lambda x: (-x[0], x[1]))
    return eligible[0][2], {"candidate_count": len(candidates), "similarity": eligible[0][0], "reason": "passed all evidence gates"}


def pct_text(current, baseline, label):
    if pd.isna(baseline) or baseline <= 0:
        return f"N/A vs {label} (insufficient baseline data)"
    change = (current / baseline - 1) * 100
    return f"{change:+.1f}% vs {label}"


def run_pipeline(data_dir: Path = INPUT, output_dir: Path = OUTPUT, threshold: float = ANOMALY_THRESHOLD):
    shipments, notes, template = load_inputs(data_dir)
    weekly = aggregate_weekly(shipments)
    out_rows = []
    for _, r in weekly.iterrows():
        cur = float(r["cost_per_tonne_km"])
        hist = r["vs_own_history_value"]
        peer = r["vs_similar_value"]
        prev = r["previous_week_cost"]
        # Require a positive increase from the previous observed week, and abnormality
        # against at least one available prescribed baseline. Missing baselines do not count.
        rising = pd.notna(prev) and cur > float(prev) * (1 + MIN_ABSOLUTE_RISE)
        hist_abnormal = pd.notna(hist) and float(hist) > 0 and cur >= float(hist) * (1 + threshold)
        peer_abnormal = pd.notna(peer) and float(peer) > 0 and cur >= float(peer) * (1 + threshold)
        flagged = bool(rising and (hist_abnormal or peer_abnormal))
        if not flagged:
            continue
        note, diag = retrieve_supporting_note(r["route"], pd.Timestamp(r["week_of"]), notes)
        if note is not None:
            note_id = str(note["note_id"])
            reason = f"No anomaly verdict was changed by AI. Increase is supported by note {note_id} dated {note['date']}: {str(note['note']).strip()}"
            flag_value = "No (justified)"
        else:
            note_id = ""
            reason = "Cost increase is anomalous against the available baseline(s), but no note was found that supports a cost increase for this exact route and week. Unexplained; review recommended."
            flag_value = "Yes"
        out_rows.append({
            "route": r["route"], "week_of": pd.Timestamp(r["week_of"]).strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(cur, 2),
            "vs_own_history": pct_text(cur, hist, "this route's past average"),
            "vs_similar_routes": pct_text(cur, peer, "similar-length routes this week"),
            "flagged": flag_value, "matched_note_id": note_id, "reason": reason,
        })
    result = pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "freight_anomalies.csv"
    # Ensure exact required output column order and stable row order.
    if not result.empty:
        result = result.sort_values(["week_of", "route"], kind="mergesort").reset_index(drop=True)
    result.to_csv(output_file, index=False)
    # Machine-readable audit log; no LLM was called, so LLM token/cost counts are honestly zero.
    run_log = {
        "shipment_rows": int(len(shipments)), "weekly_route_rows": int(len(weekly)),
        "anomaly_rows": int(len(result)), "justified_rows": int((result["flagged"] == "No (justified)").sum()) if not result.empty else 0,
        "unexplained_rows": int((result["flagged"] == "Yes").sum()) if not result.empty else 0,
        "llm_calls": 0, "llm_input_tokens": 0, "llm_output_tokens": 0, "llm_cost_usd": 0.0,
        "retrieval": "scikit-learn TF-IDF candidate ranking + deterministic route/date/cost-impact gates",
        "threshold": threshold, "min_previous_week_rise": MIN_ABSOLUTE_RISE,
        "output_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
    }
    (output_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    return result, run_log



def answer_question(question: str, data_dir: Path = INPUT, output_dir: Path = OUTPUT) -> str:
    """Evidence-grounded Q&A over generated anomaly results."""
    result_path = output_dir / "freight_anomalies.csv"

    if not result_path.exists():
        run_pipeline(data_dir, output_dir)

    df = pd.read_csv(result_path).fillna("")
    _, notes, _ = load_inputs(data_dir)
    q = question.casefold()

    if df.empty:
        return "No flagged route/week records were found in the generated results."

    # 1. Filter by route, but don't silently return unrelated routes.
    known_routes = sorted(df["route"].astype(str).unique())
    route_candidates = [
        route for route in known_routes
        if normalize_route(route) in normalize_route(question)
        or all(part.casefold() in q for part in route.split("-"))
    ]

    # If the question contains a route-like name but it isn't in the
    # anomaly results, report no match instead of showing unrelated records.
    route_pattern = re.search(
        r"\b(?:on|for|about|did|increase|increased|pricier)\s+"
        r"([A-Za-z]+(?:-[A-Za-z]+){1,3})\b",
        question,
        re.IGNORECASE,
    )

    if route_candidates:
        df = df[df["route"].isin(route_candidates)]
    elif route_pattern:
        requested_route = route_pattern.group(1)
        return (
            f"I couldn't find a matching flagged route/week for "
            f"{requested_route} in the generated results."
        )

    # 2. Filter by month. A record can match when its week starts in
    # the month OR its supporting note is effective during that month.
    month_map = {
        m.casefold(): i
        for i, m in enumerate(
            [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December",
            ],
            1,
        )
    }

    requested_month = next(
        ((name, month) for name, month in month_map.items() if name in q),
        None,
    )

    year_match = re.search(r"\b(20\d{2})\b", q)

    if requested_month:
        _, month = requested_month
        keep_indices = []

        for idx, row in df.iterrows():
            week_date = pd.to_datetime(row["week_of"], errors="coerce")

            if pd.isna(week_date):
                continue

            if year_match and week_date.year != int(year_match.group(1)):
                continue

            # The anomaly week itself falls in the requested month.
            matches_month = week_date.month == month

            # Otherwise, check whether its matched evidence note overlaps
            # any day in the requested month.
            note_id = str(row.get("matched_note_id", "")).strip()

            if not matches_month and note_id:
                note_rows = notes[
                    notes["note_id"].astype(str).eq(note_id)
                ]

                if not note_rows.empty:
                    note_start, note_end = _note_effective_window(
                        note_rows.iloc[0]
                    )

                    if note_start is not None and note_end is not None:
                        month_start = pd.Timestamp(
                            year=week_date.year, month=month, day=1
                        )
                        month_end = month_start + pd.offsets.MonthEnd(0)

                        matches_month = (
                            note_start <= month_end
                            and note_end >= month_start
                        )

            if matches_month:
                keep_indices.append(idx)

        df = df.loc[keep_indices]

    # 3. Filter by explicitly mentioned event types.
    # This only searches evidence-note text; it does not infer causation.
    event_terms = {
        "earthquake": ["earthquake", "seismic activity"],
        "flood": ["flood", "flooding"],
        "rain": ["rain", "rainfall", "heavy rain"],
        "fuel": ["fuel price", "diesel price", "fuel surcharge"],
        "toll": ["toll increase", "toll hike", "toll rate"],
        "festival": ["festival", "holiday demand"],
        "strike": ["strike", "labor strike", "labour strike"],
    }

    requested_events = [
        terms
        for event, terms in event_terms.items()
        if event in q
    ]

    if requested_events:
        note_lookup = notes.set_index("note_id")["note"].astype(str).to_dict()
        matching_indices = []

        for idx, row in df.iterrows():
            note_id = str(row.get("matched_note_id", "")).strip()
            note_text = note_lookup.get(note_id, "").casefold()

            if any(
                term in note_text
                for terms in requested_events
                for term in terms
            ):
                matching_indices.append(idx)

        df = df.loc[matching_indices]

    # 4. Never fall back to unrelated results if filters find no match.
    if df.empty:
        return (
            "I couldn't find a matching flagged route/week and supporting "
            "evidence for that question in the generated results. "
            "This does not prove the event had no effect; the available "
            "notes may simply not document it."
        )

    # 5. Return only matching records and their existing evidence.
    lines = [f"Found {len(df)} matching anomaly record(s):"]

    for _, row in df.head(10).iterrows():
        note_id = str(row.get("matched_note_id", "")).strip()
        evidence = note_id or "no supporting note"

        lines.append(
            f"- {row['route']} in week {row['week_of']}: "
            f"cost/tonne-km {row['cost_per_tonne_km']}; "
            f"own-history comparison {row['vs_own_history']}; "
            f"peer comparison {row['vs_similar_routes']}; "
            f"verdict {row['flagged']}; evidence {evidence}. "
            f"{row['reason']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Freight cost anomaly assistant")
    parser.add_argument("--data-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD, help="Relative anomaly threshold, e.g. 0.20 for 20%%")
    parser.add_argument("--ask", type=str, help="Ask a plain-English question about generated findings")
    args = parser.parse_args()
    result, log = run_pipeline(args.data_dir, args.output_dir, args.threshold)
    print(f"Wrote {len(result)} anomaly rows to {args.output_dir / 'freight_anomalies.csv'}")
    print(json.dumps(log, indent=2))
    if args.ask:
        print("\n" + answer_question(args.ask, args.data_dir, args.output_dir))

if __name__ == "__main__":
    main()
