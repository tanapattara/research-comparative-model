#!/usr/bin/env python3
"""Generate publication figures from the frozen E0--E5 aggregate evidence only.

The script deliberately uses a fixed allowlist rooted at
``reproducibility/frozen_e0_e5``. It neither imports model code nor reads raw
predictions. ``--check`` validates schemas, filters, sign conventions, and
lineage without creating an output directory or rendering figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
MANUSCRIPT_DIR = SCRIPT_PATH.parents[1]
SUITE_ROOT = SCRIPT_PATH.parents[2]
FROZEN_ROOT = (SUITE_ROOT / "reproducibility" / "frozen_e0_e5").resolve()
DEFAULT_OUTPUT_DIR = MANUSCRIPT_DIR / "figures"

SOURCE_FILES = {
    "e0": "manifests/e0_data_manifest.json",
    "e1": "manifests/e1_strategy_manifest.json",
    "e2": "manifests/e2_architecture_manifest.json",
    "e3": "manifests/e3_lookback_manifest.json",
    "e4": "manifests/e4_stations_manifest.json",
    "e5": "manifests/e5_final_manifest.json",
    "mae": "summaries/e5_final_metric_summary.csv",
    "effect": "summaries/e5_s4_vs_s0_inference.csv",
    "skill": "summaries/e5_persistence_skill_inference.csv",
    "station": "summaries/e4_station_contributions.csv",
    "event": "summaries/e5_high_water_event_metrics.csv",
}

FIGURE_NAMES = (
    "fig1_frozen_workflow",
    "fig2_horizon_mae_and_upstream_effect",
    "fig3_predictive_skill_horizon",
    "figS1_station_contribution",
    "figS2_event_sensitivity",
)
FORMATS = ("svg", "pdf", "png")

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#777777",
}

AUDIT_FIELDS = (
    "figure_id",
    "panel",
    "series",
    "item",
    "source_path",
    "source_sha256",
    "row_filter",
    "field",
    "transformation",
    "source_value_unrounded",
    "derived_value_unrounded",
    "displayed_value",
    "evidence_class",
    "notes",
)


class FigureDataError(RuntimeError):
    """Raised when frozen evidence does not satisfy the required contract."""


@dataclass(frozen=True)
class Bundle:
    paths: dict[str, Path]
    hashes: dict[str, str]
    manifests: dict[str, dict[str, Any]]
    mae_rows: list[dict[str, str]]
    effect_rows: list[dict[str, str]]
    skill_rows: list[dict[str, str]]
    station_rows: list[dict[str, str]]
    event_rows: list[dict[str, str]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(relative: str) -> Path:
    if "posthoc" in relative.lower() or "post-hoc" in relative.lower():
        raise FigureDataError(f"Post-hoc source is forbidden: {relative}")
    path = (FROZEN_ROOT / relative).resolve()
    try:
        path.relative_to(FROZEN_ROOT)
    except ValueError as exc:
        raise FigureDataError(f"Source escapes frozen evidence root: {path}") from exc
    if not path.is_file():
        raise FigureDataError(f"Required frozen source is missing: {path}")
    return path


def display_path(path: Path) -> str:
    return path.relative_to(SUITE_ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise FigureDataError(f"Expected JSON object: {path}")
    return data


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise FigureDataError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise FigureDataError(f"Frozen CSV is empty: {path}")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureDataError(message)


def as_float(value: str, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureDataError(f"Expected finite numeric {field}, received {value!r}") from exc
    if not math.isfinite(number):
        raise FigureDataError(f"Expected finite numeric {field}, received {value!r}")
    return number


def finite_or_none(value: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def load_and_validate() -> Bundle:
    paths = {key: source_path(relative) for key, relative in SOURCE_FILES.items()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    manifests = {key: read_json(paths[key]) for key in ("e0", "e1", "e2", "e3", "e4", "e5")}

    mae_rows = read_csv(
        paths["mae"],
        {"station_set", "horizon_days", "MAE_m_mean", "MAE_m_std"},
    )
    effect_rows = read_csv(
        paths["effect"],
        {
            "comparison",
            "horizon_days",
            "block_length_days",
            "mean_MAE_improvement_m",
            "ci_low_m",
            "ci_high_m",
            "dm_holm_p_value",
        },
    )
    skill_rows = read_csv(
        paths["skill"],
        {
            "station_set",
            "horizon_days",
            "MAE_skill_vs_persistence",
            "skill_ci_low",
            "skill_ci_high",
            "NSE",
            "predictively_useful",
        },
    )
    station_rows = read_csv(
        paths["station"],
        {
            "analysis",
            "station",
            "horizon_days",
            "reference_set",
            "comparison_set",
            "MAE_improvement_m",
        },
    )
    event_rows = read_csv(
        paths["event"],
        {
            "fold",
            "station_set",
            "seed",
            "model",
            "horizon_days",
            "threshold_name",
            "POD_recall",
            "FAR",
            "CSI",
            "F1",
            "evidence_class",
        },
    )

    e0, e1, e2, e3, e4, e5 = (manifests[f"e{i}"] for i in range(6))
    require(e0.get("status") == "PASS", "E0 status must be PASS")
    require(e1.get("selected_strategy") == "lstm_mimo", "E1 must select lstm_mimo")
    require(e2.get("selected_strategy") == "mimo", "E2 strategy must be mimo")
    require(e2.get("selected_architecture") == "lstm", "E2 must select lstm")
    require(
        e2.get("status") == "COMPLETE_WITH_PROTOCOL_DEVIATION"
        and e2.get("architecture_tuning_trials_completed") == 0,
        "E2 protocol deviation contract changed",
    )
    require(e3.get("selected_lookback_days") == 7, "E3 must select a 7-day look-back")
    require(
        e4.get("evaluation_partition") == "validation_only"
        and e4.get("development_seed") == 42,
        "E4 must remain validation-only and single-seed",
    )
    require(
        e5.get("selected_strategy") == "mimo"
        and e5.get("selected_architecture") == "lstm"
        and e5.get("selected_lookback_days") == 7,
        "E5 frozen chain must be MIMO + LSTM + 7-day look-back",
    )
    require(e5.get("predictively_useful_horizon_days") == 6, "E5 useful horizon must be day 6")

    # The E0 -> E1 mismatch must remain visible; later manifest links must match.
    require(
        e1.get("e0_manifest_sha256") != hashes["e0"],
        "Historical E0 linkage unexpectedly matches; review Figure 1 semantics",
    )
    lineage = (
        ("e1", "e2", "e1_manifest_sha256"),
        ("e2", "e3", "e2_manifest_sha256"),
        ("e3", "e4", "e3_manifest_sha256"),
        ("e4", "e5", "e4_manifest_sha256"),
    )
    for predecessor, successor, field in lineage:
        require(
            manifests[successor].get(field) == hashes[predecessor],
            f"Frozen lineage mismatch for {predecessor.upper()} -> {successor.upper()}",
        )

    wanted_horizons = {1, 3, 5, 7, 14}
    mae_keys = {(row["station_set"], int(row["horizon_days"])) for row in mae_rows}
    require(
        mae_keys == {(station_set, horizon) for station_set in ("S0", "S4") for horizon in wanted_horizons},
        "MAE rows must be exactly S0/S4 at horizons 1, 3, 5, 7, 14",
    )
    for row in mae_rows:
        require(as_float(row["MAE_m_std"], field="MAE_m_std") >= 0, "MAE SD cannot be negative")

    primary_effects = [
        row
        for row in effect_rows
        if row["comparison"] == "S4_vs_S0" and int(row["block_length_days"]) == 30
    ]
    require(
        {int(row["horizon_days"]) for row in primary_effects} == {7, 14}
        and len(primary_effects) == 2,
        "Expected exactly day-7/day-14 30-day-block S4-vs-S0 rows",
    )
    for row in primary_effects:
        mean = as_float(row["mean_MAE_improvement_m"], field="mean_MAE_improvement_m")
        low = as_float(row["ci_low_m"], field="ci_low_m")
        high = as_float(row["ci_high_m"], field="ci_high_m")
        p_value = as_float(row["dm_holm_p_value"], field="dm_holm_p_value")
        require(mean > 0, "Positive sign must mean lower S4 MAE")
        require(low < 0 < high, "Both co-primary CIs must cross zero")
        require(p_value >= 0.05, "No significant marker is permitted for primary effects")

    require(
        len(skill_rows) == 14
        and {int(row["horizon_days"]) for row in skill_rows} == set(range(1, 15))
        and {row["station_set"] for row in skill_rows} == {"S4"},
        "Skill source must contain S4 horizons 1--14 exactly once",
    )
    useful = [row for row in skill_rows if row["predictively_useful"].strip().lower() == "true"]
    require(len(useful) == 1 and int(useful[0]["horizon_days"]) == 6, "Only day 6 may be useful")
    day7 = next(row for row in skill_rows if int(row["horizon_days"]) == 7)
    require(
        as_float(day7["skill_ci_low"], field="skill_ci_low") < 0
        < as_float(day7["skill_ci_high"], field="skill_ci_high"),
        "Day-7 skill CI must cross zero",
    )

    selected_station_rows = [row for row in station_rows if int(row["horizon_days"]) in {7, 14}]
    require(
        len(selected_station_rows) == 16
        and {row["analysis"] for row in selected_station_rows}
        == {"incremental_addition", "leave_one_out"},
        "Station figure requires 16 E4 day-7/day-14 rows",
    )

    selected_event_rows = [
        row
        for row in event_rows
        if row["threshold_name"] == "train_q95"
        and row["station_set"] in {"S0", "S4"}
        and int(row["horizon_days"]) in {7, 14}
    ]
    require(len(selected_event_rows) == 100, "Expected 100 train_q95 event rows")
    require(
        {row["evidence_class"] for row in selected_event_rows} == {"exploratory"},
        "Event rows must remain exploratory",
    )
    expected_rounded = {
        ("S0", 7, "POD_recall"): 0.3828,
        ("S0", 7, "FAR"): 0.3783,
        ("S0", 7, "CSI"): 0.3037,
        ("S0", 7, "F1"): 0.4136,
        ("S4", 7, "POD_recall"): 0.5971,
        ("S4", 7, "FAR"): 0.3786,
        ("S4", 7, "CSI"): 0.4365,
        ("S4", 7, "F1"): 0.5937,
        ("S0", 14, "POD_recall"): 0.1822,
        ("S0", 14, "FAR"): 0.5214,
        ("S0", 14, "CSI"): 0.1374,
        ("S0", 14, "F1"): 0.2228,
        ("S4", 14, "POD_recall"): 0.1721,
        ("S4", 14, "FAR"): 0.4839,
        ("S4", 14, "CSI"): 0.1380,
        ("S4", 14, "F1"): 0.2262,
    }
    for key, expected in expected_rounded.items():
        station_set, horizon, metric = key
        group = [
            row
            for row in selected_event_rows
            if row["station_set"] == station_set and int(row["horizon_days"]) == horizon
        ]
        require(len(group) == 25, f"Expected 25 event cells for {key}")
        finite = [value for row in group if (value := finite_or_none(row[metric])) is not None]
        require(finite, f"No finite event cells for {key}")
        require(round(sum(finite) / len(finite), 4) == expected, f"Event mean mismatch for {key}")

    return Bundle(
        paths=paths,
        hashes=hashes,
        manifests=manifests,
        mae_rows=mae_rows,
        effect_rows=effect_rows,
        skill_rows=skill_rows,
        station_rows=selected_station_rows,
        event_rows=selected_event_rows,
    )


def audit_row(
    bundle: Bundle,
    *,
    figure_id: str,
    panel: str,
    series: str,
    item: str,
    source_key: str,
    row_filter: str,
    field: str,
    transformation: str,
    source_value: str,
    derived_value: str,
    displayed_value: str,
    evidence_class: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "figure_id": figure_id,
        "panel": panel,
        "series": series,
        "item": item,
        "source_path": display_path(bundle.paths[source_key]),
        "source_sha256": bundle.hashes[source_key],
        "row_filter": row_filter,
        "field": field,
        "transformation": transformation,
        "source_value_unrounded": source_value,
        "derived_value_unrounded": derived_value,
        "displayed_value": displayed_value,
        "evidence_class": evidence_class,
        "notes": notes,
    }


def configure_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "paper2-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper2-frozen-e0-e5-figures-v1",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    fixed_date = datetime(2026, 8, 25, tzinfo=timezone.utc)
    outputs: list[Path] = []
    for extension in FORMATS:
        target = output_dir / f"{stem}.{extension}"
        common = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            fig.savefig(target, dpi=300, metadata={"Software": "Paper 2 frozen-evidence figure generator"}, **common)
        elif extension == "pdf":
            fig.savefig(
                target,
                metadata={
                    "Creator": "Paper 2 frozen-evidence figure generator",
                    "CreationDate": fixed_date,
                    "ModDate": fixed_date,
                },
                **common,
            )
        else:
            fig.savefig(
                target,
                metadata={"Creator": "Paper 2 frozen-evidence figure generator", "Date": "2026-08-25"},
                **common,
            )
            svg_text = target.read_text(encoding="utf-8")
            normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            target.write_text(normalized_svg, encoding="utf-8", newline="\n")
        outputs.append(target)
    return outputs


def plot_workflow(bundle: Bundle, plt: Any) -> tuple[Any, list[dict[str, str]]]:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    phases = [
        ("E0", "Data integrity\nleakage-aware\ngate", OKABE_ITO["sky"]),
        ("E1", "Strategy\nMIMO\nretained", OKABE_ITO["blue"]),
        ("E2", "Architecture\nLSTM selected\nprotocol deviation", OKABE_ITO["vermillion"]),
        ("E3", "Look-back\n7 days\n(one-SE)", OKABE_ITO["green"]),
        ("E4", "Station\nablation\nvalidation-only", OKABE_ITO["orange"]),
        ("E5", "Frozen evaluation\n2025 final-period\ntest", OKABE_ITO["purple"]),
    ]
    xs = [0.2, 2.2, 4.2, 6.2, 8.2, 10.2]
    audit: list[dict[str, str]] = []
    workflow_values = {
        "E0": (
            "status;policies.interpolate_before_split;policies.scaler_fit_scope",
            {
                "status": bundle.manifests["e0"]["status"],
                "interpolate_before_split": bundle.manifests["e0"]["policies"]["interpolate_before_split"],
                "scaler_fit_scope": bundle.manifests["e0"]["policies"]["scaler_fit_scope"],
            },
        ),
        "E1": (
            "selected_strategy",
            {"selected_strategy": bundle.manifests["e1"]["selected_strategy"]},
        ),
        "E2": (
            "selected_architecture;status;architecture_tuning_trials_completed",
            {
                "selected_architecture": bundle.manifests["e2"]["selected_architecture"],
                "status": bundle.manifests["e2"]["status"],
                "architecture_tuning_trials_completed": bundle.manifests["e2"][
                    "architecture_tuning_trials_completed"
                ],
            },
        ),
        "E3": (
            "selected_lookback_days",
            {"selected_lookback_days": bundle.manifests["e3"]["selected_lookback_days"]},
        ),
        "E4": (
            "evaluation_partition;development_seed",
            {
                "evaluation_partition": bundle.manifests["e4"]["evaluation_partition"],
                "development_seed": bundle.manifests["e4"]["development_seed"],
            },
        ),
        "E5": (
            "evaluation_partition;selected_strategy;selected_architecture;selected_lookback_days",
            {
                "evaluation_partition": bundle.manifests["e5"]["evaluation_partition"],
                "selected_strategy": bundle.manifests["e5"]["selected_strategy"],
                "selected_architecture": bundle.manifests["e5"]["selected_architecture"],
                "selected_lookback_days": bundle.manifests["e5"]["selected_lookback_days"],
            },
        ),
    }
    for index, ((phase, label, color), x) in enumerate(zip(phases, xs, strict=True)):
        box = FancyBboxPatch(
            (x, 2.15),
            1.65,
            1.55,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + 0.825, 3.45, phase, ha="center", va="center", weight="bold", color=color, fontsize=9)
        ax.text(x + 0.825, 2.8, label, ha="center", va="center", fontsize=6.3, linespacing=1.05)
        fields, values = workflow_values[phase]
        serialized_values = json.dumps(values, sort_keys=True, separators=(",", ":"))
        audit.append(
            audit_row(
                bundle,
                figure_id="fig1_frozen_workflow",
                panel="workflow",
                series="phase",
                item=phase,
                source_key=phase.lower(),
                row_filter=f"JSON phase={phase}",
                field=fields,
                transformation="none; conceptual workflow label",
                source_value=serialized_values,
                derived_value=serialized_values,
                displayed_value=label.replace("\n", " | "),
                evidence_class="frozen protocol/evaluation metadata",
            )
        )
        if index < len(phases) - 1:
            dashed = index == 0
            arrow = FancyArrowPatch(
                (x + 1.67, 2.92),
                (xs[index + 1] - 0.03, 2.92),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                linestyle="--" if dashed else "-",
                color=OKABE_ITO["vermillion"] if dashed else OKABE_ITO["black"],
            )
            ax.add_patch(arrow)
            if dashed:
                ax.text(x + 1.83, 3.25, "⚠", ha="center", color=OKABE_ITO["vermillion"], fontsize=10)

    expected = bundle.manifests["e1"]["e0_manifest_sha256"]
    available = bundle.hashes["e0"]
    ax.text(
        1.75,
        1.55,
        "Historical E0 hash linkage unresolved\n(expected ≠ available; dashed link)",
        ha="center",
        va="center",
        color=OKABE_ITO["vermillion"],
        fontsize=7,
        weight="bold",
    )
    ax.text(
        7.0,
        1.35,
        "Verified manifest links: E1 → E2 → E3 → E4 → E5",
        ha="center",
        va="center",
        fontsize=7.5,
        color=OKABE_ITO["green"],
        weight="bold",
    )
    ax.text(
        6.0,
        0.55,
        "Frozen chain: MIMO + LSTM + 7-day look-back",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )
    audit.append(
        audit_row(
            bundle,
            figure_id="fig1_frozen_workflow",
            panel="workflow",
            series="lineage",
            item="E0_to_E1",
            source_key="e1",
            row_filter="JSON phase=E1",
            field="e0_manifest_sha256",
            transformation="compare recorded predecessor hash with SHA-256 of available E0 manifest",
            source_value=str(expected),
            derived_value=f"expected={expected};available={available}",
            displayed_value="historical E0 hash linkage unresolved",
            evidence_class="frozen provenance limitation",
            notes="Rendered as dashed warning link; the chain is not labelled fully verified.",
        )
    )
    for predecessor, successor, field in (
        ("e1", "e2", "e1_manifest_sha256"),
        ("e2", "e3", "e2_manifest_sha256"),
        ("e3", "e4", "e3_manifest_sha256"),
        ("e4", "e5", "e4_manifest_sha256"),
    ):
        value = bundle.manifests[successor][field]
        audit.append(
            audit_row(
                bundle,
                figure_id="fig1_frozen_workflow",
                panel="workflow",
                series="lineage",
                item=f"{predecessor.upper()}_to_{successor.upper()}",
                source_key=successor,
                row_filter=f"JSON phase={successor.upper()}",
                field=field,
                transformation=f"compare with SHA-256 of {predecessor.upper()} manifest",
                source_value=str(value),
                derived_value=str(value == bundle.hashes[predecessor]),
                displayed_value="verified",
                evidence_class="frozen provenance metadata",
            )
        )
    return fig, audit


def plot_mae_effect(bundle: Bundle, plt: Any) -> tuple[Any, list[dict[str, str]]]:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    audit: list[dict[str, str]] = []
    horizons = [1, 3, 5, 7, 14]
    styles = {
        "S0": (OKABE_ITO["blue"], "o", "none"),
        "S4": (OKABE_ITO["orange"], "s", "none"),
    }
    for station_set, (color, marker, linestyle) in styles.items():
        rows = sorted(
            (row for row in bundle.mae_rows if row["station_set"] == station_set),
            key=lambda row: int(row["horizon_days"]),
        )
        means = [as_float(row["MAE_m_mean"], field="MAE_m_mean") for row in rows]
        sds = [as_float(row["MAE_m_std"], field="MAE_m_std") for row in rows]
        ax_a.errorbar(
            horizons,
            means,
            yerr=sds,
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2.5,
            label=station_set,
            markerfacecolor="white" if station_set == "S4" else color,
        )
        for row in rows:
            horizon = int(row["horizon_days"])
            for field, value, label in (
                ("MAE_m_mean", row["MAE_m_mean"], "mean"),
                ("MAE_m_std", row["MAE_m_std"], "SD"),
            ):
                audit.append(
                    audit_row(
                        bundle,
                        figure_id="fig2_horizon_mae_and_upstream_effect",
                        panel="A",
                        series=station_set,
                        item=f"day_{horizon}_{label}",
                        source_key="mae",
                        row_filter=f"station_set={station_set};horizon_days={horizon}",
                        field=field,
                        transformation="none",
                        source_value=value,
                        derived_value=value,
                        displayed_value=f"{float(value):.3f}",
                        evidence_class="descriptive frozen E5 fold/seed summary",
                        notes="SD is across recorded fold/seed evaluation units; it is not a confidence interval.",
                    )
                )
    ax_a.set_xlabel("Forecast horizon (days)")
    ax_a.set_ylabel("Mean MAE (m)")
    ax_a.set_xticks(horizons)
    ax_a.set_title("A  Horizon-specific MAE", loc="left", weight="bold")
    ax_a.legend(frameon=False)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    effects = sorted(
        (
            row
            for row in bundle.effect_rows
            if row["comparison"] == "S4_vs_S0" and int(row["block_length_days"]) == 30
        ),
        key=lambda row: int(row["horizon_days"]),
    )
    y_positions = [1, 0]
    for y, row, color, marker in zip(
        y_positions,
        effects,
        (OKABE_ITO["blue"], OKABE_ITO["orange"]),
        ("o", "s"),
        strict=True,
    ):
        mean = as_float(row["mean_MAE_improvement_m"], field="mean_MAE_improvement_m")
        low = as_float(row["ci_low_m"], field="ci_low_m")
        high = as_float(row["ci_high_m"], field="ci_high_m")
        p_value = as_float(row["dm_holm_p_value"], field="dm_holm_p_value")
        ax_b.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt=marker,
            color=color,
            markerfacecolor="white" if marker == "s" else color,
            capsize=3,
        )
        ax_b.text(0.103, y, f"Holm p={p_value:.3f}", va="center", fontsize=7)
        horizon = int(row["horizon_days"])
        for field, value, label in (
            ("mean_MAE_improvement_m", row["mean_MAE_improvement_m"], "effect"),
            ("ci_low_m", row["ci_low_m"], "ci_low"),
            ("ci_high_m", row["ci_high_m"], "ci_high"),
            ("dm_holm_p_value", row["dm_holm_p_value"], "holm_p"),
        ):
            audit.append(
                audit_row(
                    bundle,
                    figure_id="fig2_horizon_mae_and_upstream_effect",
                    panel="B",
                    series="S4_vs_S0",
                    item=f"day_{horizon}_{label}",
                    source_key="effect",
                    row_filter=f"comparison=S4_vs_S0;horizon_days={horizon};block_length_days=30",
                    field=field,
                    transformation="none; positive effect means S4 has lower MAE",
                    source_value=value,
                    derived_value=value,
                    displayed_value=f"{float(value):.3f}",
                    evidence_class="primary frozen E5 paired inference",
                    notes="No significance marker: 95% CI crosses zero and Holm-adjusted p >= 0.05.",
                )
            )
    ax_b.axvline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_b.set_yticks(y_positions, ["Day 7", "Day 14"])
    ax_b.set_ylim(-0.6, 1.6)
    ax_b.set_xlim(-0.06, 0.16)
    ax_b.set_xlabel("MAE(S0) − MAE(S4) (m)")
    ax_b.set_title("B  Paired upstream effect", loc="left", weight="bold")
    ax_b.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    fig.tight_layout(w_pad=2.0)
    return fig, audit


def plot_skill(bundle: Bundle, plt: Any) -> tuple[Any, list[dict[str, str]]]:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.55, 1]})
    rows = sorted(bundle.skill_rows, key=lambda row: int(row["horizon_days"]))
    horizons = [int(row["horizon_days"]) for row in rows]
    skills = [100 * as_float(row["MAE_skill_vs_persistence"], field="skill") for row in rows]
    lows = [100 * as_float(row["skill_ci_low"], field="skill_ci_low") for row in rows]
    highs = [100 * as_float(row["skill_ci_high"], field="skill_ci_high") for row in rows]
    nse = [as_float(row["NSE"], field="NSE") for row in rows]
    audit: list[dict[str, str]] = []

    ax_a.errorbar(
        horizons,
        skills,
        yerr=[[value - low for value, low in zip(skills, lows, strict=True)], [high - value for value, high in zip(skills, highs, strict=True)]],
        color=OKABE_ITO["blue"],
        marker="o",
        markerfacecolor="white",
        linestyle="-",
        capsize=2,
    )
    day6_index = horizons.index(6)
    day7_index = horizons.index(7)
    ax_a.scatter(6, skills[day6_index], marker="D", s=45, color=OKABE_ITO["green"], zorder=4)
    ax_a.scatter(7, skills[day7_index], marker="s", s=42, facecolor="white", edgecolor=OKABE_ITO["vermillion"], zorder=4)
    ax_a.annotate(
        "Day 6: farthest\npredictively useful",
        xy=(6, skills[day6_index]),
        xytext=(4.5, -42),
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["green"], "linewidth": 0.8},
        fontsize=7,
        color=OKABE_ITO["green"],
    )
    ax_a.annotate(
        "Day 7 CI crosses zero",
        xy=(7, lows[day7_index]),
        xytext=(8.3, -35),
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["vermillion"], "linewidth": 0.8},
        fontsize=7,
        color=OKABE_ITO["vermillion"],
    )
    ax_a.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_a.set_xlabel("Forecast horizon (days)")
    ax_a.set_ylabel("MAE skill vs persistence (%)")
    ax_a.set_xticks(range(1, 15))
    ax_a.set_title("A  Persistence-relative MAE skill", loc="left", weight="bold")
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    ax_b.plot(horizons, nse, color=OKABE_ITO["orange"], marker="s", markerfacecolor="white", linestyle="--")
    ax_b.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_b.scatter(6, nse[day6_index], marker="D", s=45, color=OKABE_ITO["green"], zorder=4)
    ax_b.set_xlabel("Forecast horizon (days)")
    ax_b.set_ylabel("NSE")
    ax_b.set_xticks([1, 3, 5, 7, 10, 14])
    ax_b.set_ylim(0, 1.02)
    ax_b.set_title("B  Nash–Sutcliffe efficiency", loc="left", weight="bold")
    ax_b.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    for row in rows:
        horizon = int(row["horizon_days"])
        for field, multiplier, label, fmt in (
            ("MAE_skill_vs_persistence", 100.0, "skill", ".2f"),
            ("skill_ci_low", 100.0, "ci_low", ".2f"),
            ("skill_ci_high", 100.0, "ci_high", ".2f"),
            ("NSE", 1.0, "NSE", ".3f"),
        ):
            source = row[field]
            derived = as_float(source, field=field) * multiplier
            audit.append(
                audit_row(
                    bundle,
                    figure_id="fig3_predictive_skill_horizon",
                    panel="A" if field != "NSE" else "B",
                    series="S4",
                    item=f"day_{horizon}_{label}",
                    source_key="skill",
                    row_filter=f"station_set=S4;horizon_days={horizon}",
                    field=field,
                    transformation="multiply by 100 for percent axis" if multiplier == 100 else "none",
                    source_value=source,
                    derived_value=repr(derived),
                    displayed_value=format(derived, fmt),
                    evidence_class="frozen E5 persistence-skill inference",
                    notes="Predictively useful is a statistical criterion, not an operational warning lead time.",
                )
            )
    day6_row = rows[day6_index]
    audit.append(
        audit_row(
            bundle,
            figure_id="fig3_predictive_skill_horizon",
            panel="A",
            series="S4",
            item="day_6_predictively_useful_flag",
            source_key="skill",
            row_filter="station_set=S4;horizon_days=6",
            field="predictively_useful",
            transformation="none; determines highlighted diamond marker",
            source_value=day6_row["predictively_useful"],
            derived_value=day6_row["predictively_useful"],
            displayed_value="day 6 highlighted as farthest predictively useful horizon",
            evidence_class="frozen E5 persistence-skill inference",
            notes="Statistical criterion only; not an operational warning lead time.",
        )
    )
    fig.tight_layout(w_pad=2.0)
    return fig, audit


def plot_station(bundle: Bundle, plt: Any) -> tuple[Any, list[dict[str, str]]]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True)
    analyses = (
        ("incremental_addition", "A  Incremental addition", ["VIE", "CKH", "LUA", "CSA"]),
        ("leave_one_out", "B  Leave-one-out", ["CSA", "LUA", "CKH", "VIE"]),
    )
    audit: list[dict[str, str]] = []
    for ax, (analysis, title, order) in zip(axes, analyses, strict=True):
        base_y = {station: index for index, station in enumerate(reversed(order))}
        for horizon, offset, color, marker in (
            (7, -0.12, OKABE_ITO["blue"], "o"),
            (14, 0.12, OKABE_ITO["orange"], "s"),
        ):
            rows = [
                row
                for row in bundle.station_rows
                if row["analysis"] == analysis and int(row["horizon_days"]) == horizon
            ]
            rows.sort(key=lambda row: order.index(row["station"]))
            x = [as_float(row["MAE_improvement_m"], field="MAE_improvement_m") for row in rows]
            y = [base_y[row["station"]] + offset for row in rows]
            ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
                facecolor="white" if horizon == 14 else color,
                label=f"Day {horizon}",
                zorder=3,
            )
            for row in rows:
                value = row["MAE_improvement_m"]
                audit.append(
                    audit_row(
                        bundle,
                        figure_id="figS1_station_contribution",
                        panel="A" if analysis == "incremental_addition" else "B",
                        series=analysis,
                        item=f"{row['station']}_day_{horizon}",
                        source_key="station",
                        row_filter=(
                            f"analysis={analysis};station={row['station']};horizon_days={horizon};"
                            f"reference_set={row['reference_set']};comparison_set={row['comparison_set']}"
                        ),
                        field="MAE_improvement_m",
                        transformation="none; positive means adding/retaining station lowered validation MAE",
                        source_value=value,
                        derived_value=value,
                        displayed_value=f"{float(value):.3f}",
                        evidence_class="exploratory E4 validation-only single-seed",
                        notes="Predictive ablation; not causal station importance.",
                    )
                )
        ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.8)
        ax.set_yticks(range(len(order)), list(reversed(order)))
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("MAE improvement (m)")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    return fig, audit


def event_aggregates(bundle: Bundle) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for metric in ("POD_recall", "FAR", "CSI", "F1"):
        for horizon in (7, 14):
            for station_set in ("S0", "S4"):
                rows = [
                    row
                    for row in bundle.event_rows
                    if row["station_set"] == station_set and int(row["horizon_days"]) == horizon
                ]
                values = [value for row in rows if (value := finite_or_none(row[metric])) is not None]
                aggregates.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "station_set": station_set,
                        "mean": sum(values) / len(values),
                        "finite": len(values),
                        "total": len(rows),
                        "source_values": values,
                    }
                )
    return aggregates


def plot_events(bundle: Bundle, plt: Any) -> tuple[Any, list[dict[str, str]]]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    metrics = (
        ("POD_recall", "POD"),
        ("FAR", "FAR"),
        ("CSI", "CSI"),
        ("F1", "F1"),
    )
    aggregates = event_aggregates(bundle)
    audit: list[dict[str, str]] = []
    x_map = {(7, "S0"): 0, (7, "S4"): 1, (14, "S0"): 3, (14, "S4"): 4}
    for ax, (metric, label), panel in zip(axes.flat, metrics, ("A", "B", "C", "D"), strict=True):
        for horizon, color, marker in ((7, OKABE_ITO["blue"], "o"), (14, OKABE_ITO["orange"], "s")):
            subset = [row for row in aggregates if row["metric"] == metric and row["horizon"] == horizon]
            subset.sort(key=lambda row: row["station_set"])
            xs = [x_map[(horizon, row["station_set"])] for row in subset]
            ys = [row["mean"] for row in subset]
            ax.plot(xs, ys, color=color, linestyle="-" if horizon == 7 else "--", linewidth=0.9)
            ax.scatter(
                xs,
                ys,
                color=color,
                marker=marker,
                facecolor="white" if horizon == 14 else color,
                zorder=3,
                label=f"Day {horizon}",
            )
            for x, row in zip(xs, subset, strict=True):
                ax.text(x, row["mean"] + 0.045, f"{row['finite']}/{row['total']}", ha="center", fontsize=6.5)
                raw_values = ";".join(format(value, ".17g") for value in row["source_values"])
                mean = row["mean"]
                audit.append(
                    audit_row(
                        bundle,
                        figure_id="figS2_event_sensitivity",
                        panel=panel,
                        series=f"{row['station_set']}_day_{horizon}",
                        item=label,
                        source_key="event",
                        row_filter=(
                            f"threshold_name=train_q95;station_set={row['station_set']};"
                            f"horizon_days={horizon};metric={metric}"
                        ),
                        field=metric,
                        transformation="arithmetic mean of finite cells only; undefined cells excluded, never replaced by zero",
                        source_value=raw_values,
                        derived_value=repr(mean),
                        displayed_value=f"mean={mean:.3f};finite/total={row['finite']}/{row['total']}",
                        evidence_class="exploratory E5 event sensitivity",
                        notes="train_q95 only; not flood-warning validation.",
                    )
                )
        ax.set_title(f"{panel}  {label}", loc="left", weight="bold")
        ax.set_xticks([0, 1, 3, 4], ["S0", "S4", "S0", "S4"])
        ax.text(0.5, -0.18, "Day 7", ha="center", transform=ax.get_xaxis_transform(), fontsize=7)
        ax.text(3.5, -0.18, "Day 14", ha="center", transform=ax.get_xaxis_transform(), fontsize=7)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0, 0].set_ylabel("Finite-cell mean")
    axes[1, 0].set_ylabel("Finite-cell mean")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01, "Labels show finite/total cells; undefined cells are omitted, not set to zero.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.6, w_pad=1.4)
    return fig, audit


def write_audit(rows: Iterable[dict[str, str]], output_dir: Path) -> Path:
    path = output_dir / "FIGURE_DATA_AUDIT.csv"
    ordered = sorted(rows, key=lambda row: (row["figure_id"], row["panel"], row["series"], row["item"], row["field"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    return path


def png_dpi(path: Path) -> tuple[float, float] | None:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == b"pHYs" and len(chunk) == 9 and chunk[8] == 1:
            x_ppm, y_ppm = struct.unpack(">II", chunk[:8])
            return x_ppm * 0.0254, y_ppm * 0.0254
        offset += 12 + length
    return None


def validate_outputs(outputs: Iterable[Path], plt: Any) -> None:
    expected = {f"{stem}.{extension}" for stem in FIGURE_NAMES for extension in FORMATS}
    output_list = list(outputs)
    require({path.name for path in output_list} == expected, "Rendered output list is incomplete")
    for path in output_list:
        require(path.stat().st_size > 1000, f"Rendered file is unexpectedly small: {path}")
        if path.suffix == ".svg":
            text = path.read_text(encoding="utf-8")
            require("<svg" in text and "<image" not in text, f"SVG is invalid or contains raster content: {path}")
            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                raise FigureDataError(f"SVG cannot be parsed: {path}") from exc
            require(root.tag.endswith("svg"), f"SVG root element is invalid: {path}")
        elif path.suffix == ".pdf":
            data = path.read_bytes()
            require(
                data.startswith(b"%PDF-") and b"%%EOF" in data and b"/Type /Page" in data,
                f"PDF structure invalid: {path}",
            )
            require(b"/Subtype /Image" not in data, f"PDF contains raster image content: {path}")
        elif path.suffix == ".png":
            image = plt.imread(path)
            require(image.size > 0, f"PNG cannot be decoded: {path}")
            dpi = png_dpi(path)
            require(dpi is not None and all(abs(value - 300) < 1 for value in dpi), f"PNG is not 300 dpi: {path}")


def render(bundle: Bundle, output_dir: Path) -> tuple[list[Path], Path]:
    plt = configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    audits: list[dict[str, str]] = []
    plotters = (
        ("fig1_frozen_workflow", plot_workflow),
        ("fig2_horizon_mae_and_upstream_effect", plot_mae_effect),
        ("fig3_predictive_skill_horizon", plot_skill),
        ("figS1_station_contribution", plot_station),
        ("figS2_event_sensitivity", plot_events),
    )
    for stem, plotter in plotters:
        fig, rows = plotter(bundle, plt)
        outputs.extend(save_figure(fig, output_dir, stem))
        audits.extend(rows)
        plt.close(fig)
    audit_path = write_audit(audits, output_dir)
    require(audit_path.stat().st_size > 1000, "Figure data audit is unexpectedly small")
    validate_outputs(outputs, plt)
    return outputs, audit_path


def print_check(bundle: Bundle) -> None:
    print("Figure source check: PASS")
    print(f"Frozen source root: {FROZEN_ROOT}")
    print("Sign convention: positive MAE(S0)-MAE(S4) means S4 has lower MAE")
    print("Primary effect rows: day 7 and day 14, 30-day blocks; both CIs cross zero")
    print("Predictive-skill rows: S4 days 1-14; day 6 is farthest useful; day 7 CI crosses zero")
    print("E4 station rows: validation-only, seed 42, day 7/day 14")
    print("Event rows: train_q95 exploratory finite-cell means; undefined cells remain undefined")
    print("Source SHA-256:")
    for key in SOURCE_FILES:
        print(f"  {display_path(bundle.paths[key])}  {bundle.hashes[key]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate frozen source schemas, filters, lineage, and sign conventions without rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SVG, PDF, PNG, and FIGURE_DATA_AUDIT.csv outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bundle = load_and_validate()
        if args.check:
            print_check(bundle)
            return 0
        output_dir = args.output_dir.expanduser().resolve()
        outputs, audit_path = render(bundle, output_dir)
        print(f"Figure render: PASS ({len(outputs)} files)")
        for path in sorted(outputs):
            print(f"  {path.name}: {path.stat().st_size} bytes")
        print(f"  {audit_path.name}: {audit_path.stat().st_size} bytes")
        return 0
    except FigureDataError as exc:
        print(f"Figure validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
