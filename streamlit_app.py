from __future__ import annotations

import io
import math
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.app.algorithms.construction import construct
from backend.app.algorithms.distance import calculate_route_distance, generate_distance_matrix
from backend.app.algorithms.local_search import local_search
from backend.app.algorithms.simulated_annealing import simulated_annealing
from backend.app.algorithms.tabu_search import tabu_search
from backend.app.algorithms.validation import validate_cities, validate_route
from backend.app.models.schemas import City
from backend.app.services.experiment import multi_run


APP_TITLE = "TSP Optimizer"
APP_SUBTITLE = "Simulated Annealing & Tabu Search"
NAV = [
    "Dashboard",
    "Data Input",
    "Distance Setting",
    "Initial Solution",
    "Local Search",
    "Simulated Annealing",
    "Tabu Search",
    "Optimization Summary",
    "Results",
    "Visualization",
    "Analysis",
    "Experiment",
    "Export",
]

ROOT = Path(__file__).resolve().parent


def default_cities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"City ID": "C1", "X": 10.0, "Y": 20.0},
            {"City ID": "C2", "X": 20.0, "Y": 40.0},
            {"City ID": "C3", "X": 35.0, "Y": 15.0},
            {"City ID": "C4", "X": 50.0, "Y": 35.0},
            {"City ID": "C5", "X": 65.0, "Y": 20.0},
            {"City ID": "C6", "X": 80.0, "Y": 45.0},
        ]
    )


DEFAULTS = {
    "distance_metric": "euclidean",
    "initial_solution": "nn",
    "starting_mode": "specific",
    "starting_city": "C1",
    "manual_route": ["C1", "C2", "C3", "C4", "C5", "C6"],
    "local_search_swap": False,
    "local_search_2opt": True,
    "local_search_3opt": False,
    "local_search_strategy": "best",
    "local_search_before_sa": True,
    "local_search_after_sa": False,
    "local_search_after_ts": False,
    "sa_enabled": True,
    "ts_enabled": True,
    "sa_initial_temperature": 100.0,
    "sa_alpha": 0.95,
    "sa_min_temperature": 0.01,
    "sa_max_iteration": 300,
    "sa_operator": "2-opt",
    "sa_stopping_condition": "both",
    "sa_reheating": False,
    "sa_reheat_interval": 100,
    "ts_max_iteration": 150,
    "ts_tabu_size": 10,
    "ts_neighborhood_size": 30,
    "ts_operator": "2-opt",
    "ts_aspiration": True,
    "ts_stopping_condition": "iterations",
    "ts_no_improvement_limit": 50,
    "random_seed": 12345,
}


def init_state() -> None:
    if "cities_df" not in st.session_state:
        st.session_state.cities_df = default_cities()
    if "config" not in st.session_state:
        st.session_state.config = dict(DEFAULTS)
    if "result" not in st.session_state:
        st.session_state.result = None
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "message" not in st.session_state:
        st.session_state.message = ""
    if "error" not in st.session_state:
        st.session_state.error = ""
    if "ts_explore" not in st.session_state:
        st.session_state.ts_explore = 0
    if "xlsx_sheets" not in st.session_state:
        st.session_state.xlsx_sheets = []
    if "xlsx_file_name" not in st.session_state:
        st.session_state.xlsx_file_name = ""
    if "experiment_out" not in st.session_state:
        st.session_state.experiment_out = None


def set_message(message: str = "", error: str = "") -> None:
    st.session_state.message = message
    st.session_state.error = error


def config_get(key: str):
    return st.session_state.config[key]


def config_set(key: str, value) -> None:
    st.session_state.config[key] = value
    set_message()


def cities_from_df(df: pd.DataFrame) -> list[City]:
    work = df.copy()
    work.columns = ["City ID", "X", "Y"]
    cities: list[City] = []
    for _, row in work.iterrows():
        city_id = "" if pd.isna(row["City ID"]) else str(row["City ID"]).strip()
        if not city_id:
            continue
        x = row["X"]
        y = row["Y"]
        if pd.isna(x) or pd.isna(y):
            raise ValueError(f"Koordinat X/Y untuk {city_id} harus terisi.")
        cities.append(City(id=city_id, x=float(x), y=float(y)))
    return cities


def cities_to_df(cities: list[City]) -> pd.DataFrame:
    return pd.DataFrame([{"City ID": c.id, "X": c.x, "Y": c.y} for c in cities])


def validate_current_cities() -> tuple[list[City], list[str]]:
    errors: list[str] = []
    try:
        cities = cities_from_df(st.session_state.cities_df)
    except Exception as exc:
        return [], [str(exc)]

    errors.extend(validate_cities(cities))
    ids = [c.id.strip() for c in cities]
    if len(ids) < len(st.session_state.cities_df):
        errors.append("Ada baris data yang belum lengkap. Isi City ID, X, dan Y atau hapus baris kosong.")
    return cities, list(dict.fromkeys(errors))


def validate_manual_route(route: list[str], cities: list[City]) -> list[str]:
    ids = [c.id.strip() for c in cities]
    normalized = [x.strip() for x in route if x.strip()]
    errors: list[str] = []
    if len(normalized) != len(ids):
        errors.append("Manual route harus memuat setiap kota tepat satu kali.")
    if len(set(normalized)) != len(normalized):
        errors.append("Manual route tidak boleh memiliki duplikat.")
    if any(x not in ids for x in normalized):
        errors.append("Manual route berisi City ID yang tidak ada.")
    if not errors and set(normalized) != set(ids):
        errors.append("Ada kota yang hilang dari manual route.")
    return errors


def generate_random_cities(n: int, xmin: float, xmax: float, ymin: float, ymax: float, seed: int) -> pd.DataFrame:
    x = seed & 0xFFFFFFFF

    def next_rand() -> float:
        nonlocal x
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        return x / 4294967296

    data = []
    for i in range(n):
        data.append(
            {
                "City ID": f"C{i + 1}",
                "X": round(xmin + next_rand() * (xmax - xmin), 2),
                "Y": round(ymin + next_rand() * (ymax - ymin), 2),
            }
        )
    return pd.DataFrame(data)


def parse_frame(
    df: pd.DataFrame,
    city_col: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[list[City], dict]:
    cols = list(df.columns)

    def pick(preferred, fallback):
        for p in preferred:
            for c in cols:
                if str(c).strip().lower() == p:
                    return c
        return fallback

    city_col = city_col or pick(["city", "city id", "id"], cols[0])
    x_col = x_col or pick(["x"], cols[1] if len(cols) > 1 else cols[0])
    y_col = y_col or pick(["y"], cols[2] if len(cols) > 2 else cols[-1])

    cities = []
    for _, row in df.iterrows():
        if pd.isna(row[city_col]) or pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        cities.append(City(id=str(row[city_col]), x=float(row[x_col]), y=float(row[y_col])))
    return cities, {
        "columns": cols,
        "city_column": city_col,
        "x_column": x_col,
        "y_column": y_col,
    }


def build_matrix(cities: list[City], metric: str):
    return generate_distance_matrix(cities, metric)


def run_optimization() -> None:
    cities, errors = validate_current_cities()
    cfg = st.session_state.config.copy()

    if cfg["initial_solution"] == "manual" or cfg["starting_mode"] == "manual":
        errors.extend(validate_manual_route(cfg["manual_route"], cities))

    if cfg["starting_mode"] == "specific" and cities and cfg["starting_city"] not in [c.id for c in cities]:
        errors.append("Starting City tidak ditemukan di data kota.")

    if not cfg["sa_enabled"] and not cfg["ts_enabled"]:
        errors.append("Minimal satu algoritma harus ON.")

    if errors:
        set_message(error=" ".join(dict.fromkeys(errors)))
        return

    rng = random.Random(cfg["random_seed"])
    ids, matrix = build_matrix(cities, cfg["distance_metric"])

    start_city = cfg["starting_city"] if cfg["starting_mode"] == "specific" else None
    manual = cfg["manual_route"] if cfg["starting_mode"] == "manual" or cfg["initial_solution"] == "manual" else None
    route = construct(cfg["initial_solution"], ids, matrix, start_city, rng, manual)

    ok, route_errors = validate_route(route, ids)
    if not ok:
        set_message(error="; ".join(route_errors))
        return

    initial_route = route[:]
    initial_distance = calculate_route_distance(route, ids, matrix)

    operators = []
    if cfg["local_search_swap"]:
        operators.append("swap")
    if cfg["local_search_2opt"]:
        operators.append("2-opt")
    if cfg["local_search_3opt"]:
        operators.append("3-opt")

    pre_ls = None
    if cfg["local_search_before_sa"] and operators:
        route, _, pre_ls = local_search(
            route,
            ids,
            matrix,
            operators,
            cfg["local_search_strategy"],
            rng=rng,
        )

    sa = None
    ts = None
    after_sa_ls = None
    after_ts_ls = None

    if cfg["sa_enabled"]:
        sa = simulated_annealing(
            route,
            ids,
            matrix,
            rng=rng,
            initial_temperature=float(cfg["sa_initial_temperature"]),
            alpha=float(cfg["sa_alpha"]),
            min_temperature=float(cfg["sa_min_temperature"]),
            max_iteration=int(cfg["sa_max_iteration"]),
            operator=cfg["sa_operator"],
            stopping_condition=cfg["sa_stopping_condition"],
            reheating=cfg["sa_reheating"],
            reheat_interval=int(cfg["sa_reheat_interval"]),
        )
        route = sa["best_route"]

        if cfg["local_search_after_sa"] and operators:
            route, _, after_sa_ls = local_search(
                route,
                ids,
                matrix,
                operators,
                cfg["local_search_strategy"],
                rng=rng,
            )

    ts_start = route[:]
    if cfg["ts_enabled"]:
        ts = tabu_search(
            ts_start,
            ids,
            matrix,
            rng=rng,
            max_iteration=int(cfg["ts_max_iteration"]),
            tabu_size=int(cfg["ts_tabu_size"]),
            neighborhood_size=int(cfg["ts_neighborhood_size"]),
            operator=cfg["ts_operator"],
            aspiration=cfg["ts_aspiration"],
            stopping_condition=cfg["ts_stopping_condition"],
            no_improvement_limit=int(cfg["ts_no_improvement_limit"]),
        )
        route = ts["best_route"]

        if cfg["local_search_after_ts"] and operators:
            route, _, after_ts_ls = local_search(
                route,
                ids,
                matrix,
                operators,
                cfg["local_search_strategy"],
                rng=rng,
            )

    sa_best = sa["best_distance"] if sa else None
    ts_best = ts["best_distance"] if ts else None

    def improvement(best):
        if best is None or initial_distance == 0:
            return None
        return ((initial_distance - best) / initial_distance) * 100

    if sa_best is not None and ts_best is not None:
        if sa_best < ts_best:
            winner = "Simulated Annealing"
        elif ts_best < sa_best:
            winner = "Tabu Search"
        else:
            winner = "Tie"
    elif sa_best is not None:
        winner = "Simulated Annealing"
    elif ts_best is not None:
        winner = "Tabu Search"
    else:
        winner = None

    st.session_state.result = {
        "city_ids": ids,
        "cities": [c.model_dump() for c in cities],
        "distance_metric": cfg["distance_metric"],
        "distance_matrix": matrix.tolist(),
        "initial": {
            "route": initial_route,
            "distance": initial_distance,
            "method": cfg["initial_solution"],
            "starting_city": start_city,
        },
        "local_search": {
            "operators": operators,
            "strategy": cfg["local_search_strategy"],
            "before_sa": pre_ls,
            "after_sa": after_sa_ls,
            "after_ts": after_ts_ls,
        },
        "sa": None if sa is None else {**sa, "improvement_percent": improvement(sa_best)},
        "ts": None if ts is None else {**ts, "improvement_percent": improvement(ts_best)},
        "comparison": {"best_algorithm": winner},
        "random_seed": cfg["random_seed"],
    }
    st.session_state.ts_explore = 0
    set_message("Optimasi selesai. Semua route, log, dan grafik berasal dari eksekusi ini.")
    st.session_state.page = "Results"


def route_rows(result: dict, route: list[str]) -> list[dict]:
    ids = result["city_ids"]
    matrix = result["distance_matrix"]
    rows = []
    for i, from_id in enumerate(route):
        to_id = route[(i + 1) % len(route)]
        fi = ids.index(from_id)
        ti = ids.index(to_id)
        rows.append({"Step": i + 1, "From": from_id, "To": to_id, "Distance": matrix[fi][ti]})
    return rows


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def route_text(route: list[str] | None) -> str:
    if not route:
        return "—"
    return " → ".join(route) + f" → {route[0]}"


def metric_label(metric: str) -> str:
    return "Euclidean" if metric == "euclidean" else "Manhattan"


def plot_route(cities: list[dict], route: list[str] | None, title: str, show_labels: bool = True, metric: str = "euclidean"):
    by_id = {c["id"]: c for c in cities}
    xs = [c["x"] for c in cities]
    ys = [c["y"] for c in cities]
    fig = go.Figure()

    if route and len(route) > 1:
        closed = route + [route[0]]
        fig.add_trace(
            go.Scatter(
                x=[by_id[c]["x"] for c in closed],
                y=[by_id[c]["y"] for c in closed],
                mode="lines+markers",
                name="Route",
                line={"width": 2.5},
                marker={"size": 7},
                hovertemplate="%{x:.2f}, %{y:.2f}<extra>Route</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text" if show_labels else "markers",
            text=[c["id"] for c in cities] if show_labels else None,
            textposition="top center",
            name="Cities",
            marker={"size": 10, "line": {"width": 1}},
            hovertemplate="<b>%{text}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
        )
    )

    route_distance = None
    if route and len(route) > 1:
        coords = [by_id[c] for c in route]
        if metric == "manhattan":
            route_distance = sum(
                abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
                for a, b in zip(coords, coords[1:] + coords[:1])
            )
        else:
            route_distance = sum(
                math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                for a, b in zip(coords, coords[1:] + coords[:1])
            )

    fig.update_layout(
        title=title,
        height=450,
        margin={"l": 35, "r": 20, "t": 55, "b": 35},
        xaxis={"title": "X", "zeroline": True, "showgrid": True},
        yaxis={"title": "Y", "zeroline": True, "showgrid": True, "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        hovermode="closest",
        dragmode="pan",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )

    cols = st.columns(3)
    cols[0].caption("Zoom: scroll / pinch")
    cols[1].caption("Pan: drag canvas")
    cols[2].caption(f"Distance: {fmt(route_distance)} ({metric_label(metric)})")


def convergence_chart(sa_history: list[dict], ts_history: list[dict]):
    if not sa_history and not ts_history:
        st.info("Convergence history belum tersedia.")
        return

    fig = go.Figure()
    if sa_history:
        fig.add_trace(
            go.Scatter(
                x=[r["iteration"] for r in sa_history],
                y=[r["best_distance"] for r in sa_history],
                mode="lines",
                name="SA",
            )
        )
    if ts_history:
        fig.add_trace(
            go.Scatter(
                x=[r["iteration"] for r in ts_history],
                y=[r["best_distance"] for r in ts_history],
                mode="lines",
                name="TS",
            )
        )
    fig.update_layout(
        title="Convergence • Best Distance per Iteration",
        height=360,
        margin={"l": 30, "r": 20, "t": 55, "b": 35},
        xaxis_title="Iteration",
        yaxis_title="Best Distance",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def comparison_chart(result: dict):
    sa = result.get("sa")
    ts = result.get("ts")
    if not sa and not ts:
        return
    fig = go.Figure()
    metrics = ["Best Distance", "Improvement %", "Runtime (s)"]
    for algo, data in [("SA", sa), ("TS", ts)]:
        if data:
            values = [data["best_distance"], data["improvement_percent"], data["runtime"]]
            fig.add_trace(go.Bar(x=metrics, y=values, name=algo))
    fig.update_layout(
        title="Actual Result Comparison",
        height=350,
        barmode="group",
        margin={"l": 30, "r": 20, "t": 55, "b": 35},
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def show_stats(items: list[tuple[str, object]], columns: int = 4) -> None:
    cols = st.columns(columns)
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, fmt(value))


def render_dashboard():
    result = st.session_state.result
    cities, _ = validate_current_cities()
    summary = {
        "n": len(cities),
        "initial": result["initial"]["distance"] if result else None,
        "sa": result["sa"]["best_distance"] if result and result.get("sa") else None,
        "ts": result["ts"]["best_distance"] if result and result.get("ts") else None,
        "winner": result["comparison"]["best_algorithm"] if result else "—",
    }

    st.markdown("### Full-stack → algorithm-first")
    st.markdown("## Initial solution → SA → TS → comparison → analysis.")
    st.write("Bangun eksperimen TSP yang reproducible dengan hasil algoritma nyata dan histori aktual.")

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("Start with Data Input", type="primary", use_container_width=True):
            st.session_state.page = "Data Input"
            st.rerun()
    with c2:
        st.caption("Local-first • reproducible seed")

    show_stats(
        [
            ("Cities", summary["n"]),
            ("Initial", summary["initial"]),
            ("SA Best", summary["sa"]),
            ("TS Best", summary["ts"]),
        ]
    )
    show_stats(
        [
            ("Distance metric", result["distance_metric"] if result else "Euclidean"),
            ("Seed", result["random_seed"] if result else "—"),
            ("Winner", summary["winner"]),
            ("Status", "Ready" if result else "Awaiting run"),
        ]
    )

    st.divider()
    st.subheader("Core workflow")
    cols = st.columns(4)
    workflow = [
        ("1", "Initial Solution", "NN / insertion / manual"),
        ("2", "Simulated Annealing", "Acceptance + cooling"),
        ("3", "Tabu Search", "Neighborhood + tabu list"),
        ("4", "Comparison & Analysis", "Distance, runtime, convergence"),
    ]
    for col, (number, title, desc) in zip(cols, workflow):
        with col:
            st.markdown(f"**{number}**")
            st.markdown(f"**{title}**")
            st.caption(desc)


def render_data_input():
    st.subheader("Manual Input")
    st.caption("Edit langsung pada tabel. Baris dinamis bisa ditambah atau dihapus.")

    edited = st.data_editor(
        st.session_state.cities_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "City ID": st.column_config.TextColumn("City ID", required=False),
            "X": st.column_config.NumberColumn("X", format="%.2f"),
            "Y": st.column_config.NumberColumn("Y", format="%.2f"),
        },
        key="cities_editor",
    )
    st.session_state.cities_df = edited.copy()
    st.caption(f"{len(edited)} rows")

    controls = st.columns(3)
    with controls[0]:
        if st.button("Add City", use_container_width=True):
            cities, _ = validate_current_cities()
            used = {c.id for c in cities}
            n = 1
            while f"C{n}" in used:
                n += 1
            new_row = pd.DataFrame([{"City ID": f"C{n}", "X": 50.0, "Y": 50.0}])
            st.session_state.cities_df = pd.concat([st.session_state.cities_df, new_row], ignore_index=True)
            st.session_state.config["manual_route"] = [c.id for c in cities] + [f"C{n}"]
            set_message(f"Kota C{n} ditambahkan.")
            st.rerun()
    with controls[1]:
        if st.button("Reset", use_container_width=True):
            st.session_state.cities_df = default_cities()
            st.session_state.config = dict(DEFAULTS)
            st.session_state.result = None
            set_message("Data dikembalikan ke sample 6 kota.")
            st.rerun()
    with controls[2]:
        cities, errors = validate_current_cities()
        st.metric("Cities", len(cities))
        if errors:
            st.warning(" ".join(errors))

    st.divider()
    st.subheader("Generate Random")
    st.caption("Deterministic LCG seed, sama seperti versi sebelumnya.")
    cols = st.columns(6)
    n = cols[0].number_input("Number of Cities", min_value=3, value=6, step=1, key="random_n")
    xmin = cols[1].number_input("X Min", value=0.0, key="random_xmin")
    xmax = cols[2].number_input("X Max", value=100.0, key="random_xmax")
    ymin = cols[3].number_input("Y Min", value=0.0, key="random_ymin")
    ymax = cols[4].number_input("Y Max", value=100.0, key="random_ymax")
    seed = cols[5].number_input("Random Seed", value=12345, step=1, key="random_seed_input")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("GENERATE RANDOM CITIES", type="primary", use_container_width=True):
            if xmin >= xmax or ymin >= ymax:
                set_message(error="X Min harus lebih kecil dari X Max dan Y Min harus lebih kecil dari Y Max.")
            else:
                st.session_state.cities_df = generate_random_cities(int(n), float(xmin), float(xmax), float(ymin), float(ymax), int(seed))
                ids = st.session_state.cities_df["City ID"].astype(str).tolist()
                st.session_state.config["starting_city"] = ids[0]
                st.session_state.config["manual_route"] = ids
                set_message(f"{int(n)} kota berhasil dibuat dari seed {int(seed)}.")
                st.rerun()
    with b2:
        if st.button("Next Seed", use_container_width=True):
            st.session_state.random_seed_input += 1
            st.rerun()

    st.divider()
    st.subheader("Import CSV / Excel")
    uploaded = st.file_uploader("CSV/XLSX", type=["csv", "xlsx", "xls"], key="data_uploader")
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            if st.button("Import CSV", use_container_width=True):
                try:
                    df = pd.read_csv(uploaded)
                    cities, meta = parse_frame(df)
                    st.session_state.cities_df = cities_to_df(cities)
                    ids = [c.id for c in cities]
                    st.session_state.config["starting_city"] = ids[0] if ids else ""
                    st.session_state.config["manual_route"] = ids
                    set_message(
                        f"{len(cities)} kota berhasil diimpor dari CSV. Auto-detect: "
                        f"{meta['city_column']}, {meta['x_column']}, {meta['y_column']}."
                    )
                    st.rerun()
                except Exception as exc:
                    set_message(error=f"CSV tidak dapat dibaca: {exc}")
        else:
            try:
                raw = uploaded.getvalue()
                book = pd.ExcelFile(io.BytesIO(raw))
                st.session_state.xlsx_sheets = list(book.sheet_names)
                st.session_state.xlsx_file_name = uploaded.name
                sheet = st.selectbox("Sheet", st.session_state.xlsx_sheets, key="xlsx_sheet")
                preview = pd.read_excel(io.BytesIO(raw), sheet_name=sheet).head(20)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                if st.button("Import Selected Sheet", use_container_width=True):
                    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
                    cities, meta = parse_frame(df)
                    st.session_state.cities_df = cities_to_df(cities)
                    ids = [c.id for c in cities]
                    st.session_state.config["starting_city"] = ids[0] if ids else ""
                    st.session_state.config["manual_route"] = ids
                    set_message(
                        f"{len(cities)} kota berhasil diimpor dari {sheet}. Auto-detect: "
                        f"{meta['city_column']}, {meta['x_column']}, {meta['y_column']}."
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"Excel tidak dapat dibaca: {exc}")

    st.divider()
    cities, errors = validate_current_cities()
    if cities:
        plot_route(
            [c.model_dump() for c in cities],
            None,
            "Interactive Coordinate Plane",
            show_labels=True,
            metric=config_get("distance_metric"),
        )
    if errors:
        st.warning(" ".join(errors))

    st.info("Data akan divalidasi lagi sebelum algoritma dijalankan.")
    if st.button("Run Optimization", type="primary", use_container_width=True, disabled=len(cities) < 3):
        run_optimization()
        st.rerun()


def render_distance():
    metric = st.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan"],
        index=["euclidean", "manhattan"].index(config_get("distance_metric")),
        format_func=metric_label,
    )
    config_set("distance_metric", metric)
    st.code(
        "Euclidean: d = √((Xi − Xj)² + (Yi − Yj)²)\n"
        "Manhattan: d = |Xi − Xj| + |Yi − Yj|"
    )
    st.caption("Distance matrix dibentuk dari data saat Run Optimization.")

    result = st.session_state.result
    if not result:
        st.info("Run optimization terlebih dahulu untuk membentuk distance matrix aktual.")
        return

    st.subheader(f"Distance Matrix • {metric_label(result['distance_metric'])}")
    matrix = pd.DataFrame(result["distance_matrix"], index=result["city_ids"], columns=result["city_ids"])
    st.dataframe(matrix.style.format("{:.2f}"), use_container_width=True)
    csv = matrix.reset_index(names="City").to_csv(index=False).encode("utf-8-sig")
    st.download_button("Export distance-matrix.csv", csv, "distance-matrix.csv", "text/csv")


def render_initial():
    cities, errors = validate_current_cities()
    ids = [c.id for c in cities]
    cfg = st.session_state.config

    method_names = {
        "nn": "Nearest Neighbor",
        "ni": "Nearest Insertion",
        "ci": "Cheapest Insertion",
        "fi": "Farthest Insertion",
        "ai": "Arbitrary Insertion",
        "manual": "Manual Route",
    }
    method = st.selectbox(
        "Construction Method",
        list(method_names),
        index=list(method_names).index(cfg["initial_solution"]),
        format_func=lambda x: method_names[x],
    )
    config_set("initial_solution", method)

    mode_options = ["random", "specific", "manual"]
    mode = st.selectbox(
        "Starting Node",
        mode_options,
        index=mode_options.index("manual" if method == "manual" else cfg["starting_mode"]),
        format_func=lambda x: {"random": "Random", "specific": "Specific City", "manual": "Manual Route"}[x],
        disabled=method == "manual",
    )
    config_set("starting_mode", mode)

    if mode == "specific" and method != "manual" and ids:
        current = cfg["starting_city"] if cfg["starting_city"] in ids else ids[0]
        config_set("starting_city", st.selectbox("Starting City", ids, index=ids.index(current)))

    if method == "manual" or mode == "manual":
        route_input = st.text_input(
            "Manual Route",
            value=" → ".join(cfg["manual_route"]),
            help="Pisahkan dengan → atau koma. Setiap kota harus muncul tepat sekali.",
        )
        config_set("manual_route", [x.strip() for x in route_input.replace(",", "→").split("→") if x.strip()])

    st.info("Manual Route harus memuat semua kota tepat satu kali. Starting City hanya berlaku untuk construction yang membutuhkan titik awal.")

    result = st.session_state.result
    if result:
        st.subheader("Initial Route")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.code(route_text(result["initial"]["route"]))
            st.metric("Initial Distance", fmt(result["initial"]["distance"]))
        with c2:
            plot_route(
                result["cities"],
                result["initial"]["route"],
                "Initial Solution",
                show_labels=True,
                metric=result["distance_metric"],
            )


def render_local_search():
    cfg = st.session_state.config
    cols = st.columns(3)
    config_set("local_search_swap", cols[0].checkbox("Swap", value=cfg["local_search_swap"]))
    config_set("local_search_2opt", cols[1].checkbox("2-opt", value=cfg["local_search_2opt"]))
    config_set("local_search_3opt", cols[2].checkbox("3-opt", value=cfg["local_search_3opt"]))

    config_set(
        "local_search_strategy",
        st.selectbox(
            "Improvement Strategy",
            ["first", "best"],
            index=["first", "best"].index(cfg["local_search_strategy"]),
            format_func=lambda x: "First Improvement" if x == "first" else "Best Improvement",
        ),
    )

    cols = st.columns(3)
    config_set("local_search_before_sa", cols[0].toggle("Apply Before SA", value=cfg["local_search_before_sa"]))
    config_set("local_search_after_sa", cols[1].toggle("Apply After SA", value=cfg["local_search_after_sa"]))
    config_set("local_search_after_ts", cols[2].toggle("Apply After TS", value=cfg["local_search_after_ts"]))

    st.info(
        "Neighborhood operator (Swap / 2-opt / 3-opt) membentuk kandidat solusi. "
        "Local Search di sini bisa dipakai sebelum atau sesudah metaheuristic."
    )


def render_sa():
    cfg = st.session_state.config
    cols = st.columns(2)
    config_set("sa_enabled", cols[0].toggle("Enable SA", value=cfg["sa_enabled"]))
    config_set(
        "sa_initial_temperature",
        cols[1].number_input("Initial Temperature", min_value=1e-6, value=float(cfg["sa_initial_temperature"]), format="%.6f"),
    )

    cols = st.columns(3)
    config_set("sa_alpha", cols[0].number_input("Alpha / Cooling Rate", min_value=1e-6, max_value=0.999999, value=float(cfg["sa_alpha"]), format="%.6f"))
    config_set("sa_min_temperature", cols[1].number_input("Minimum Temperature", min_value=1e-6, value=float(cfg["sa_min_temperature"]), format="%.6f"))
    config_set("sa_max_iteration", cols[2].number_input("Maximum Iteration", min_value=1, max_value=100000, value=int(cfg["sa_max_iteration"]), step=1))

    cols = st.columns(3)
    config_set(
        "sa_operator",
        cols[0].selectbox("Neighborhood Operator", ["swap", "2-opt", "3-opt"], index=["swap", "2-opt", "3-opt"].index(cfg["sa_operator"])),
    )
    config_set(
        "sa_stopping_condition",
        cols[1].selectbox(
            "Stopping Condition",
            ["both", "temperature", "iterations"],
            index=["both", "temperature", "iterations"].index(cfg["sa_stopping_condition"]),
            format_func=lambda x: {
                "both": "Temperature + Iterations",
                "temperature": "Temperature",
                "iterations": "Iterations",
            }[x],
        ),
    )
    config_set("sa_reheating", cols[2].toggle("Reheating", value=cfg["sa_reheating"]))

    if cfg["sa_reheating"]:
        config_set(
            "sa_reheat_interval",
            st.number_input("Reheat Interval", min_value=1, value=int(cfg["sa_reheat_interval"]), step=1),
        )

    result = st.session_state.result
    if result and result.get("sa"):
        sa = result["sa"]
        show_stats(
            [
                ("Best Distance", sa["best_distance"]),
                ("Improvement", f"{fmt(sa['improvement_percent'])}%"),
                ("Runtime", f"{fmt(sa['runtime'])} s"),
                ("Iterations", sa["iterations"]),
            ]
        )


def render_ts():
    cfg = st.session_state.config
    cols = st.columns(2)
    config_set("ts_enabled", cols[0].toggle("Enable TS", value=cfg["ts_enabled"]))
    config_set("ts_max_iteration", cols[1].number_input("Maximum Iteration", min_value=1, max_value=100000, value=int(cfg["ts_max_iteration"]), step=1))

    cols = st.columns(3)
    config_set("ts_tabu_size", cols[0].number_input("Tabu Size", min_value=1, max_value=100000, value=int(cfg["ts_tabu_size"]), step=1))
    config_set("ts_neighborhood_size", cols[1].number_input("Neighborhood Size", min_value=1, max_value=100000, value=int(cfg["ts_neighborhood_size"]), step=1))
    config_set(
        "ts_operator",
        cols[2].selectbox("Neighborhood Operator", ["swap", "2-opt", "3-opt"], index=["swap", "2-opt", "3-opt"].index(cfg["ts_operator"])),
    )

    cols = st.columns(2)
    config_set("ts_aspiration", cols[0].toggle("Aspiration Criteria", value=cfg["ts_aspiration"]))
    config_set(
        "ts_stopping_condition",
        cols[1].selectbox(
            "Stopping Condition",
            ["iterations", "no_improvement", "both"],
            index=["iterations", "no_improvement", "both"].index(cfg["ts_stopping_condition"]),
            format_func=lambda x: {
                "iterations": "Iterations",
                "no_improvement": "No Improvement",
                "both": "Both",
            }[x],
        ),
    )
    if cfg["ts_stopping_condition"] != "iterations":
        config_set(
            "ts_no_improvement_limit",
            st.number_input("No Improvement Limit", min_value=1, value=int(cfg["ts_no_improvement_limit"]), step=1),
        )

    result = st.session_state.result
    if not result or not result.get("ts"):
        st.info("Run optimization terlebih dahulu untuk melihat histori TS aktual.")
        return

    ts = result["ts"]
    show_stats(
        [
            ("Best Distance", ts["best_distance"]),
            ("Improvement", f"{fmt(ts['improvement_percent'])}%"),
            ("Runtime", f"{fmt(ts['runtime'])} s"),
            ("Iterations", ts["iterations"]),
        ]
    )

    history = ts["history"]
    if history:
        st.subheader("Tabu Search Explorer")
        st.caption("Histori yang sama dengan eksekusi TS, tanpa simulasi ulang.")
        idx = max(0, min(st.session_state.ts_explore, len(history) - 1))
        current = history[idx]

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("Previous", disabled=idx <= 0, use_container_width=True):
                st.session_state.ts_explore = max(0, idx - 1)
                st.rerun()
        with b2:
            st.markdown(f"<div style='text-align:center;padding-top:7px'>Iteration {idx + 1} / {len(history)}</div>", unsafe_allow_html=True)
        with b3:
            if st.button("Next", disabled=idx >= len(history) - 1, use_container_width=True):
                st.session_state.ts_explore = min(len(history) - 1, idx + 1)
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Current:** {route_text(current['current_route'])}")
            st.write(f"Selected Move: `{current['selected_move']}`")
            st.write(f"Current Distance: **{fmt(current['current_distance'])}**")
            st.write(f"Best Distance: **{fmt(current['best_distance'])}**")
            st.write(f"Move Added: `{current['move_added']}`")
            st.write(f"Move Removed: `{current['move_removed'] or '—'}`")
        with c2:
            st.markdown("**Current Tabu List**")
            tabu_df = pd.DataFrame(current.get("tabu_list", []))
            if tabu_df.empty:
                st.caption("Empty")
            else:
                st.dataframe(tabu_df, use_container_width=True, hide_index=True)

        neighborhood = pd.DataFrame(current.get("neighborhood", []))
        if not neighborhood.empty:
            st.markdown("**Candidate Neighborhood for this iteration**")
            st.dataframe(neighborhood, use_container_width=True, hide_index=True)


def render_summary():
    cfg = st.session_state.config
    cities, _ = validate_current_cities()
    result = st.session_state.result

    st.subheader("Configuration Summary")
    items = [
        ("Data", [f"{len(cities)} cities", metric_label(cfg["distance_metric"])]),
        ("Initial Solution", [cfg["initial_solution"], cfg["starting_mode"], route_text(result["initial"]["route"]) if result else "Belum dihitung"]),
        (
            "Local Search",
            [
                " ".join(x for enabled, x in [
                    (cfg["local_search_swap"], "Swap"),
                    (cfg["local_search_2opt"], "2-opt"),
                    (cfg["local_search_3opt"], "3-opt"),
                ] if enabled) or "None",
                cfg["local_search_strategy"],
            ],
        ),
        (
            "Simulated Annealing",
            [f"T0 {cfg['sa_initial_temperature']}", f"α {cfg['sa_alpha']}", f"iter {cfg['sa_max_iteration']}", cfg["sa_operator"], "ON" if cfg["sa_enabled"] else "OFF"],
        ),
        (
            "Tabu Search",
            [f"tabu {cfg['ts_tabu_size']}", f"iter {cfg['ts_max_iteration']}", f"neighborhood {cfg['ts_neighborhood_size']}", cfg["ts_operator"], "ON" if cfg["ts_enabled"] else "OFF"],
        ),
    ]
    cols = st.columns(5)
    for col, (title, lines) in zip(cols, items):
        with col:
            st.markdown(f"**{title}**")
            for line in lines:
                st.caption(line)

    if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
        run_optimization()
        st.rerun()


def render_results():
    result = st.session_state.result
    if not result:
        st.info("Run optimization first.")
        return

    show_stats(
        [
            ("Initial", result["initial"]["distance"]),
            ("SA Best", result["sa"]["best_distance"] if result.get("sa") else None),
            ("TS Best", result["ts"]["best_distance"] if result.get("ts") else None),
            ("Best Algorithm", result["comparison"]["best_algorithm"] or "—"),
        ]
    )

    tabs = st.tabs(["Comparison", "Route Detail", "SA Iteration Log", "TS Iteration Log"])

    with tabs[0]:
        st.subheader("Comparison")
        rows = [
            {"Metric": "Best Distance", "Simulated Annealing": result["sa"]["best_distance"] if result.get("sa") else None, "Tabu Search": result["ts"]["best_distance"] if result.get("ts") else None},
            {"Metric": "Improvement", "Simulated Annealing": result["sa"]["improvement_percent"] if result.get("sa") else None, "Tabu Search": result["ts"]["improvement_percent"] if result.get("ts") else None},
            {"Metric": "Runtime (s)", "Simulated Annealing": result["sa"]["runtime"] if result.get("sa") else None, "Tabu Search": result["ts"]["runtime"] if result.get("ts") else None},
            {"Metric": "Iterations", "Simulated Annealing": result["sa"]["iterations"] if result.get("sa") else None, "Tabu Search": result["ts"]["iterations"] if result.get("ts") else None},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[1]:
        cards = [("Initial Route", result["initial"]["route"], result["initial"]["distance"])]
        if result.get("sa"):
            cards.append(("SA Route", result["sa"]["best_route"], result["sa"]["best_distance"]))
        if result.get("ts"):
            cards.append(("TS Route", result["ts"]["best_route"], result["ts"]["best_distance"]))

        for title, route, distance in cards:
            with st.expander(title, expanded=True):
                st.code(route_text(route))
                st.metric("Total Distance", fmt(distance))
                st.dataframe(pd.DataFrame(route_rows(result, route)), use_container_width=True, hide_index=True)

    with tabs[2]:
        if not result.get("sa"):
            st.info("SA tidak dijalankan.")
        else:
            history = result["sa"]["history"]
            rows = []
            for h in history:
                rows.append(
                    {
                        "Iteration": h["iteration"],
                        "Current Solution": route_text(h["current_route"]),
                        "Current Distance": h["current_distance"],
                        "Best Distance": h["best_distance"],
                        "Temperature": h["temperature"],
                        "Candidate Distance": h["candidate_distance"],
                        "Delta": h["delta"],
                        "Acceptance": "Accepted" if h["acceptance"] else "Rejected",
                        "Move": h["move"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        if not result.get("ts"):
            st.info("TS tidak dijalankan.")
        else:
            rows = []
            for h in result["ts"]["history"]:
                rows.append(
                    {
                        "Iteration": h["iteration"],
                        "Current Solution": route_text(h["current_route"]),
                        "Current Distance": h["current_distance"],
                        "Candidate Solution": route_text(h["candidate_route"]),
                        "Candidate Distance": h["candidate_distance"],
                        "Selected Move": h["selected_move"],
                        "Best Distance": h["best_distance"],
                        "Tabu List": " | ".join(f"{x['move']} [{x['age']}]" for x in h.get("tabu_list", [])),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_visualization():
    result = st.session_state.result
    if not result:
        st.info("Run optimization first.")
        return

    tab = st.radio("Route View", ["INITIAL", "SA", "TS"], horizontal=True)
    show_labels = st.toggle("Show Labels", value=True)

    route = {
        "INITIAL": result["initial"]["route"],
        "SA": result["sa"]["best_route"] if result.get("sa") else None,
        "TS": result["ts"]["best_route"] if result.get("ts") else None,
    }[tab]

    if route is None:
        st.info(f"{tab} tidak dijalankan.")
    else:
        plot_route(result["cities"], route, f"{tab} Route", show_labels=show_labels, metric=result["distance_metric"])

    convergence_chart(
        result["sa"]["history"] if result.get("sa") else [],
        result["ts"]["history"] if result.get("ts") else [],
    )
    comparison_chart(result)


def render_analysis():
    result = st.session_state.result
    if not result:
        st.info("Run optimization first.")
        return

    winner = result["comparison"]["best_algorithm"] or "Tidak ada pemenang tunggal."

    show_stats(
        [
            ("Initial Distance", result["initial"]["distance"]),
            ("SA Final", result["sa"]["best_distance"] if result.get("sa") else None),
            ("TS Final", result["ts"]["best_distance"] if result.get("ts") else None),
            ("Winner", winner),
        ]
    )

    def plateau(history: list[dict]) -> str:
        if not history:
            return "Tidak tersedia."
        if len(history) < 10:
            return "Histori terlalu pendek untuk menilai stabilisasi secara kuat."
        start = history[int(len(history) * 0.75)]["best_distance"]
        end = history[-1]["best_distance"]
        if abs(start - end) < 1e-9:
            return "Best distance relatif stabil pada 25% iterasi terakhir."
        return "Best distance masih berubah pada 25% iterasi terakhir."

    st.subheader("Automatic Result Analysis")
    if result.get("sa"):
        sa = result["sa"]
        st.write(
            f"**SA:** distance {fmt(sa['best_distance'])}, improvement {fmt(sa['improvement_percent'])}%, "
            f"{sa['iterations']} iterasi, runtime {fmt(sa['runtime'])} s. {plateau(sa['history'])}"
        )
    else:
        st.write("SA tidak dijalankan.")

    if result.get("ts"):
        ts = result["ts"]
        st.write(
            f"**TS:** distance {fmt(ts['best_distance'])}, improvement {fmt(ts['improvement_percent'])}%, "
            f"{ts['iterations']} iterasi, runtime {fmt(ts['runtime'])} s. {plateau(ts['history'])}"
        )
    else:
        st.write("TS tidak dijalankan.")

    convergence_chart(
        result["sa"]["history"] if result.get("sa") else [],
        result["ts"]["history"] if result.get("ts") else [],
    )
    comparison_chart(result)


def experiment_base(algo: str, seed: int) -> dict:
    cities, _ = validate_current_cities()
    ids, matrix = build_matrix(cities, config_get("distance_metric"))
    cfg = st.session_state.config
    return {
        "cities": [c.model_dump() for c in cities],
        "ids": ids,
        "matrix": matrix.tolist(),
        "distance_metric": cfg["distance_metric"],
        "seed": seed,
        "initial_solution": cfg["initial_solution"],
        "starting_city": cfg["starting_city"] if cfg["starting_mode"] == "specific" else None,
        "manual_route": cfg["manual_route"],
        "max_iteration": cfg["ts_max_iteration"] if algo == "ts" else cfg["sa_max_iteration"],
        "sa_initial_temperature": cfg["sa_initial_temperature"],
        "sa_alpha": cfg["sa_alpha"],
        "sa_min_temperature": cfg["sa_min_temperature"],
        "operator": cfg["ts_operator"] if algo == "ts" else cfg["sa_operator"],
        "tabu_size": cfg["ts_tabu_size"],
        "neighborhood_size": cfg["ts_neighborhood_size"],
        "aspiration": cfg["ts_aspiration"],
        "ts_stopping_condition": cfg["ts_stopping_condition"],
        "ts_no_improvement_limit": cfg["ts_no_improvement_limit"],
    }


def run_sensitivity_local(base: dict, parameter: str, values: list, algorithm: str, runs: int) -> dict:
    results = []
    for value in values:
        cfg = dict(base)
        if parameter == "alpha":
            cfg["sa_alpha"] = float(value)
        elif parameter == "tabu_size":
            cfg["tabu_size"] = int(value)
        elif parameter == "max_iteration":
            cfg["max_iteration"] = int(value)
        elif parameter == "neighborhood_size":
            cfg["neighborhood_size"] = int(value)
        elif parameter == "operator":
            cfg["operator"] = str(value)
        elif parameter == "initial_solution":
            cfg["initial_solution"] = str(value)
        else:
            raise ValueError("Unsupported sensitivity parameter")

        out = multi_run({**cfg, "algorithm": algorithm, "runs": runs})
        flat = list(out["runs"])
        results.append(
            {
                "parameter": parameter,
                "value": value,
                "best_distance": min(x["best_distance"] for x in flat),
                "average_distance": mean(x["best_distance"] for x in flat),
                "runtime": mean(x["runtime"] for x in flat),
                "runs": len(flat),
            }
        )
    return {"parameter": parameter, "algorithm": algorithm, "results": results}


def render_experiment():
    cities, errors = validate_current_cities()
    if errors:
        st.warning(" ".join(errors))

    params = {
        "alpha": "SA Alpha",
        "tabu_size": "TS Tabu Size",
        "max_iteration": "Maximum Iteration",
        "neighborhood_size": "Neighborhood Size",
        "operator": "Neighborhood Operator",
        "initial_solution": "Initial Solution",
        "none": "Multi-Run Summary",
    }

    cols = st.columns(4)
    parameter = cols[0].selectbox(
        "Experiment Parameter",
        list(params),
        format_func=lambda x: params[x],
        index=list(params).index(st.session_state.get("experiment_parameter", "alpha")),
        key="experiment_parameter",
    )
    algorithm = cols[1].selectbox(
        "Algorithm",
        ["sa", "ts", "sa_vs_ts"],
        format_func=lambda x: {"sa": "SA", "ts": "TS", "sa_vs_ts": "SA vs TS"}[x],
        index=["sa", "ts", "sa_vs_ts"].index(st.session_state.get("experiment_algorithm", "sa_vs_ts")),
        key="experiment_algorithm",
    )
    runs = cols[2].number_input("Number of Runs", min_value=1, value=5, step=1, key="experiment_runs")
    seed = cols[3].number_input("Random Seed", value=12345, step=1, key="experiment_seed")

    values_text = st.text_input(
        "Values (comma separated)",
        value=st.session_state.get("experiment_values", "0.80,0.85,0.90,0.95"),
        disabled=parameter == "none",
        key="experiment_values",
    )

    if algorithm == "sa_vs_ts" and parameter == "alpha":
        st.info("Alpha hanya memengaruhi Simulated Annealing. TS tidak dijalankan untuk parameter ini.")
    if algorithm == "sa_vs_ts" and parameter == "tabu_size":
        st.info("Tabu Size hanya memengaruhi Tabu Search. SA tidak dijalankan untuk parameter ini.")

    if st.button("Run Experiment", type="primary", use_container_width=True):
        try:
            if len(cities) < 3:
                raise ValueError("Minimal 3 kota.")
            if parameter != "none":
                raw_values = [x.strip() for x in values_text.split(",") if x.strip()]
                numeric_params = {"alpha", "tabu_size", "max_iteration", "neighborhood_size"}
                if parameter in numeric_params:
                    parsed = [float(x) if parameter in {"alpha"} else int(float(x)) for x in raw_values]
                else:
                    parsed = raw_values
            else:
                parsed = []

            start = time.perf_counter()
            if parameter == "none":
                base = experiment_base("ts" if algorithm == "ts" else "sa", int(seed))
                out = multi_run({**base, "algorithm": algorithm, "runs": int(runs)})
            elif algorithm == "sa_vs_ts":
                if parameter == "alpha":
                    out = {
                        "combined": True,
                        "sa": run_sensitivity_local(experiment_base("sa", int(seed)), parameter, parsed, "sa", int(runs))["results"],
                        "ts": [],
                        "note": "Alpha hanya memengaruhi Simulated Annealing.",
                    }
                elif parameter == "tabu_size":
                    out = {
                        "combined": True,
                        "sa": [],
                        "ts": run_sensitivity_local(experiment_base("ts", int(seed)), parameter, parsed, "ts", int(runs))["results"],
                        "note": "Tabu Size hanya memengaruhi Tabu Search.",
                    }
                else:
                    sa = run_sensitivity_local(experiment_base("sa", int(seed)), parameter, parsed, "sa", int(runs))
                    ts = run_sensitivity_local(experiment_base("ts", int(seed)), parameter, parsed, "ts", int(runs))
                    out = {"combined": True, "sa": sa["results"], "ts": ts["results"]}
            else:
                out = run_sensitivity_local(experiment_base(algorithm, int(seed)), parameter, parsed, algorithm, int(runs))
            out["_elapsed"] = time.perf_counter() - start
            st.session_state.experiment_out = out
            set_message("Experiment selesai.")
        except Exception as exc:
            st.session_state.experiment_out = {"error": str(exc)}
            set_message(error=str(exc))

    out = st.session_state.experiment_out
    if not out:
        return

    if out.get("error"):
        st.error(out["error"])
        return

    if out.get("note"):
        st.info(out["note"])

    if out.get("combined"):
        c1, c2 = st.columns(2)
        if out.get("sa"):
            with c1:
                st.subheader("SA Sensitivity")
                st.dataframe(pd.DataFrame(out["sa"]), use_container_width=True, hide_index=True)
        if out.get("ts"):
            with c2:
                st.subheader("TS Sensitivity")
                st.dataframe(pd.DataFrame(out["ts"]), use_container_width=True, hide_index=True)

    if out.get("results"):
        st.subheader("Actual Sensitivity Results")
        st.dataframe(pd.DataFrame(out["results"]), use_container_width=True, hide_index=True)

    if out.get("runs"):
        st.subheader("Actual Multi-Run Results")
        rows = pd.DataFrame(out["runs"])
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.json(out["summary"])
        st.caption(f"Elapsed: {fmt(out.get('_elapsed'))} s")


def make_csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def render_export():
    result = st.session_state.result
    if not result:
        st.info("Run optimization first.")
        return

    st.subheader("CSV Export")
    exports = {
        "Input Data": (
            "input-data.csv",
            result["cities"],
        ),
        "Initial Route": (
            "initial-route.csv",
            route_rows(result, result["initial"]["route"]),
        ),
    }
    if result.get("sa"):
        exports["SA Route"] = ("sa-route.csv", route_rows(result, result["sa"]["best_route"]))
        exports["SA Iteration Log"] = (
            "sa-iteration-log.csv",
            [
                {
                    "Iteration": h["iteration"],
                    "CurrentSolution": route_text(h["current_route"]),
                    "CurrentDistance": h["current_distance"],
                    "BestSolution": route_text(h["best_route"]),
                    "BestDistance": h["best_distance"],
                    "Temperature": h["temperature"],
                    "CandidateSolution": route_text(h["candidate_route"]),
                    "CandidateDistance": h["candidate_distance"],
                    "Delta": h["delta"],
                    "Acceptance": h["acceptance"],
                    "AcceptanceProbability": h["acceptance_probability"],
                    "Move": h["move"],
                }
                for h in result["sa"]["history"]
            ],
        )
    if result.get("ts"):
        exports["TS Route"] = ("ts-route.csv", route_rows(result, result["ts"]["best_route"]))
        exports["TS Iteration Log"] = (
            "ts-iteration-log.csv",
            [
                {
                    "Iteration": h["iteration"],
                    "CurrentSolution": route_text(h["current_route"]),
                    "CurrentDistance": h["current_distance"],
                    "CandidateSolution": route_text(h["candidate_route"]),
                    "CandidateDistance": h["candidate_distance"],
                    "SelectedMove": h["selected_move"],
                    "BestSolution": route_text(h["best_route"]),
                    "BestDistance": h["best_distance"],
                    "TabuList": " | ".join(f"{x['move']} [{x['age']}]" for x in h.get("tabu_list", [])),
                    "MoveAdded": h["move_added"],
                    "MoveRemoved": h["move_removed"],
                    "WasTabu": h["was_tabu"],
                    "Aspiration": h["aspiration_used"],
                }
                for h in result["ts"]["history"]
            ],
        )

    cols = st.columns(3)
    for i, (label, (filename, rows)) in enumerate(exports.items()):
        with cols[i % 3]:
            st.download_button(
                label=label,
                data=make_csv_bytes(rows),
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.subheader("PNG")
    st.write("Gunakan chart menu tiga titik untuk menyimpan visualisasi route/chart sebagai gambar.")
    st.caption("Convergence dan comparison chart memakai histori aktual dari result yang sama.")


def render_page(page: str):
    pages = {
        "Dashboard": render_dashboard,
        "Data Input": render_data_input,
        "Distance Setting": render_distance,
        "Initial Solution": render_initial,
        "Local Search": render_local_search,
        "Simulated Annealing": render_sa,
        "Tabu Search": render_ts,
        "Optimization Summary": render_summary,
        "Results": render_results,
        "Visualization": render_visualization,
        "Analysis": render_analysis,
        "Experiment": render_experiment,
        "Export": render_export,
    }
    pages[page]()


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="TSP", layout="wide", initial_sidebar_state="expanded")
    init_state()

    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1450px; }
        section[data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
        .metric-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }
        div[data-testid="stMetric"] { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }
        .small-note { color: #6b7280; font-size: .85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## TSP OPTIMIZER")
        st.caption(APP_SUBTITLE)
        st.divider()
        page = st.radio(
            "Navigation",
            NAV,
            index=NAV.index(st.session_state.page),
            key="nav_page",
        )
        if page != st.session_state.page:
            st.session_state.page = page
            set_message()
        st.divider()
        if st.button("Run Optimization", type="primary", use_container_width=True):
            run_optimization()
            st.rerun()
        st.caption("Local-first • reproducible seed")

    title = st.session_state.page
    st.title(title)
    st.caption("Coordinate-plane TSP optimization workspace")

    if st.session_state.error:
        st.error(st.session_state.error)
    if st.session_state.message:
        st.success(st.session_state.message)

    render_page(st.session_state.page)


if __name__ == "__main__":
    main()
