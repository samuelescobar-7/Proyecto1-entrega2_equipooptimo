"""
Análisis de sensibilidad comparando Pyomo (exacto) vs GA para Caso 2.

Escenarios predefinidos:
- baseline
- costos +20% (fijos, distancia, tiempo, energía, combustible)
- rango drones -15%
- velocidad drones -15%

Genera un CSV resumen en verificaciones/analisis_sensibilidad_caso2.csv
con costo total, distancia y vehículos usados por escenario y solver.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging

import pandas as pd

from metaheuristica_ga_caso2 import (
    run_ga_case2,
    build_output_rows,
    apply_param_adjustments,
)
from caso2 import (
    load_case_data,
    build_vehicles,
    build_model,
    solve_model,
    reconstruct_routes,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUT_SUMMARY = Path("verificaciones") / "analisis_sensibilidad_caso2.csv"


def adjust_vehicle_df(df: pd.DataFrame, range_scale_drone=1.0, range_scale_truck=1.0,
                      speed_scale_drone=1.0, speed_scale_truck=1.0) -> pd.DataFrame:
    df2 = df.copy()
    for idx, row in df2.iterrows():
        vtype = str(row["Type"]).strip().lower()
        if vtype == "drone":
            df2.at[idx, "Range"] = float(row["Range"]) * range_scale_drone
            df2.at[idx, "Speed"] = float(row["Speed"]) * speed_scale_drone
        else:
            df2.at[idx, "Range"] = float(row["Range"]) * range_scale_truck
            df2.at[idx, "Speed"] = float(row["Speed"]) * speed_scale_truck
    return df2


def run_pyomo_scenario(
    name: str,
    param_multipliers: Dict[str, float],
    range_scale_drone: float,
    range_scale_truck: float,
    speed_scale_drone: float,
    speed_scale_truck: float,
) -> Dict[str, float]:
    clients_df, vehicles_df, depots_df, params = load_case_data()
    params = apply_param_adjustments(params, multipliers=param_multipliers)
    vehicles_df = adjust_vehicle_df(
        vehicles_df, range_scale_drone, range_scale_truck, speed_scale_drone, speed_scale_truck
    )
    vehs = build_vehicles(vehicles_df)
    depot = depots_df.iloc[0]
    model, meta = build_model(clients_df, vehs, depot, params)
    try:
        _ = solve_model(model, solver="glpk")
    except Exception as exc:
        LOG.error("Pyomo fallo en %s: %s", name, exc)
        return {"status": "fail"}
    routes = reconstruct_routes(model, meta, clients_df, depot)
    total_cost = sum(r["Cost"] for r in routes)
    total_dist = sum(r["TotalDistance"] for r in routes)
    return {
        "status": "ok",
        "vehicles_used": len(routes),
        "total_cost": round(total_cost, 2),
        "total_distance": round(total_dist, 3),
    }


def run_ga_scenario(
    param_multipliers: Dict[str, float],
    range_scale_drone: float,
    range_scale_truck: float,
    speed_scale_drone: float,
    speed_scale_truck: float,
) -> Dict[str, float]:
    rutas, best_cost, resumen, meta = run_ga_case2(
        pop_size=40,
        generations=80,
        crossover_prob=0.9,
        mutation_prob=0.2,
        elitism=2,
        random_seed=1,
        param_multipliers=param_multipliers,
        range_scale_drone=range_scale_drone,
        range_scale_truck=range_scale_truck,
        speed_scale_drone=speed_scale_drone,
        speed_scale_truck=speed_scale_truck,
    )
    rows = build_output_rows(rutas, meta)
    total_cost = sum(r["Cost"] for r in rows)
    total_dist = sum(r["TotalDistance"] for r in rows)
    return {
        "status": "ok",
        "vehicles_used": len(rows),
        "total_cost": round(total_cost, 2),
        "total_distance": round(total_dist, 3),
        "best_cost": round(best_cost, 2),
    }


def main():
    scenarios: List[Dict] = [
        {
            "name": "baseline",
            "param_mult": {},
            "range_drone": 1.0,
            "range_truck": 1.0,
            "speed_drone": 1.0,
            "speed_truck": 1.0,
        },
        {
            "name": "costs_up_20",
            "param_mult": {
                "C_fixed_drone": 1.2,
                "C_fixed_truck": 1.2,
                "C_dist_drone": 1.2,
                "C_dist_truck": 1.2,
                "C_time_drone": 1.2,
                "C_time_truck": 1.2,
                "energy_price_drone": 1.2,
                "fuel_price_truck": 1.2,
            },
            "range_drone": 1.0,
            "range_truck": 1.0,
            "speed_drone": 1.0,
            "speed_truck": 1.0,
        },
        {
            "name": "range_drone_down_15",
            "param_mult": {},
            "range_drone": 0.85,
            "range_truck": 1.0,
            "speed_drone": 1.0,
            "speed_truck": 1.0,
        },
        {
            "name": "speed_drone_down_15",
            "param_mult": {},
            "range_drone": 1.0,
            "range_truck": 1.0,
            "speed_drone": 0.85,
            "speed_truck": 1.0,
        },
    ]

    rows_out = []
    for sc in scenarios:
        LOG.info("Escenario %s - Pyomo", sc["name"])
        pyomo_res = run_pyomo_scenario(
            sc["name"], sc["param_mult"], sc["range_drone"], sc["range_truck"],
            sc["speed_drone"], sc["speed_truck"]
        )
        rows_out.append({
            "scenario": sc["name"],
            "solver": "Pyomo",
            **pyomo_res,
        })

        LOG.info("Escenario %s - GA", sc["name"])
        ga_res = run_ga_scenario(
            sc["param_mult"], sc["range_drone"], sc["range_truck"],
            sc["speed_drone"], sc["speed_truck"]
        )
        rows_out.append({
            "scenario": sc["name"],
            "solver": "GA",
            **ga_res,
        })

    df_out = pd.DataFrame(rows_out)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_SUMMARY, index=False)
    LOG.info("Resumen guardado en %s", OUT_SUMMARY)


if __name__ == "__main__":
    main()
