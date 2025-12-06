"""
Algoritmo genético para el Proyecto B - Caso 2 (flota híbrida con ventanas de tiempo).
Usa la misma estructura básica del GA del Caso 1 pero adaptado a:
- Drones y 4x4 con costos/velocidades/rango diferenciados.
- Ventanas de tiempo duras (se penaliza con valor alto cualquier violación).
- Datos en project_b/Proyecto_B_Caso2 (clients, vehicles, depots, parameters_rural).

Al ejecutar el script genera:
- CSV/JSON de verificación en verificaciones/GA_Caso2/
- Gráficos: convergencia, mapa de rutas y gantt de cumplimiento de ventanas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caso2 import (
    load_case_data,
    build_vehicles,
    build_distance_time,
    parse_time_to_minutes,
    minutes_to_hhmm,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----------------------------
# Estructuras y utilidades
# ----------------------------


@dataclass
class Vehicle:
    id: str
    vtype: str  # "drone" o "4x4"
    capacity: float
    vrange: float
    speed: float


def build_params(params_df: pd.DataFrame) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for _, row in params_df.iterrows():
        val = row["Value"]
        try:
            params[row["Parameter"]] = float(val)
        except Exception:
            params[row["Parameter"]] = str(val)
    return params


def apply_param_adjustments(
    params: Dict[str, float],
    overrides: Optional[Dict[str, float]] = None,
    multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Devuelve una copia de params aplicando multiplicadores y/o overrides."""
    adj = dict(params)
    if multipliers:
        for k, m in multipliers.items():
            if k in adj and isinstance(adj[k], (int, float)):
                adj[k] = adj[k] * float(m)
    if overrides:
        for k, v in overrides.items():
            adj[k] = v
    return adj


def simulate_route(
    route: List[int],
    veh: Vehicle,
    dist: Dict[Tuple[int, int], float],
    travel_time: Dict[Tuple[int, int, str], float],
    demand: Dict[int, float],
    time_windows: Dict[int, Tuple[int, int]],
    service_time: float,
    start_time: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Evalúa la ruta (0->route->0) respetando TW. Devuelve (factible, métricas)."""
    if not route:
        return True, {
            "distance": 0.0,
            "time_min": 0.0,
            "load": 0.0,
            "arrival_minutes": [],
        }

    total_dist = 0.0
    cur_time = float(start_time)
    arrival_minutes: List[float] = []
    current = 0  # depot idx

    # Capacidad
    load = sum(demand[c] for c in route)
    if load - 1e-6 > veh.capacity:
        return False, {}

    # Recorrido
    for c in route:
        leg_dist = dist[(current, c)]
        leg_time = travel_time[(current, c, veh.id)]
        total_dist += leg_dist
        cur_time += leg_time
        tw_start, tw_end = time_windows[c]
        if cur_time < tw_start:
            cur_time = float(tw_start)  # espera
        if cur_time - 1e-6 > tw_end:
            return False, {}
        arrival_minutes.append(cur_time)
        cur_time += service_time
        current = c

    # Regreso a depósito
    leg_dist = dist[(current, 0)]
    leg_time = travel_time[(current, 0, veh.id)]
    total_dist += leg_dist
    cur_time += leg_time
    total_time_min = cur_time - start_time

    # Rango
    if total_dist - 1e-6 > veh.vrange:
        return False, {}

    metrics = {
        "distance": total_dist,
        "time_min": total_time_min,
        "load": load,
        "arrival_minutes": arrival_minutes,
    }
    return True, metrics


def route_cost(
    metrics: Dict[str, Any],
    veh: Vehicle,
    params: Dict[str, float],
    energy_cons: float,
    fuel_eff: float,
) -> float:
    dist_km = metrics["distance"]
    time_hours = metrics["time_min"] / 60.0
    if veh.vtype == "drone":
        cost = (
            params["C_fixed_drone"]
            + params["C_dist_drone"] * dist_km
            + params["C_time_drone"] * time_hours
            + dist_km * energy_cons * params["energy_price_drone"]
        )
    else:
        gallons = (dist_km / max(fuel_eff, 1e-6)) * (1 / 3.78541)
        cost = (
            params["C_fixed_truck"]
            + params["C_dist_truck"] * dist_km
            + params["C_time_truck"] * time_hours
            + gallons * params["fuel_price_truck"]
        )
    return float(cost)


def decode_and_repair(
    perm: List[int],
    vehicles: List[Vehicle],
    demand: Dict[int, float],
    time_windows: Dict[int, Tuple[int, int]],
    dist: Dict[Tuple[int, int], float],
    travel_time: Dict[Tuple[int, int, str], float],
    service_time: float,
    start_time: int,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """First-fit rotativo + intento de insertar sobrantes."""
    rutas: Dict[str, List[int]] = {v.id: [] for v in vehicles}
    leftovers: List[int] = []
    idx_v = 0
    n_v = len(vehicles)

    # Asignación inicial
    for c in perm:
        assigned = False
        tries = 0
        while tries < n_v and not assigned:
            v = vehicles[idx_v]
            candidate = rutas[v.id] + [c]
            feasible, _ = simulate_route(
                candidate, v, dist, travel_time, demand, time_windows, service_time, start_time
            )
            if feasible:
                rutas[v.id].append(c)
                assigned = True
            idx_v = (idx_v + 1) % n_v
            tries += 1
        if not assigned:
            leftovers.append(c)

    # Reparación simple: probar inserciones en mejor posición
    for c in leftovers[:]:
        best_v = None
        best_pos = None
        for v in vehicles:
            r = rutas[v.id]
            for pos in range(len(r) + 1):
                cand = r[:]
                cand.insert(pos, c)
                feasible, metrics = simulate_route(
                    cand, v, dist, travel_time, demand, time_windows, service_time, start_time
                )
                if feasible:
                    best_v = v.id
                    best_pos = pos
                    break
            if best_v:
                break
        if best_v is not None and best_pos is not None:
            rutas[best_v].insert(best_pos, c)
            leftovers.remove(c)
    return rutas, leftovers


def order_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def ox(pa, pb):
        child = [-1] * n
        child[a : b + 1] = pa[a : b + 1]
        fill = [x for x in pb if x not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child

    return ox(p1, p2), ox(p2, p1)


def mutate_swap(perm: List[int], prob: float) -> List[int]:
    if random.random() > prob:
        return perm
    i, j = random.sample(range(len(perm)), 2)
    perm2 = perm[:]
    perm2[i], perm2[j] = perm2[j], perm2[i]
    return perm2


def mutate_inversion(perm: List[int], prob: float) -> List[int]:
    if random.random() > prob:
        return perm
    i, j = sorted(random.sample(range(len(perm)), 2))
    perm2 = perm[:]
    perm2[i : j + 1] = list(reversed(perm2[i : j + 1]))
    return perm2


def tournament_select(pop: List[List[int]], scores: List[float], k: int = 3) -> List[int]:
    sel = random.sample(range(len(pop)), k)
    best_idx = min(sel, key=lambda i: scores[i])
    return pop[best_idx]


def fitness(
    rutas: Dict[str, List[int]],
    vehicles: List[Vehicle],
    demand: Dict[int, float],
    time_windows: Dict[int, Tuple[int, int]],
    dist: Dict[Tuple[int, int], float],
    travel_time: Dict[Tuple[int, int, str], float],
    params: Dict[str, float],
    energy_cons: float,
    fuel_eff: float,
    service_time: float,
    start_time: int,
    penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    total = 0.0
    detalle: Dict[str, Any] = {}
    infeasible = 0
    for v in vehicles:
        r = rutas[v.id]
        feasible, metrics = simulate_route(
            r, v, dist, travel_time, demand, time_windows, service_time, start_time
        )
        if not feasible:
            total += penalty
            infeasible += 1
            continue
        cost = route_cost(metrics, v, params, energy_cons, fuel_eff)
        detalle[v.id] = {
            "route": r,
            "cost": cost,
            "distance": metrics["distance"],
            "time_min": metrics["time_min"],
            "arrival_minutes": metrics["arrival_minutes"],
            "load": metrics["load"],
            "type": v.vtype,
        }
        total += cost

    return total + infeasible * penalty, detalle


def run_ga_case2(
    pop_size: int = 60,
    generations: int = 120,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    elitism: int = 2,
    random_seed: Optional[int] = None,
    param_overrides: Optional[Dict[str, float]] = None,
    param_multipliers: Optional[Dict[str, float]] = None,
    range_scale_drone: float = 1.0,
    range_scale_truck: float = 1.0,
    speed_scale_drone: float = 1.0,
    speed_scale_truck: float = 1.0,
    service_time_override: Optional[float] = None,
) -> Tuple[Dict[str, List[int]], float, Dict[str, Any], Dict[str, Any]]:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Datos
    clients_df, vehicles_df, depots_df, _ = load_case_data()
    vehicles_df = vehicles_df.copy()
    # Escalar rangos/velocidades por tipo
    for idx, row in vehicles_df.iterrows():
        vtype = str(row["Type"]).strip().lower()
        if vtype == "drone":
            vehicles_df.at[idx, "Range"] = float(row["Range"]) * range_scale_drone
            vehicles_df.at[idx, "Speed"] = float(row["Speed"]) * speed_scale_drone
        else:
            vehicles_df.at[idx, "Range"] = float(row["Range"]) * range_scale_truck
            vehicles_df.at[idx, "Speed"] = float(row["Speed"]) * speed_scale_truck

    vehicles_list = build_vehicles(vehicles_df)
    depot = depots_df.iloc[0]
    params = build_params(pd.read_csv(Path("project_b/Proyecto_B_Caso2/parameters_rural.csv")))
    params = apply_param_adjustments(params, overrides=param_overrides, multipliers=param_multipliers)

    n = len(clients_df)
    demand = {i: float(row["Demand"]) for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    time_windows = {i: tuple(map(parse_time_to_minutes, str(row["TimeWindow"]).split("-")))
                    for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    start_time = parse_time_to_minutes(str(params.get("start_time", "07:45")))
    service_time = float(service_time_override) if service_time_override is not None else 5.0  # minutos

    dist, travel_time = build_distance_time(clients_df, depot, vehicles_list)

    energy_cons = (params.get("energy_consumption_drone_min", 0.06) +
                   params.get("energy_consumption_drone_max", 0.12)) / 2.0
    fuel_eff = (params.get("fuel_efficiency_truck_min", 2.1) +
                params.get("fuel_efficiency_truck_max", 2.5)) / 2.0

    # Población inicial
    population = [random.sample(list(range(1, n + 1)), n) for _ in range(pop_size)]
    penalty = 1e7

    def eval_ind(ind):
        rutas, leftovers = decode_and_repair(
            ind, vehicles_list, demand, time_windows, dist, travel_time, service_time, start_time
        )
        cost, detalle = fitness(
            rutas,
            vehicles_list,
            demand,
            time_windows,
            dist,
            travel_time,
            params,
            energy_cons,
            fuel_eff,
            service_time,
            start_time,
            penalty,
        )
        cost += len(leftovers) * penalty
        return cost, rutas, detalle, leftovers

    scores = []
    cached = []
    for ind in population:
        c, rts, det, lo = eval_ind(ind)
        scores.append(c)
        cached.append((rts, det, lo))

    best_idx = int(np.argmin(scores))
    best_cost = scores[best_idx]
    best_perm = population[best_idx][:]
    best_rutas, best_detalle, best_left = cached[best_idx]
    LOG.info(f"Inicial: mejor coste {best_cost:.2f}")

    history: List[float] = []
    for g in range(generations):
        new_pop: List[List[int]] = []
        new_scores: List[float] = []
        new_cached: List[Any] = []

        ranked = sorted(range(len(population)), key=lambda i: scores[i])
        for i in ranked[:elitism]:
            new_pop.append(population[i][:])
            new_scores.append(scores[i])
            new_cached.append(cached[i])

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            if random.random() < crossover_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = mutate_inversion(mutate_swap(c1, mutation_prob), mutation_prob * 0.5)
            c2 = mutate_inversion(mutate_swap(c2, mutation_prob), mutation_prob * 0.5)

            for child in (c1, c2):
                if len(new_pop) >= pop_size:
                    break
                c, rts, det, lo = eval_ind(child)
                new_pop.append(child)
                new_scores.append(c)
                new_cached.append((rts, det, lo))

        population = new_pop
        scores = new_scores
        cached = new_cached

        gen_best_idx = int(np.argmin(scores))
        gen_best_cost = scores[gen_best_idx]
        history.append(gen_best_cost)
        if gen_best_cost + 1e-6 < best_cost:
            best_cost = gen_best_cost
            best_perm = population[gen_best_idx][:]
            best_rutas, best_detalle, best_left = cached[gen_best_idx]
            LOG.info(f"Gen {g}: nuevo mejor {best_cost:.2f}")

    resumen = {
        "best_cost": best_cost,
        "generations": generations,
        "population": pop_size,
        "history": history,
    }
    final_cost, _, _, _ = eval_ind(best_perm)
    resumen["best_cost_eval"] = final_cost
    return best_rutas, best_cost, resumen, {
        "detalle": best_detalle,
        "leftovers": best_left,
        "vehicles": vehicles_list,
        "demand": demand,
        "time_windows": time_windows,
        "start_time": start_time,
        "service_time": service_time,
        "dist": dist,
        "travel_time": travel_time,
        "params": params,
        "energy_cons": energy_cons,
        "fuel_eff": fuel_eff,
        "clients_df": clients_df,
        "depot": depot,
    }


# ----------------------------
# Reportes y visualizaciones
# ----------------------------

def build_output_rows(best_rutas: Dict[str, List[int]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for v in meta["vehicles"]:
        ruta = best_rutas.get(v.id, [])
        if not ruta:
            continue
        feasible, metrics = simulate_route(
            ruta,
            v,
            meta["dist"],
            meta["travel_time"],
            meta["demand"],
            meta["time_windows"],
            meta["service_time"],
            meta["start_time"],
        )
        if not feasible:
            continue
        cost = route_cost(metrics, v, meta["params"], meta["energy_cons"], meta["fuel_eff"])
        arrival_times = [minutes_to_hhmm(m) for m in metrics["arrival_minutes"]]
        rows.append({
            "VehicleId": v.id,
            "VehicleType": "Drone" if v.vtype == "drone" else "Truck",
            "InitialLoad": round(metrics["load"], 2),
            "RouteSequence": "-".join(
                ["CD01"] + [meta["clients_df"].iloc[c - 1]["StandardizedID"] for c in ruta] + ["CD01"]
            ),
            "ClientsServed": len(ruta),
            "DemandSatisfied": "-".join(str(int(meta["demand"][c])) for c in ruta),
            "ArrivalTimes": "-".join(arrival_times),
            "TotalDistance": round(metrics["distance"], 3),
            "TotalTime": round(metrics["time_min"], 2),
            "Cost": round(cost, 2),
            "RouteNodeIdx": ruta,
            "ArrivalMinutes": metrics["arrival_minutes"],
            "Coords": [(
                float(meta["clients_df"].iloc[c - 1]["Latitude"]),
                float(meta["clients_df"].iloc[c - 1]["Longitude"]),
            ) for c in ruta],
            "FuelCap": round(v.vrange / max(meta["fuel_eff"], 1e-6) * (1 / 3.78541), 3) if v.vtype != "drone" else 0.0,
        })
    return rows


def plot_convergence(history: List[float], out_path: Path) -> None:
    if not history:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Generación")
    plt.ylabel("Mejor coste por generación")
    plt.title("Convergencia GA - Caso 2")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_routes(rows: List[Dict[str, Any]], depot: pd.Series, out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    plt.scatter(depot["Longitude"], depot["Latitude"], c="black", marker="s", label="Depot CD01")
    for r in rows:
        lats = [depot["Latitude"]] + [lat for lat, _ in r["Coords"]] + [depot["Latitude"]]
        lons = [depot["Longitude"]] + [lon for _, lon in r["Coords"]] + [depot["Longitude"]]
        color = "tab:blue" if r["VehicleType"] == "Truck" else "tab:orange"
        plt.plot(lons, lats, "-o", color=color, label=f"{r['VehicleId']} ({r['VehicleType']})")
    plt.title("Rutas GA - Caso 2 (Drone vs Truck)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gantt(rows: List[Dict[str, Any]], clients_df: pd.DataFrame, out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    y_ticks = []
    y_labels = []
    y = 0
    for r in rows:
        for node_idx, arr_min in zip(r["RouteNodeIdx"], r["ArrivalMinutes"]):
            tw = str(clients_df.iloc[node_idx - 1]["TimeWindow"])
            tw_start, tw_end = map(parse_time_to_minutes, tw.split("-"))
            ax.plot([tw_start, tw_end], [y, y], color="gray", linewidth=6, alpha=0.4)
            ax.scatter(arr_min, y, color="tab:blue" if r["VehicleType"] == "Truck" else "tab:orange", zorder=5)
            y_ticks.append(y)
            cid = clients_df.iloc[node_idx - 1]["StandardizedID"]
            y_labels.append(f"{r['VehicleId']}:{cid}")
            y += 1
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Minutos desde 00:00")
    ax.set_title("Cumplimiento de ventanas de tiempo (GA Caso 2)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUT_DIR = Path("verificaciones") / "GA_Caso2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "verificacion_metaheuristica_GA_Caso2.csv"
    out_json = OUT_DIR / "verificacion_metaheuristica_GA_Caso2.json"
    out_conv = OUT_DIR / "metaheuristica_GA_Caso2_convergence.png"
    out_routes = OUT_DIR / "metaheuristica_GA_Caso2_rutas.png"
    out_gantt = OUT_DIR / "metaheuristica_GA_Caso2_gantt.png"

    LOG.info("Ejecutando GA para Proyecto B - Caso 2 ...")
    best_rutas, best_cost, resumen, meta = run_ga_case2(
        pop_size=50, generations=120, crossover_prob=0.9, mutation_prob=0.2, elitism=2, random_seed=1
    )
    LOG.info(f"GA finalizado. Mejor coste: {best_cost:.2f}")

    rows = build_output_rows(best_rutas, meta)
    if not rows:
        LOG.error("No se generaron rutas factibles. Ajusta parámetros o penalizaciones.")
        return

    df_out = pd.DataFrame(rows, columns=[
        "VehicleId",
        "VehicleType",
        "InitialLoad",
        "RouteSequence",
        "ClientsServed",
        "DemandSatisfied",
        "ArrivalTimes",
        "TotalDistance",
        "TotalTime",
        "Cost",
    ])
    df_out.to_csv(out_csv, index=False)
    total_distance = sum(r["TotalDistance"] for r in rows)
    total_cost = sum(r["Cost"] for r in rows)
    resumen_out = {
        "routes": rows,
        "total_cost": round(total_cost, 2),
        "total_distance": round(total_distance, 3),
        "vehicle_count": len(rows),
        "solver_used": "GA",
        "ga_summary": resumen,
    }
    with open(out_json, "w", encoding="utf8") as fh:
        json.dump(resumen_out, fh, indent=2, ensure_ascii=False)

    plot_convergence(resumen.get("history", []), out_conv)
    plot_routes(rows, meta["depot"], out_routes)
    plot_gantt(rows, meta["clients_df"], out_gantt)
    LOG.info(f"CSV: {out_csv}")
    LOG.info(f"JSON: {out_json}")
    LOG.info(f"Gráficos: {out_conv}, {out_routes}, {out_gantt}")


if __name__ == "__main__":
    main()
