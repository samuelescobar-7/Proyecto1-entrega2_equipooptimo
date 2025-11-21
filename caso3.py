"""
Heurística optimizada para Caso 3 (Proyecto B) con:
- Asignación greedy K-NN (elige siempre el cliente factible más cercano)
- Mejora TSP por viaje con 2-OPT para reducir distancia/tiempo
- Soporta múltiples viajes por vehículo (resupply en CD01)
Salida: verificaciones/caso3/verificacion_caso3_opt.csv
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CASE_DIR = BASE_DIR / "project_b" / "Proyecto_B_Caso3"
OUT_DIR = BASE_DIR / "verificaciones" / "caso3"
OUT_CSV = OUT_DIR / "verificacion_caso3_opt.csv"
OUT_MAP = OUT_DIR / "caso3_opt_rutas.png"
OUT_GANTT = OUT_DIR / "caso3_opt_gantt.png"

SERVICE_TIME_MIN = 5.0
DEFAULT_TRUCK_SPEED = 50.0
MAX_TRIPS = 3
GAL_PER_LITER = 1 / 3.78541

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def minutes_to_hhmm(minutes: float) -> str:
    minutes = max(0.0, minutes)
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    return f"{h:02d}:{m:02d}"


def load_case_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, float]]:
    clients = pd.read_csv(CASE_DIR / "clients.csv")
    vehicles = pd.read_csv(CASE_DIR / "vehicles.csv")
    depots = pd.read_csv(CASE_DIR / "depots.csv")
    params_df = pd.read_csv(CASE_DIR / "parameters_rural.csv")
    params: Dict[str, float] = {}
    for _, row in params_df.iterrows():
        key = str(row["Parameter"]).strip()
        val = row["Value"]
        try:
            params[key] = float(val)
        except Exception:
            params[key] = str(val)
    depot = depots.iloc[0]
    return clients, vehicles, depot, params


def build_distance_time(clients: pd.DataFrame,
                        depot: pd.Series,
                        speeds: Dict[str, float]) -> Tuple[List[List[float]], Dict[Tuple[int, int, str], float], List[str]]:
    coords = [(float(depot["Latitude"]), float(depot["Longitude"]))] + \
             list(zip(clients["Latitude"].astype(float), clients["Longitude"].astype(float)))
    ids = ["CD01"] + clients["StandardizedID"].tolist()
    n = len(coords)
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    travel_time: Dict[Tuple[int, int, str], float] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            D[i][j] = d
            for vid, speed in speeds.items():
                travel_time[(i, j, vid)] = (d / max(speed, 1e-6)) * 60.0
    return D, travel_time, ids


def two_opt(route: List[int], dist: List[List[float]]) -> List[int]:
    """2-OPT para mejorar la secuencia (sin nodo depósito)."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route, dist) + 1e-6 < route_distance(best, dist):
                    best = new_route
                    improved = True
        route = best
    return best


def route_distance(route: List[int], dist: List[List[float]]) -> float:
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for a, b in zip(route[:-1], route[1:]):
        total += dist[a][b]
    total += dist[route[-1]][0]
    return total


def build_time_windows(clients: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    windows: Dict[int, Tuple[float, float]] = {}
    for idx, (_, row) in enumerate(clients.iterrows(), start=1):
        start, end = str(row["TimeWindow"]).split("-")
        windows[idx] = (parse_time_to_minutes(start), parse_time_to_minutes(end))
    return windows


def compute_cost(dist_total: float, time_min: float, vehicle: Dict[str, float], params: Dict[str, float]) -> float:
    time_hours = time_min / 60.0
    if vehicle["Type"].lower() == "drone":
        energy_cons = (params.get("energy_consumption_drone_min", 0.06) +
                       params.get("energy_consumption_drone_max", 0.12)) / 2.0
        energy_price = params.get("energy_price_drone", 0.0)
        variable = params.get("C_dist_drone", 0.0) * dist_total + params.get("C_time_drone", 0.0) * time_hours
        variable += dist_total * energy_cons * energy_price
        fixed = params.get("C_fixed_drone", 0.0)
    else:
        fuel_eff = (params.get("fuel_efficiency_truck_min", 2.1) +
                    params.get("fuel_efficiency_truck_max", 2.5)) / 2.0
        fuel_price = params.get("fuel_price_truck", 0.0)
        variable = params.get("C_dist_truck", 0.0) * dist_total + params.get("C_time_truck", 0.0) * time_hours
        liters = dist_total / max(fuel_eff, 1e-6)
        gallons = liters * GAL_PER_LITER
        variable += gallons * fuel_price
        fixed = params.get("C_fixed_truck", 0.0)
    return round(fixed + variable, 2)


def feasible_next(current: int,
                  remaining: Set[int],
                  cap_rem: float,
                  range_rem: float,
                  time_now: float,
                  tw: Dict[int, Tuple[float, float]],
                  demand: Dict[int, float],
                  dist: List[List[float]],
                  travel_time: Dict[Tuple[int, int, str], float],
                  vid: str) -> List[Tuple[int, float, float, float]]:
    """Retorna candidatos factibles ordenados por distancia (K-NN)."""
    candidates = []
    for c in remaining:
        dem = demand[c]
        if dem > cap_rem + 1e-6:
            continue
        d_to = dist[current][c]
        d_back = dist[c][0]
        if d_to + d_back > range_rem + 1e-6:
            continue
        t_travel = travel_time[(current, c, vid)]
        arrive = time_now + t_travel
        start_tw, end_tw = tw[c]
        wait = max(0.0, start_tw - arrive)
        effective_arrive = arrive + wait
        if effective_arrive > end_tw + 1e-6:
            continue
        candidates.append((c, d_to, wait, effective_arrive))
    candidates.sort(key=lambda x: (x[1], x[3]))  # K-NN: menor distancia, luego llegada
    return candidates


def plan_vehicle(vehicle_row: pd.Series,
                 remaining: Set[int],
                 demand: Dict[int, float],
                 tw: Dict[int, Tuple[float, float]],
                 dist: List[List[float]],
                 travel_time: Dict[Tuple[int, int, str], float],
                 ids: List[str],
                 start_time: float) -> List[Dict]:
    trips_out: List[Dict] = []
    vid = str(vehicle_row["StandardizedID"])
    cap = float(vehicle_row["Capacity"])
    rng = float(vehicle_row["Range"])
    vtype = str(vehicle_row["Type"]).strip().lower()
    trip_idx = 0
    while remaining and trip_idx < MAX_TRIPS:
        route_clients: List[int] = []
        cap_rem = cap
        range_rem = rng
        time_now = start_time + trip_idx * 15  # pequeño desfase entre viajes
        current = 0

        while True:
            cands = feasible_next(current, remaining, cap_rem, range_rem, time_now, tw, demand, dist, travel_time, vid)
            if not cands:
                break
            nxt, d_to, wait, eff_arrive = cands[0]
            service = SERVICE_TIME_MIN
            route_clients.append(nxt)
            remaining.remove(nxt)
            cap_rem -= demand[nxt]
            range_rem -= d_to
            time_now = eff_arrive + service
            current = nxt

        if route_clients:
            # 2-OPT para mejorar distancia sin violar factibilidad horaria (aprox: aceptamos si no rompe TW)
            improved = two_opt(route_clients, dist)
            if improved != route_clients:
                # Verificar TW con secuencia mejorada
                seq_ok = True
                temp_time = start_time + trip_idx * 15
                cap_chk = cap
                rng_chk = rng
                curr = 0
                for c in improved:
                    temp_time += travel_time[(curr, c, vid)]
                    tw_start, tw_end = tw[c]
                    if temp_time > tw_end + 1e-6:
                        seq_ok = False
                        break
                    temp_time = max(temp_time, tw_start) + SERVICE_TIME_MIN
                    rng_chk -= dist[curr][c]
                    cap_chk -= demand[c]
                    curr = c
                if seq_ok and rng_chk >= 0 and cap_chk >= -1e-6:
                    route_clients = improved

            route_seq_tokens = ["CD01"] + [ids[c] for c in route_clients] + ["CD01"]
            arrival_times: List[str] = []
            arrival_minutes: List[float] = []
            temp_time = start_time + trip_idx * 15
            curr = 0
            dist_trip = 0.0
            for c in route_clients:
                dist_trip += dist[curr][c]
                temp_time += travel_time[(curr, c, vid)]
                tw_start, _ = tw[c]
                if temp_time < tw_start:
                    temp_time = tw_start
                arrival_times.append(minutes_to_hhmm(temp_time))
                arrival_minutes.append(temp_time)
                temp_time += SERVICE_TIME_MIN
                curr = c
            dist_trip += dist[curr][0]  # regreso
            temp_time += travel_time[(curr, 0, vid)]
            time_trip_min = temp_time - (start_time + trip_idx * 15)
            load_trip = sum(demand[c] for c in route_clients)

            trips_out.append({
                "VehicleId": vid,
                "VehicleType": "Drone" if vtype == "drone" else "Truck",
                "TripIndex": trip_idx,
                "Clients": len(route_clients),
                "DemandSatisfied": "-".join(str(int(demand[c])) for c in route_clients),
                "RouteSequence": "-".join(route_seq_tokens),
                "ArrivalTimes": "-".join(arrival_times),
                "ArrivalMinutes": arrival_minutes,
                "RouteNodeIdx": route_clients,
                "Distance": round(dist_trip, 3),
                "Time": round(time_trip_min, 2),
                "Load": load_trip,
            })
        trip_idx += 1
    return trips_out


def run_case3_opt():
    logging.info("Cargando datos Caso 3 para heurística optimizada...")
    clients_df, vehicles_df, depot_df, params = load_case_data()
    speeds = {str(r["StandardizedID"]): (float(r["Speed"]) if not pd.isna(r["Speed"]) else DEFAULT_TRUCK_SPEED)
              for _, r in vehicles_df.iterrows()}
    D, travel_time, ids = build_distance_time(clients_df, depot_df, speeds)
    tw = build_time_windows(clients_df)
    demand = {i: float(row["Demand"]) for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    remaining: Set[int] = set(range(1, len(clients_df) + 1))
    start_time = parse_time_to_minutes(str(params.get("start_time", "07:45")))

    all_trips: List[Dict] = []
    for _, veh_row in vehicles_df.iterrows():
        if not remaining:
            break
        trips = plan_vehicle(veh_row, remaining, demand, tw, D, travel_time, ids, start_time)
        all_trips.extend(trips)

    # Agregar métricas por vehículo consolidando viajes
    rows: List[Dict] = []
    by_vehicle: Dict[str, List[Dict]] = {}
    for trip in all_trips:
        by_vehicle.setdefault(trip["VehicleId"], []).append(trip)

    routes_for_plots: List[Dict] = []

    for vid, trips in by_vehicle.items():
        trips = sorted(trips, key=lambda x: x["TripIndex"])
        vehicle_row = vehicles_df[vehicles_df["StandardizedID"] == vid].iloc[0]
        total_distance = sum(t["Distance"] for t in trips)
        total_time = sum(t["Time"] for t in trips)
        total_clients = sum(t["Clients"] for t in trips)
        total_load = sum(t["Load"] for t in trips)
        resup = max(0, len(trips) - 1)
        resup_amounts = "-".join(str(int(t["Load"])) for t in trips[1:]) if resup > 0 else ""
        route_sequence = "-".join(trips[0]["RouteSequence"].split("-")[:-1] + sum([tr["RouteSequence"].split("-")[1:] for tr in trips], []))
        arrival_times = "-".join(sum([tr["ArrivalTimes"].split("-") for tr in trips], []))
        arrival_minutes = sum([tr["ArrivalMinutes"] for tr in trips], [])
        route_node_idx = sum([tr["RouteNodeIdx"] for tr in trips], [])
        demand_satisfied = "-".join(sum([tr["DemandSatisfied"].split("-") for tr in trips if tr["DemandSatisfied"]], []))

        cost = compute_cost(total_distance, total_time, vehicle_row, params)
        rows.append({
            "VehicleId": vid,
            "VehicleType": trips[0]["VehicleType"],
            "InitLoad": trips[0]["Load"],
            "RouteSequence": route_sequence,
            "Clients": total_clients,
            "DemandSatisfied": demand_satisfied,
            "ArrivalTimes": arrival_times,
            "Resup": resup,
            "ResupAmounts": resup_amounts,
            "Distance": round(total_distance, 3),
            "Time": round(total_time, 2),
            "Cost": cost,
        })

        routes_for_plots.append({
            "VehicleId": vid,
            "VehicleType": trips[0]["VehicleType"],
            "RouteNodeIdx": route_node_idx,
            "ArrivalMinutes": arrival_minutes,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    plot_routes(routes_for_plots, clients_df, depot_df)
    plot_gantt(routes_for_plots, clients_df)
    logging.info("Heurística optimizada completada. CSV y figuras en %s", OUT_DIR)


def plot_routes(routes: List[Dict], clients_df: pd.DataFrame, depot: pd.Series) -> None:
    if not routes:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.scatter(depot["Longitude"], depot["Latitude"], c="black", marker="s", label="Depot")
    coords = {0: (float(depot["Latitude"]), float(depot["Longitude"]))}
    for idx, (_, row) in enumerate(clients_df.iterrows(), start=1):
        coords[idx] = (float(row["Latitude"]), float(row["Longitude"]))

    for r in routes:
        if not r["RouteNodeIdx"]:
            continue
        lats = [coords[0][0]] + [coords[i][0] for i in r["RouteNodeIdx"]] + [coords[0][0]]
        lons = [coords[0][1]] + [coords[i][1] for i in r["RouteNodeIdx"]] + [coords[0][1]]
        color = "tab:blue" if r["VehicleType"] == "Truck" else "tab:orange"
        plt.plot(lons, lats, "-o", color=color, label=f"{r['VehicleId']} ({r['VehicleType']})")

    plt.title("Rutas Caso 3 OPT")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_MAP, dpi=200)
    plt.close()


def plot_gantt(routes: List[Dict], clients_df: pd.DataFrame) -> None:
    if not routes:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_ticks = []
    y_labels = []
    y = 0
    for r in routes:
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
    ax.set_title("Cumplimiento de ventanas de tiempo (Caso 3 OPT)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(OUT_GANTT, dpi=200)
    plt.close()


if __name__ == "__main__":
    run_case3_opt()
