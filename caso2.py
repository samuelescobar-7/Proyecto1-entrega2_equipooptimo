"""
Modelado Pyomo del Caso 2 (entorno rural con flota híbrida).
- Lee datos de project_b/Proyecto_B_Caso2
- Formula VRP con ventanas de tiempo (duras), capacidades y rangos
- Minimiza costos específicos por tipo de vehículo (drone / 4x4)
- Genera CSV de verificación y visualizaciones básicas
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

BASE_DIR = Path(__file__).resolve().parent
CASE_DIR = BASE_DIR / "project_b" / "Proyecto_B_Caso2"
OUT_DIR = BASE_DIR / "verificaciones" / "caso2"
OUT_CSV = OUT_DIR / "verificacion_caso2.csv"
OUT_MAP = OUT_DIR / "caso2_rutas.png"
OUT_GANTT = OUT_DIR / "caso2_gantt.png"

# Parámetros operativos
SERVICE_TIME_MIN = 5.0           # minutos de servicio por cliente (supuesto)
DEFAULT_TRUCK_SPEED = 50.0       # km/h si falta en el dataset
BIG_M_TIME = 1e4
GAL_PER_LITER = 1 / 3.78541


@dataclass
class Vehicle:
    id: str
    vtype: str  # "drone" o "4x4"
    capacity: float
    vrange: float
    speed: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en kilómetros."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_time_to_minutes(hhmm: str) -> int:
    """Convierte 'HH:MM' a minutos desde medianoche."""
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def minutes_to_hhmm(minutes: float) -> str:
    """Convierte minutos (float) desde medianoche a 'HH:MM'."""
    base = datetime(2000, 1, 1) + timedelta(minutes=float(minutes))
    return base.strftime("%H:%M")


def load_case_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    clients = pd.read_csv(CASE_DIR / "clients.csv")
    vehicles = pd.read_csv(CASE_DIR / "vehicles.csv")
    depots = pd.read_csv(CASE_DIR / "depots.csv")
    params_df = pd.read_csv(CASE_DIR / "parameters_rural.csv")
    params = {}
    for _, row in params_df.iterrows():
        val = row["Value"]
        try:
            params[row["Parameter"]] = float(val)
        except Exception:
            params[row["Parameter"]] = str(val)
    return clients, vehicles, depots, params


def build_vehicles(vehicles_df: pd.DataFrame) -> List[Vehicle]:
    vehs = []
    for _, row in vehicles_df.iterrows():
        speed = float(row["Speed"]) if not pd.isna(row["Speed"]) else DEFAULT_TRUCK_SPEED
        vehs.append(Vehicle(
            id=str(row["StandardizedID"]),
            vtype=str(row["Type"]).strip().lower(),
            capacity=float(row["Capacity"]),
            vrange=float(row["Range"]),
            speed=float(speed)
        ))
    return vehs


def build_distance_time(clients: pd.DataFrame, depot: pd.Series, vehicles: List[Vehicle]) -> Tuple[Dict, Dict]:
    """
    Retorna:
    - dist[(i,j)] en km para nodos (0 = depot, 1..n = clientes)
    - travel_time[(i,j,v)] en minutos
    """
    coords = [(float(depot["Latitude"]), float(depot["Longitude"]))] + \
             list(zip(clients["Latitude"].astype(float), clients["Longitude"].astype(float)))
    n_nodes = len(coords)
    dist = {}
    travel_time = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dist[(i, j)] = d
            for v in vehicles:
                minutes = (d / max(v.speed, 1e-6)) * 60.0
                travel_time[(i, j, v.id)] = minutes
    return dist, travel_time


def build_model(clients_df: pd.DataFrame,
                vehicles: List[Vehicle],
                depot: pd.Series,
                params: Dict[str, float]) -> Tuple[pyo.ConcreteModel, Dict]:
    """
    Construye el modelo Pyomo del VRP con ventanas de tiempo (Caso 2).
    """
    n_clients = len(clients_df)
    nodes = list(range(n_clients + 1))  # 0 = depot
    clients_idx = list(range(1, n_clients + 1))
    veh_ids = [v.id for v in vehicles]
    veh_by_id = {v.id: v for v in vehicles}

    # Datos auxiliares
    demand = {i: float(row["Demand"]) for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    time_windows = {i: tuple(map(parse_time_to_minutes, str(row["TimeWindow"]).split("-")))
                    for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    service_time = {i: SERVICE_TIME_MIN for i in clients_idx}
    start_time = parse_time_to_minutes(str(params.get("start_time", "07:45")))

    dist, travel_time = build_distance_time(clients_df, depot, vehicles)

    # Cost parameters
    energy_price = params.get("energy_price_drone", 0.0)
    energy_cons = (params.get("energy_consumption_drone_min", 0.06) +
                   params.get("energy_consumption_drone_max", 0.12)) / 2.0
    fuel_price = params.get("fuel_price_truck", 0.0)
    fuel_eff = (params.get("fuel_efficiency_truck_min", 2.1) +
                params.get("fuel_efficiency_truck_max", 2.5)) / 2.0

    cost_fixed = {v.id: params["C_fixed_drone"] if v.vtype == "drone" else params["C_fixed_truck"] for v in vehicles}
    cost_dist = {v.id: params["C_dist_drone"] if v.vtype == "drone" else params["C_dist_truck"] for v in vehicles}
    cost_time = {v.id: params["C_time_drone"] if v.vtype == "drone" else params["C_time_truck"] for v in vehicles}

    m = pyo.ConcreteModel()
    m.Nodes = pyo.Set(initialize=nodes, ordered=True)
    m.Clients = pyo.Set(initialize=clients_idx, ordered=True)
    m.V = pyo.Set(initialize=veh_ids, ordered=True)

    m.x = pyo.Var([(i, j, v) for i in nodes for j in nodes if i != j for v in veh_ids], domain=pyo.Binary)
    m.z = pyo.Var([(c, v) for c in clients_idx for v in veh_ids], domain=pyo.Binary)
    m.y = pyo.Var(veh_ids, domain=pyo.Binary)
    m.t = pyo.Var([(i, v) for i in nodes for v in veh_ids], domain=pyo.NonNegativeReals)
    m.u = pyo.Var([(c, v) for c in clients_idx for v in veh_ids], domain=pyo.NonNegativeReals)

    # Cada cliente atendido exactamente una vez
    def _attend_rule(model, c):
        return sum(model.z[c, v] for v in model.V) == 1
    m.AttendOnce = pyo.Constraint(m.Clients, rule=_attend_rule)

    # Flujo y relación con z
    def _flow_in_rule(model, c, v):
        return sum(model.x[i, c, v] for i in nodes if i != c) == model.z[c, v]
    m.FlowIn = pyo.Constraint(m.Clients, m.V, rule=_flow_in_rule)

    def _flow_out_rule(model, c, v):
        return sum(model.x[c, j, v] for j in nodes if j != c) == model.z[c, v]
    m.FlowOut = pyo.Constraint(m.Clients, m.V, rule=_flow_out_rule)

    # Salida y llegada al depósito si y[v]=1
    def _depot_out_rule(model, v):
        return sum(model.x[0, j, v] for j in model.Nodes if j != 0) == model.y[v]
    m.DepotOut = pyo.Constraint(m.V, rule=_depot_out_rule)

    def _depot_in_rule(model, v):
        return sum(model.x[i, 0, v] for i in model.Nodes if i != 0) == model.y[v]
    m.DepotIn = pyo.Constraint(m.V, rule=_depot_in_rule)

    # Capacidad
    def _capacity_rule(model, v):
        return sum(demand[c] * model.z[c, v] for c in model.Clients) <= veh_by_id[v].capacity * model.y[v]
    m.Capacity = pyo.Constraint(m.V, rule=_capacity_rule)

    # Rango por vehículo
    def _range_rule(model, v):
        return sum(dist[i, j] * model.x[i, j, v] for i, j, _ in model.x if _ == v) <= veh_by_id[v].vrange * model.y[v]
    m.Range = pyo.Constraint(m.V, rule=_range_rule)

    # Relación z <= y
    def _use_vehicle_if_z(model, c, v):
        return model.z[c, v] <= model.y[v]
    m.UseVehicle = pyo.Constraint(m.Clients, m.V, rule=_use_vehicle_if_z)

    # Ventanas de tiempo
    def _tw_lower(model, c, v):
        a, _ = time_windows[c]
        return model.t[c, v] >= a - BIG_M_TIME * (1 - model.z[c, v])
    m.TWLower = pyo.Constraint(m.Clients, m.V, rule=_tw_lower)

    def _tw_upper(model, c, v):
        _, b = time_windows[c]
        return model.t[c, v] <= b + BIG_M_TIME * (1 - model.z[c, v])
    m.TWUpper = pyo.Constraint(m.Clients, m.V, rule=_tw_upper)

    # Fijar salida desde el depósito en start_time si el vehículo se usa
    def _depot_time_low(model, v):
        return model.t[0, v] >= start_time * model.y[v]
    m.DepotTimeLow = pyo.Constraint(m.V, rule=_depot_time_low)

    def _depot_time_up(model, v):
        return model.t[0, v] <= start_time + BIG_M_TIME * (1 - model.y[v])
    m.DepotTimeUp = pyo.Constraint(m.V, rule=_depot_time_up)

    # Precedencia temporal
    def _time_flow_rule(model, i, j, v):
        if i == j or j == 0:
            return pyo.Constraint.Skip
        serv = service_time.get(i, 0.0) if i != 0 else 0.0
        tij = travel_time[(i, j, v)]
        return model.t[j, v] >= model.t[i, v] + serv + tij - BIG_M_TIME * (1 - model.x[i, j, v])
    m.TimeFlow = pyo.Constraint([(i, j, v) for i in nodes for j in nodes if i != j for v in veh_ids],
                                rule=_time_flow_rule)

    # Eliminación de subciclos (MTZ)
    def _mtz_rule(model, i, j, v):
        if i == j or i == 0 or j == 0:
            return pyo.Constraint.Skip
        return model.u[i, v] - model.u[j, v] + len(clients_idx) * model.x[i, j, v] <= len(clients_idx) - 1
    m.MTZ = pyo.Constraint([(i, j, v) for i in clients_idx for j in clients_idx for v in veh_ids if i != j],
                           rule=_mtz_rule)

    # Objetivo: costos fijos + km + tiempo + energía/combustible
    def _object_expr(model):
        expr = 0
        for v in model.V:
            dist_v = sum(dist[i, j] * model.x[i, j, v] for i, j, _ in model.x if _ == v)
            time_hours_v = sum((travel_time[(i, j, v)] / 60.0) * model.x[i, j, v] for i, j, _ in model.x if _ == v)
            time_hours_v += sum((service_time[c] / 60.0) * model.z[c, v] for c in model.Clients)
            expr += cost_fixed[v] * model.y[v] + cost_dist[v] * dist_v + cost_time[v] * time_hours_v
            if veh_by_id[v].vtype == "drone":
                expr += dist_v * energy_cons * energy_price
            else:
                liters = dist_v / max(fuel_eff, 1e-6)
                gallons = liters * GAL_PER_LITER
                expr += gallons * fuel_price
        return expr
    m.TotalCost = pyo.Objective(rule=_object_expr, sense=pyo.minimize)

    metadata = {
        "clients_idx": clients_idx,
        "veh_ids": veh_ids,
        "demand": demand,
        "time_windows": time_windows,
        "service_time": service_time,
        "start_time": start_time,
        "dist": dist,
        "travel_time": travel_time,
        "veh_by_id": veh_by_id,
        "energy_cons": energy_cons,
        "fuel_eff": fuel_eff,
        "energy_price": energy_price,
        "fuel_price": fuel_price,
        "cost_fixed": cost_fixed,
        "cost_dist": cost_dist,
        "cost_time": cost_time,
    }
    return m, metadata


def solve_model(model: pyo.ConcreteModel, solver: str = "glpk") -> pyo.SolverResults:
    candidates = [solver, "appsi_highs", "cbc"]
    last_exc = None
    for cand in candidates:
        try:
            solver_obj = pyo.SolverFactory(cand)
            if solver_obj is None or not solver_obj.available(False):
                continue
            return solver_obj.solve(model, tee=False)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"No se encontró solver disponible. Intenta instalar GLPK/CBC/HiGHS. Último error: {last_exc}")


def reconstruct_routes(model: pyo.ConcreteModel,
                       data: Dict,
                       clients_df: pd.DataFrame,
                       depot_df: pd.Series) -> List[Dict]:
    routes = []
    nodes = [0] + data["clients_idx"]
    coords = {0: (float(depot_df["Latitude"]), float(depot_df["Longitude"]))}
    for idx, (_, row) in enumerate(clients_df.iterrows(), start=1):
        coords[idx] = (float(row["Latitude"]), float(row["Longitude"]))

    for v in data["veh_ids"]:
        if pyo.value(model.y[v]) < 0.5:
            continue
        arcs = {(i, j): pyo.value(model.x[i, j, v]) for i, j, _ in model.x if _ == v and pyo.value(model.x[i, j, v]) > 0.5}
        route = []
        current = 0
        visited = set()
        while True:
            next_nodes = [j for (i, j), val in arcs.items() if abs(val) > 0.5 and i == current]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            if nxt == 0:
                break
            if nxt in visited:
                break
            route.append(nxt)
            visited.add(nxt)
            current = nxt

        # Métricas
        dist_total = sum(data["dist"][i, j] * pyo.value(model.x[i, j, v]) for i, j, _ in model.x if _ == v)
        time_min = sum(data["travel_time"][(i, j, v)] * pyo.value(model.x[i, j, v]) for i, j, _ in model.x if _ == v)
        time_min += sum(data["service_time"][c] * pyo.value(model.z[c, v]) for c in data["clients_idx"])
        load = sum(data["demand"][c] * pyo.value(model.z[c, v]) for c in data["clients_idx"])

        # Costo por vehículo siguiendo la estructura del objetivo
        dist_cost = data["cost_dist"][v] * dist_total
        time_cost = data["cost_time"][v] * (time_min / 60.0)
        fixed_cost = data["cost_fixed"][v] * pyo.value(model.y[v])
        if data["veh_by_id"][v].vtype == "drone":
            energy_term = dist_total * data["energy_cons"] * data["energy_price"]
            fuel_term = 0.0
            fuel_cap_gal = 0.0
        else:
            liters = dist_total / max(data["fuel_eff"], 1e-6)
            gallons = liters * GAL_PER_LITER
            fuel_term = gallons * data["fuel_price"]
            energy_term = 0.0
            fuel_cap_gal = data["veh_by_id"][v].vrange / max(data["fuel_eff"], 1e-6) * GAL_PER_LITER
        vehicle_cost = float(fixed_cost + dist_cost + time_cost + energy_term + fuel_term)

        # Arrival times de las visitas en orden
        arrival_times = []
        for c in route:
            arrival_times.append(minutes_to_hhmm(pyo.value(model.t[c, v])))
        demand_str = "-".join(str(int(data["demand"][c])) for c in route)
        arrival_str = "-".join(arrival_times)

        routes.append({
            "VehicleId": v,
            "VehicleType": "Drone" if data["veh_by_id"][v].vtype == "drone" else "Truck",
            "LoadCap": data["veh_by_id"][v].capacity,
            "RouteNodeIdx": route,
            "RouteSequence": "-".join(["CD01"] + [clients_df.iloc[c - 1]["StandardizedID"] for c in route] + ["CD01"]),
            "ClientsServed": len(route),
            "DemandSatisfied": demand_str,
            "ArrivalTimes": arrival_str,
            "TotalDistance": round(dist_total, 3),
            "TotalTime": round(time_min, 2),
            "InitialLoad": round(load, 2),
            "Cost": round(vehicle_cost, 2),
            "FuelCost": round(fuel_term, 2),
            "EnergyCost": round(energy_term, 2),
            "FuelCap": round(fuel_cap_gal, 3),
            "RefuelStops": 0,
            "RefuelAmounts": "",
            "ArrivalMinutes": [pyo.value(model.t[c, v]) for c in route],
            "Coords": [coords[c] for c in route],
        })
    return routes


def write_verification(routes: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in routes:
        rows.append({
            "VehicleId": r["VehicleId"],
            "VehicleType": r["VehicleType"],
            "LoadCap": r.get("LoadCap", ""),
            "FuelCap": r.get("FuelCap", ""),
            "InitialLoad": r["InitialLoad"],
            "InitialFuel": r.get("FuelCap", ""),
            "RouteSequence": r["RouteSequence"],
            "Municipalities": r["ClientsServed"],
            "DemandSatisfied": r["DemandSatisfied"],
            "RefuelStops": r.get("RefuelStops", 0),
            "RefuelAmounts": r.get("RefuelAmounts", ""),
            "ArrivalTimes": r["ArrivalTimes"],
            "TotalDistance": r["TotalDistance"],
            "TotalTime": r["TotalTime"],
            "FuelCost": r.get("FuelCost", 0),
            "EnergyCost": r.get("EnergyCost", 0),
            "TotalCost": r["Cost"],
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def plot_routes(routes: List[Dict], depot: pd.Series) -> None:
    if not routes:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.scatter(depot["Longitude"], depot["Latitude"], c="black", marker="s", label="Depot")

    for r in routes:
        lats = [depot["Latitude"]] + [lat for lat, _ in r["Coords"]] + [depot["Latitude"]]
        lons = [depot["Longitude"]] + [lon for _, lon in r["Coords"]] + [depot["Longitude"]]
        color = "tab:blue" if r["VehicleType"] == "Truck" else "tab:orange"
        plt.plot(lons, lats, "-o", color=color, label=r["VehicleType"] if r["VehicleType"] else r["VehicleId"])

    plt.title("Rutas Caso 2 (Drone vs Truck)")
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
    ax.set_title("Cumplimiento de ventanas de tiempo (Caso 2)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(OUT_GANTT, dpi=200)
    plt.close()


def main():
    clients_df, vehicles_df, depots_df, params = load_case_data()
    vehs = build_vehicles(vehicles_df)
    depot = depots_df.iloc[0]
    model, meta = build_model(clients_df, vehs, depot, params)
    results = solve_model(model, solver="glpk")
    print(results)

    routes = reconstruct_routes(model, meta, clients_df, depot)
    write_verification(routes)
    plot_routes(routes, depot)
    plot_gantt(routes, clients_df)

    if not routes:
        print("No se generaron rutas (revisar factibilidad o solver).")
    else:
        print("\nRutas encontradas (Caso 2):")
        for r in routes:
            print(f"- {r.get('VehicleId')} ({r.get('VehicleType')}): {r.get('RouteSequence')}")
            print(f"  ArrivalTimes: {r.get('ArrivalTimes')} | Distance km: {r.get('TotalDistance')} | Time min: {r.get('TotalTime')}")
            print(f"  Cost: {r.get('Cost')} (Fuel {r.get('FuelCost', 0)}, Energy {r.get('EnergyCost', 0)}) | LoadCap: {r.get('LoadCap')} | FuelCap: {r.get('FuelCap')}")
        print(f"\nCSV de verificacion: {OUT_CSV}")
        print(f"Mapa: {OUT_MAP}")
        print(f"Gantt: {OUT_GANTT}")
    print(f"Archivos generados en {OUT_DIR}")


if __name__ == "__main__":
    main()
