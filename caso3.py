from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyomo.environ as pyo

BASE_DIR = Path(__file__).resolve().parent
CASE_DIR = BASE_DIR / "project_b" / "Proyecto_B_Caso3"
OUT_DIR = BASE_DIR / "verificaciones" / "caso3"
OUT_CSV = OUT_DIR / "verificacion_caso3_pyomo_opt.csv"

SERVICE_TIME_MIN = 5.0
DEFAULT_TRUCK_SPEED = 50.0
BIG_M_TIME = 1e4
GAL_PER_LITER = 1 / 3.78541
MAX_TRIPS = 3
K_NEIGHBORS = 99

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class Vehicle:
    id: str
    vtype: str
    capacity: float
    vrange: float
    speed: float


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


def build_vehicles(vehicles_df: pd.DataFrame) -> List[Vehicle]:
    vehs = []
    for _, row in vehicles_df.iterrows():
        speed = float(row["Speed"]) if not pd.isna(row["Speed"]) else DEFAULT_TRUCK_SPEED
        vehs.append(Vehicle(
            id=str(row["StandardizedID"]),
            vtype=str(row["Type"]).strip().lower(),
            capacity=float(row["Capacity"]),
            vrange=float(row["Range"]),
            speed=float(speed),
        ))
    return vehs


def build_distance_time(clients: pd.DataFrame,
                        depot: pd.Series,
                        vehicles: List[Vehicle]) -> Tuple[Dict, Dict, Dict[int, List[int]]]:
    coords = [(float(depot["Latitude"]), float(depot["Longitude"]))] + \
             list(zip(clients["Latitude"].astype(float), clients["Longitude"].astype(float)))
    n_nodes = len(coords)
    dist = {}
    travel_time = {}
    neighbors: Dict[int, List[int]] = {}
    for i in range(n_nodes):
        dlist = []
        for j in range(n_nodes):
            if i == j:
                continue
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dist[(i, j)] = d
            dlist.append((d, j))
            for v in vehicles:
                travel_time[(i, j, v.id)] = (d / max(v.speed, 1e-6)) * 60.0
        dlist.sort(key=lambda x: x[0])
        nb = [j for _, j in dlist[:K_NEIGHBORS]] + [0]
        neighbors[i] = sorted(set(nb))
    return dist, travel_time, neighbors


def build_model(clients_df: pd.DataFrame,
                vehicles: List[Vehicle],
                depot: pd.Series,
                params: Dict[str, float]) -> Tuple[pyo.ConcreteModel, Dict]:
    n_clients = len(clients_df)
    nodes = list(range(n_clients + 1))
    clients_idx = list(range(1, n_clients + 1))
    trips = list(range(MAX_TRIPS))
    veh_ids = [v.id for v in vehicles]
    veh_by_id = {v.id: v for v in vehicles}

    demand = {i: float(row["Demand"]) for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    time_windows = {i: tuple(map(parse_time_to_minutes, str(row["TimeWindow"]).split("-")))
                    for i, (_, row) in enumerate(clients_df.iterrows(), start=1)}
    service_time = {i: SERVICE_TIME_MIN for i in clients_idx}
    start_time = parse_time_to_minutes(str(params.get("start_time", "07:45")))

    dist, travel_time, neighbors = build_distance_time(clients_df, depot, vehicles)

    energy_price = params.get("energy_price_drone", 0.0)
    energy_cons = (params.get("energy_consumption_drone_min", 0.06) +
                   params.get("energy_consumption_drone_max", 0.12)) / 2.0
    fuel_price = params.get("fuel_price_truck", 0.0)
    fuel_eff = (params.get("fuel_efficiency_truck_min", 2.1) +
                params.get("fuel_efficiency_truck_max", 2.5)) / 2.0

    cost_fixed = {v.id: params["C_fixed_drone"] if v.vtype == "drone" else params["C_fixed_truck"] for v in vehicles}
    cost_dist = {v.id: params["C_dist_drone"] if v.vtype == "drone" else params["C_dist_truck"] for v in vehicles}
    cost_time = {v.id: params["C_time_drone"] if v.vtype == "drone" else params["C_time_truck"] for v in vehicles}

    arc_index = []
    for i in nodes:
        for j in neighbors[i]:
            if i == j:
                continue
            for v in veh_ids:
                for t in trips:
                    arc_index.append((i, j, v, t))

    m = pyo.ConcreteModel()
    m.Nodes = pyo.Set(initialize=nodes, ordered=True)
    m.Clients = pyo.Set(initialize=clients_idx, ordered=True)
    m.V = pyo.Set(initialize=veh_ids, ordered=True)
    m.Trips = pyo.Set(initialize=trips, ordered=True)
    m.Arcs = pyo.Set(initialize=arc_index, dimen=4)

    m.x = pyo.Var(m.Arcs, domain=pyo.Binary)
    m.z = pyo.Var([(c, v, t) for c in clients_idx for v in veh_ids for t in trips], domain=pyo.Binary)
    m.y_trip = pyo.Var([(v, t) for v in veh_ids for t in trips], domain=pyo.Binary)
    m.y_use = pyo.Var(veh_ids, domain=pyo.Binary)
    m.t = pyo.Var([(i, v, t) for i in nodes for v in veh_ids for t in trips], domain=pyo.NonNegativeReals)
    m.u = pyo.Var([(c, v, t) for c in clients_idx for v in veh_ids for t in trips], domain=pyo.NonNegativeReals)

    def _attend_rule(model, c):
        return sum(model.z[c, v, t] for v in model.V for t in model.Trips) == 1
    m.AttendOnce = pyo.Constraint(m.Clients, rule=_attend_rule)

    def _flow_in_rule(model, c, v, t):
        return sum(model.x[i, c, v, t] for i, j, vv, tt in model.Arcs if j == c and vv == v and tt == t) == model.z[c, v, t]
    m.FlowIn = pyo.Constraint(m.Clients, m.V, m.Trips, rule=_flow_in_rule)

    def _flow_out_rule(model, c, v, t):
        return sum(model.x[c, j, v, t] for i, j, vv, tt in model.Arcs if i == c and vv == v and tt == t) == model.z[c, v, t]
    m.FlowOut = pyo.Constraint(m.Clients, m.V, m.Trips, rule=_flow_out_rule)

    def _depot_out_rule(model, v, t):
        return sum(model.x[0, j, v, t] for i, j, vv, tt in model.Arcs if i == 0 and vv == v and tt == t) == model.y_trip[v, t]
    m.DepotOut = pyo.Constraint(m.V, m.Trips, rule=_depot_out_rule)

    def _depot_in_rule(model, v, t):
        return sum(model.x[i, 0, v, t] for i, j, vv, tt in model.Arcs if j == 0 and vv == v and tt == t) == model.y_trip[v, t]
    m.DepotIn = pyo.Constraint(m.V, m.Trips, rule=_depot_in_rule)

    def _capacity_rule(model, v, t):
        return sum(demand[c] * model.z[c, v, t] for c in model.Clients) <= veh_by_id[v].capacity * model.y_trip[v, t]
    m.Capacity = pyo.Constraint(m.V, m.Trips, rule=_capacity_rule)

    def _range_rule(model, v, t):
        return sum(dist[i, j] * model.x[i, j, v, t] for i, j, vv, tt in model.Arcs if vv == v and tt == t) <= veh_by_id[v].vrange * model.y_trip[v, t]
    m.Range = pyo.Constraint(m.V, m.Trips, rule=_range_rule)

    def _trip_use(model, v, t):
        return model.y_trip[v, t] <= model.y_use[v]
    m.TripUse = pyo.Constraint(m.V, m.Trips, rule=_trip_use)

    def _tw_lower(model, c, v, t):
        a, _ = time_windows[c]
        return model.t[c, v, t] >= a - BIG_M_TIME * (1 - model.z[c, v, t])
    m.TWLower = pyo.Constraint(m.Clients, m.V, m.Trips, rule=_tw_lower)

    def _tw_upper(model, c, v, t):
        _, b = time_windows[c]
        return model.t[c, v, t] <= b + BIG_M_TIME * (1 - model.z[c, v, t])
    m.TWUpper = pyo.Constraint(m.Clients, m.V, m.Trips, rule=_tw_upper)

    def _depot_time_low(model, v, t):
        return model.t[0, v, t] >= (start_time + 15 * t) * model.y_trip[v, t]
    m.DepotTimeLow = pyo.Constraint(m.V, m.Trips, rule=_depot_time_low)

    def _depot_time_up(model, v, t):
        return model.t[0, v, t] <= start_time + 15 * t + BIG_M_TIME * (1 - model.y_trip[v, t])
    m.DepotTimeUp = pyo.Constraint(m.V, m.Trips, rule=_depot_time_up)

    def _time_flow_rule(model, i, j, v, t):
        if i == j or j == 0:
            return pyo.Constraint.Skip
        serv = service_time.get(i, 0.0) if i != 0 else 0.0
        tij = travel_time[(i, j, v)]
        return model.t[j, v, t] >= model.t[i, v, t] + serv + tij - BIG_M_TIME * (1 - model.x[i, j, v, t])
    m.TimeFlow = pyo.Constraint(m.Arcs, rule=_time_flow_rule)

    def _mtz_rule(model, i, j, v, t):
        if i == j or i == 0 or j == 0:
            return pyo.Constraint.Skip
        if (i, j, v, t) not in model.Arcs:
            return pyo.Constraint.Skip
        return model.u[i, v, t] - model.u[j, v, t] + len(clients_idx) * model.x[i, j, v, t] <= len(clients_idx) - 1
    m.MTZ = pyo.Constraint([(i, j, v, t) for i in clients_idx for j in clients_idx for v in veh_ids for t in trips if (i, j, v, t) in arc_index and i != j],
                           rule=_mtz_rule)

    def _use_vehicle_if_z(model, c, v, t):
        return model.z[c, v, t] <= model.y_trip[v, t]
    m.UseVehicle = pyo.Constraint(m.Clients, m.V, m.Trips, rule=_use_vehicle_if_z)

    def _object_expr(model):
        expr = 0
        for v in model.V:
            expr += cost_fixed[v] * model.y_use[v]
            for t in model.Trips:
                dist_vt = sum(dist[i, j] * model.x[i, j, v, t] for i, j, vv, tt in model.Arcs if vv == v and tt == t)
                time_hours_vt = sum((travel_time[(i, j, v)] / 60.0) * model.x[i, j, v, t] for i, j, vv, tt in model.Arcs if vv == v and tt == t)
                time_hours_vt += sum((service_time[c] / 60.0) * model.z[c, v, t] for c in model.Clients)
                expr += cost_dist[v] * dist_vt + cost_time[v] * time_hours_vt
                if veh_by_id[v].vtype == "drone":
                    expr += dist_vt * energy_cons * energy_price
                else:
                    liters = dist_vt / max(fuel_eff, 1e-6)
                    gallons = liters * GAL_PER_LITER
                    expr += gallons * fuel_price
        return expr
    m.TotalCost = pyo.Objective(rule=_object_expr, sense=pyo.minimize)

    metadata = {
        "clients_idx": clients_idx,
        "veh_ids": veh_ids,
        "trips": trips,
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


def solve_model(model: pyo.ConcreteModel,
                solver: str = "appsi_highs",
                time_limit: int = 120,
                mip_gap: float = 0.02) -> pyo.SolverResults:

    candidates = [solver, "cbc", "glpk"]
    last_exc = None
    for cand in candidates:
        try:
            solver_obj = pyo.SolverFactory(cand)
            if solver_obj is None or not solver_obj.available(False):
                continue
            kwargs = {}
            if cand == "appsi_highs":
                kwargs = {"time_limit": time_limit, "mip_gap": mip_gap}
            return solver_obj.solve(model, tee=False, **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"No se encontró solver disponible. Instala GLPK/CBC/HiGHS. Último error: {last_exc}")


def reconstruct_routes(model: pyo.ConcreteModel,
                       data: Dict,
                       clients_df: pd.DataFrame,
                       depot_df: pd.Series) -> List[Dict]:
    routes = []
    arcs_set = list(model.Arcs)

    def safe_val(var) -> float:
        return float(pyo.value(var, exception=False) or 0.0)

    for v in data["veh_ids"]:
        if safe_val(model.y_use[v]) < 0.5:
            continue
        seq_tokens: List[str] = ["CD01"]
        arrival_log: List[str] = []
        demand_log: List[str] = []
        resup_amounts: List[float] = []
        trips_used = 0
        total_dist = 0.0
        total_time_min = 0.0
        total_load = 0.0
        variable_cost = 0.0
        veh = data["veh_by_id"][v]

        for t in data["trips"]:
            if safe_val(model.y_trip[v, t]) < 0.5:
                continue
            trips_used += 1
            arcs = {(i, j): safe_val(model.x[i, j, v, t])
                    for i, j, vv, tt in arcs_set
                    if vv == v and tt == t and safe_val(model.x[i, j, v, t]) > 0.5}
            route = []
            current = 0
            visited = set()
            while True:
                next_nodes = [j for (i, j), val in arcs.items() if i == current and val > 0.5]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                if nxt == 0 or nxt in visited:
                    break
                route.append(nxt)
                visited.add(nxt)
                current = nxt

            dist_total = sum(data["dist"][(i, j)] * safe_val(model.x[i, j, v, t])
                             for i, j, vv, tt in arcs_set if vv == v and tt == t)
            time_min = sum(data["travel_time"][(i, j, v)] * safe_val(model.x[i, j, v, t])
                           for i, j, vv, tt in arcs_set if vv == v and tt == t)
            time_min += sum(data["service_time"][c] * safe_val(model.z[c, v, t]) for c in data["clients_idx"])
            load_trip = sum(data["demand"][c] * safe_val(model.z[c, v, t]) for c in data["clients_idx"])

            dist_cost = data["cost_dist"][v] * dist_total
            time_cost = data["cost_time"][v] * (time_min / 60.0)
            if veh.vtype == "drone":
                energy_term = dist_total * data["energy_cons"] * data["energy_price"]
                fuel_term = 0.0
            else:
                liters = dist_total / max(data["fuel_eff"], 1e-6)
                gallons = liters * GAL_PER_LITER
                fuel_term = gallons * data["fuel_price"]
                energy_term = 0.0
            variable_cost += dist_cost + time_cost + energy_term + fuel_term

            total_dist += dist_total
            total_time_min += time_min
            total_load += load_trip

            for c in route:
                arrival_log.append(minutes_to_hhmm(safe_val(model.t[c, v, t])))
                demand_log.append(str(int(data["demand"][c])))
                seq_tokens.append(clients_df.iloc[c - 1]["StandardizedID"])
            seq_tokens.append("CD01")
            if trips_used > 1 and load_trip > 0:
                resup_amounts.append(load_trip)

        if trips_used == 0:
            continue
        route_cost = data["cost_fixed"][v] + variable_cost
        routes.append({
            "VehicleId": v,
            "VehicleType": "Drone" if veh.vtype == "drone" else "Truck",
            "InitLoad": total_load if not resup_amounts else total_load - sum(resup_amounts),
            "RouteSequence": "-".join(seq_tokens),
            "Clients": len(demand_log),
            "DemandSatisfied": "-".join(demand_log),
            "ArrivalTimes": "-".join(arrival_log),
            "Resup": max(0, trips_used - 1),
            "ResupAmounts": "-".join(str(int(x)) for x in resup_amounts),
            "Distance": round(total_dist, 3),
            "Time": round(total_time_min, 2),
            "Cost": round(route_cost, 2),
        })
    return routes


def write_verification(routes: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(routes).to_csv(OUT_CSV, index=False)


def run_case3_pyomo_opt():
    logging.info("Cargando datos de Caso 3 Pyomo optimizado...")
    clients_df, vehicles_df, depot_df, params = load_case_data()
    vehicles = build_vehicles(vehicles_df)
    model, meta = build_model(clients_df, vehicles, depot_df, params)
    results = solve_model(model)
    term = getattr(results.solver, "termination_condition", None)
    if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        logging.error("Solver no encontró solución (termination=%s). Revisa solver o parámetros.", term)
        return
    routes = reconstruct_routes(model, meta, clients_df, depot_df)
    write_verification(routes)
    logging.info("Archivo de verificación generado en %s", OUT_CSV)


if __name__ == "__main__":
    run_case3_pyomo_opt()
