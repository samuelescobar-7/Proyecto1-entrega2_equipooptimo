import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CASE_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "verificaciones"
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class Client:
    id: int
    code: str
    demand: float
    ready: float
    due: float
    service: float
    lat: float
    lon: float


@dataclass
class TruckParams:
    id: str
    capacity: float
    vrange: float
    speed: float
    fixed_cost: float
    cost_km: float
    time_cost_h: float
    late_penalty: float
    start_time: float


def parse_time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def minutes_to_hhmm(minutes: float) -> str:
    minutes = max(0.0, minutes)
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    return f"{h:02d}:{m:02d}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_case3_data() -> Tuple[Dict[int, Client], TruckParams,
        np.ndarray, np.ndarray, float, float]:
    clients_df = pd.read_csv(CASE_DIR /"project_b"/"Proyecto_B_Caso3"/ "clients.csv")
    vehicles_df = pd.read_csv(CASE_DIR /"project_b"/"Proyecto_B_Caso3"/ "vehicles.csv")
    depots_df = pd.read_csv(CASE_DIR /"project_b"/"Proyecto_B_Caso3"/ "depots.csv")
    params_df = pd.read_csv(CASE_DIR /"project_b"/"Proyecto_B_Caso3"/ "parameters_rural.csv")

    params: Dict[str, float] = {}
    for _, row in params_df.iterrows():
        key = str(row["Parameter"]).strip()
        val = row["Value"]
        try:
            params[key] = float(val)
        except Exception:
            params[key] = str(val)

    depot = depots_df.iloc[0]
    depot_lat = float(depot["Latitude"])
    depot_lon = float(depot["Longitude"])

    clients: Dict[int, Client] = {}
    service_time_min = float(params.get("service_time", 5.0))
    for idx, row in clients_df.iterrows():
        cid = idx + 1
        code = str(row.get("StandardizedID", row.get("ClientID", f"C{cid:03d}")))
        demand = float(row["Demand"])
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        tw = str(row["TimeWindow"])
        tw_start, tw_end = tw.split("-")
        ready = parse_time_to_minutes(tw_start)
        due = parse_time_to_minutes(tw_end)

        clients[cid] = Client(
            id=cid,
            code=code,
            demand=demand,
            ready=ready,
            due=due,
            service=service_time_min,
            lat=lat,
            lon=lon,
        )

    truck_row = vehicles_df.iloc[0]
    truck_id = str(truck_row.get("StandardizedID", truck_row.get("VehicleID", "V001")))
    capacity = float(truck_row["Capacity"])
    vrange = float(truck_row["Range"])
    speed_val = truck_row["Speed"]
    speed = float(speed_val) if not pd.isna(speed_val) else 50.0

    cost_km = float(params.get("C_dist_truck", 3000.0))
    time_cost_h = float(params.get("C_time_truck", 8000.0))
    fixed_cost = float(params.get("C_fixed_truck", 60000.0))
    late_penalty = float(params.get("late_penalty_per_min", 0.0))
    start_time = parse_time_to_minutes(str(params.get("start_time", "07:45")))

    truck = TruckParams(
        id=truck_id,
        capacity=capacity,
        vrange=vrange,
        speed=speed,
        fixed_cost=fixed_cost,
        cost_km=cost_km,
        time_cost_h=time_cost_h,
        late_penalty=late_penalty,
        start_time=start_time,
    )

    coords = [(depot_lat, depot_lon)] + [(c.lat, c.lon) for c in clients.values()]
    n_nodes = len(coords)
    dist_km = np.zeros((n_nodes, n_nodes))
    travel_min = np.zeros((n_nodes, n_nodes))

    for i, (lat_i, lon_i) in enumerate(coords):
        for j, (lat_j, lon_j) in enumerate(coords):
            if i == j:
                continue
            d = haversine_km(lat_i, lon_i, lat_j, lon_j)
            dist_km[i, j] = d
            travel_min[i, j] = d / max(truck.speed, 1e-6) * 60.0

    return clients, truck, dist_km, travel_min, depot_lat, depot_lon


def evaluate_route(
    perm: List[int],
    dist_km: np.ndarray,
    travel_min: np.ndarray,
    clients: Dict[int, Client],
    truck: TruckParams,
) -> Tuple[float, List[float]]:
    if not perm:
        return float("inf"), []

    depot = 0
    current = depot
    current_time = truck.start_time
    total_dist = 0.0
    total_late_cost = 0.0
    start_service: List[float] = []
    remaining_cap = truck.capacity

    for cid in perm:
        c = clients[cid]

        if c.demand > remaining_cap:
            total_dist += dist_km[current, depot]
            current_time += travel_min[current, depot]
            current = depot
            remaining_cap = truck.capacity

        d = dist_km[current, cid]
        t = travel_min[current, cid]
        total_dist += d
        current_time += t

        if current_time < c.ready:
            current_time = c.ready

        late = max(0.0, current_time - c.due)
        total_late_cost += late * truck.late_penalty

        start_service.append(current_time)
        current_time += c.service
        current = cid
        remaining_cap -= c.demand

    total_dist += dist_km[current, depot]
    current_time += travel_min[current, depot]

    cost_dist = total_dist * truck.cost_km
    total_time_hrs = (current_time - truck.start_time) / 60.0
    total_time_cost = total_time_hrs * truck.time_cost_h

    total_cost = truck.fixed_cost + cost_dist + total_time_cost + total_late_cost
    return total_cost, start_service


def init_population(n_clients: int, pop_size: int) -> List[List[int]]:
    base = list(range(1, n_clients + 1))
    pop: List[List[int]] = []
    for _ in range(pop_size):
        p = base[:]
        random.shuffle(p)
        pop.append(p)
    return pop


def ordered_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b+1] = p1[a:b+1]
    fill = [g for g in p2 if g not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def swap_mutation(p: List[int], prob: float) -> None:
    if random.random() < prob:
        i, j = random.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]


def tournament_selection(pop: List[List[int]], fitness: List[float], k: int = 3) -> List[int]:
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitness[i])
    return pop[best][:]


def run_ga(
    clients: Dict[int, Client],
    truck: TruckParams,
    dist_km: np.ndarray,
    travel_min: np.ndarray,
    pop_size: int = 80,
    generations: int = 200,
    cx_prob: float = 0.9,
    mut_prob: float = 0.3,
) -> Tuple[List[int], float, List[float], List[float]]:
    n = len(clients)
    population = init_population(n, pop_size)

    def eval_ind(ind: List[int]) -> float:
        cost, _ = evaluate_route(ind, dist_km, travel_min, clients, truck)
        return cost

    best_cost_history: List[float] = []
    best_ind: List[int] | None = None
    best_cost = float("inf")

    for gen in range(generations):
        fitness = [eval_ind(ind) for ind in population]

        for ind, f in zip(population, fitness):
            if f < best_cost:
                best_cost = f
                best_ind = ind[:]

        best_cost_history.append(best_cost)

        new_pop: List[List[int]] = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            if random.random() < cx_prob:
                child = ordered_crossover(p1, p2)
            else:
                child = p1[:]
            swap_mutation(child, mut_prob)
            new_pop.append(child)

        population = new_pop

    assert best_ind is not None
    _, start_service = evaluate_route(best_ind, dist_km, travel_min, clients, truck)
    return best_ind, best_cost, best_cost_history, start_service


def plot_convergence(cost_history: List[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(cost_history)), cost_history, marker="o")
    plt.xlabel("Generación")
    plt.ylabel("Mejor coste por generación")
    plt.title("Convergencia GA - Caso 3")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_time_windows(
    perm: List[int],
    start_service: List[float],
    clients: Dict[int, Client],
) -> None:
    plt.figure(figsize=(10, 6))
    y_pos: List[int] = []
    y_labels: List[str] = []
    for idx, cid in enumerate(perm):
        c = clients[cid]
        y = idx
        y_pos.append(y)
        y_labels.append(c.code)
        plt.hlines(y, c.ready, c.due, colors="lightgray", linewidth=6)
        plt.scatter(start_service[idx], y, s=30)
    plt.yticks(y_pos, y_labels)
    plt.xlabel("Minutos desde 00:00")
    plt.title("Cumplimiento de ventanas de tiempo (GA Caso 3)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_route(
    perm: List[int],
    clients: Dict[int, Client],
    depot_lat: float,
    depot_lon: float,
) -> None:
    plt.figure(figsize=(8, 6))
    lats = [depot_lat] + [clients[c].lat for c in perm] + [depot_lat]
    lons = [depot_lon] + [clients[c].lon for c in perm] + [depot_lon]

    plt.plot(lons, lats, marker="o", label="Truck")
    plt.scatter([depot_lon], [depot_lat], marker="s", s=80, label="Depot")

    for cid in perm:
        c = clients[cid]
        plt.text(c.lon, c.lat, c.code, fontsize=8, ha="right", va="bottom")

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Ruta GA - Caso 3 (Camioneta)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def guardar_verificacion_ga(
    best_perm: List[int],
    best_cost: float,
    start_service: List[float],
    clients: Dict[int, Client],
    truck: TruckParams,
    dist_km: np.ndarray,
    travel_min: np.ndarray,
    instancia: str = "Caso3",
    metodo: str = "GA",
) -> None:
    depot = 0
    current = depot
    remaining_cap = truck.capacity
    total_dist = 0.0
    for cid in best_perm:
        c = clients[cid]
        if c.demand > remaining_cap:
            total_dist += dist_km[current, depot]
            current = depot
            remaining_cap = truck.capacity
        total_dist += dist_km[current, cid]
        remaining_cap -= c.demand
        current = cid
    total_dist += dist_km[current, depot]

    cumple_capacidad = 1
    cumple_rango = int(total_dist <= truck.vrange)

    cumple_ventanas = 1
    for idx, cid in enumerate(best_perm):
        c = clients[cid]
        t = start_service[idx]
        if not (c.ready <= t <= c.due):
            cumple_ventanas = 0
            break

    ruta_str = "-".join(clients[c].code for c in best_perm)

    filename = OUT_DIR / f"verificacion_metaheuristica_{metodo}_{instancia}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metodo",
            "instancia",
            "mejor_costo",
            "ruta",
            "cumple_capacidad",
            "cumple_rango",
            "cumple_ventanas",
        ])
        writer.writerow([
            metodo,
            instancia,
            round(best_cost, 2),
            ruta_str,
            cumple_capacidad,
            cumple_rango,
            cumple_ventanas,
        ])
    print(f"Archivo de verificación generado en: {filename}")


def main() -> None:
    random.seed(0)

    clients, truck, dist_km, travel_min, depot_lat, depot_lon = load_case3_data()

    best_perm, best_cost, cost_hist, start_service = run_ga(
        clients,
        truck,
        dist_km,
        travel_min,
        pop_size=100,
        generations=150,
        cx_prob=0.9,
        mut_prob=0.3,
    )

    print("Mejor costo GA Caso 3:", best_cost)
    print("Ruta en orden:", [clients[c].code for c in best_perm])
    print("Inicio servicio:", [minutes_to_hhmm(t) for t in start_service])

    plot_convergence(cost_hist)
    plot_time_windows(best_perm, start_service, clients)
    plot_route(best_perm, clients, depot_lat, depot_lon)

    guardar_verificacion_ga(
        best_perm,
        best_cost,
        start_service,
        clients,
        truck,
        dist_km,
        travel_min,
        instancia="Caso3",
        metodo="GA",
    )


if __name__ == "__main__":
    main()