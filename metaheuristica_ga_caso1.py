from typing import List, Dict, Tuple, Optional, Any
import random
import math
import logging
import copy

import numpy as np
import pandas as pd

# Importar utilidades del caso
from caso1 import calcular_metricas_ruta, construir_matriz_distancias
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def decode_permutation_to_routes(perm: List[int], VEH: List[str], capacities: Dict[str, float],
                                 demandas: List[float]) -> Tuple[Dict[str, List[int]], List[int]]:
    rutas: Dict[str, List[int]] = {v: [] for v in VEH}
    rem_cap = {v: capacities[v] for v in VEH}

    leftovers: List[int] = []
    veh_idx = 0
    n_veh = len(VEH)

    for c in perm:
        assigned = False
        attempts = 0
        start = veh_idx
        while attempts < n_veh:
            v = VEH[veh_idx]
            if demandas[c-1] <= rem_cap[v] + 1e-9:
                rutas[v].append(c)
                rem_cap[v] -= demandas[c-1]
                assigned = True
                break
            veh_idx = (veh_idx + 1) % n_veh
            attempts += 1
        if not assigned:
            leftovers.append(c)

    return rutas, leftovers


def repair_insert(leftovers: List[int], rutas: Dict[str, List[int]], VEH: List[str],
                  capacities: Dict[str, float], demandas: List[float], D: np.ndarray,
                  df_clients: pd.DataFrame, params_por_veh: Dict[str, Dict[str,float]], velocidades: Dict[str,float]) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Intentar insertar clientes sobrantes en las rutas existentes buscando la
    mejor posición (mínima penalización/coste). Devuelve (rutas_mod, remaining).
    """
    remaining = leftovers[:]
    changed = True
    while changed and remaining:
        changed = False
        for cliente in remaining[:]:
            mejor_gain = -1e-9
            mejor_v = None
            mejor_pos = None
            for v in VEH:
                # comprobar capacidad disponible
                carga = sum(demandas[c-1] for c in rutas[v])
                if carga + demandas[cliente-1] > capacities[v] + 1e-9:
                    continue
                # probar todas las posiciones de inserción
                for pos in range(0, len(rutas[v]) + 1):
                    candidate = rutas[v][:]
                    candidate.insert(pos, cliente)
                    # calcular aumento de coste
                    # usamos la distancia de ruta para decidir (más barato -> mejor)
                    before = distancia_ruta_safe(rutas[v], D)
                    after = distancia_ruta_safe(candidate, D)
                    gain = before - after
                    if gain < 0:  
                        # preferimos menor aumento absoluto 
                        pass
                    # buscamos minimizar aumento -> maximizar 
                    if (before - after) > mejor_gain:
                        mejor_gain = (before - after)
                        mejor_v = v
                        mejor_pos = pos
            if mejor_v is not None:
                rutas[mejor_v].insert(mejor_pos, cliente)
                remaining.remove(cliente)
                changed = True
        # si no se pudo insertar ninguno, salimos
        if not changed:
            break
    return rutas, remaining

def distancia_ruta_safe(ruta: List[int], D: np.ndarray) -> float:
    if not ruta:
        return 0.0
    seq = [0] + ruta + [0]
    d = 0.0
    for a,b in zip(seq[:-1], seq[1:]):
        d += float(D[a,b])
    return d

def fitness_from_routes(rutas: Dict[str, List[int]], D: np.ndarray, ids: List[str], df_clients: pd.DataFrame,
                        params_por_veh: Dict[str, Dict[str,float]], velocidades: Dict[str,float],
                        capacities: Dict[str,float], penalty_unassigned: float=1e8) -> Tuple[float, Dict[str,Any]]:
    """
    Calcula el coste total (operativo) de una solución (rutas). Si hay violaciones
    de capacidad o autonomía se suman penalizaciones.
    Retorna (cost_total, detalle)
    """
    total = 0.0
    n_unassigned = 0
    detalle = {}
    for v, ruta in rutas.items():
        if not ruta:
            detalle[v] = {"OperationCost": 0.0}
            continue
        met = calcular_metricas_ruta(ruta, D, ids, df_clients, params_por_veh[v], velocidades[v])
        cost = float(met["OperationCost"])
        total += cost
        detalle[v] = met
        # penalizaciones simples (si se sobrepasa capacidad, se penaliza)
        carga = sum(df_clients.reset_index(drop=True).loc[c-1, "Demand"] for c in ruta)
        if carga > capacities[v] + 1e-6:
            total += penalty_unassigned + (carga - capacities[v]) * 1e4
    detalle["unassigned"] = n_unassigned
    return total, detalle

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(p1, p2):
        child = [-1]*n
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]; idx += 1
        return child
    return ox(parent1, parent2), ox(parent2, parent1)

def mutate_swap(perm: List[int], mutation_prob: float) -> List[int]:
    if random.random() > mutation_prob:
        return perm
    n = len(perm)
    i, j = random.sample(range(n), 2)
    perm2 = perm[:]
    perm2[i], perm2[j] = perm2[j], perm2[i]
    return perm2

def mutate_inversion(perm: List[int], mutation_prob: float) -> List[int]:
    if random.random() > mutation_prob:
        return perm
    n = len(perm)
    i, j = sorted(random.sample(range(n), 2))
    perm2 = perm[:]
    perm2[i:j+1] = list(reversed(perm2[i:j+1]))
    return perm2

def tournament_select(pop: List[List[int]], scores: List[float], k: int=3) -> List[int]:
    sel = random.sample(range(len(pop)), k)
    sel_best = min(sel, key=lambda i: scores[i])
    return pop[sel_best]

def run_ga(deposito_coord: Tuple[float,float], df_clients: pd.DataFrame, df_vehicles: pd.DataFrame,
           params_por_veh: Dict[str, Dict[str,float]], velocidades: Dict[str,float],
           pop_size: int=60, generations: int=150, crossover_prob: float=0.9,
           mutation_prob: float=0.2, elitism: int=2, random_seed: Optional[int]=None,
           D_mat: Optional[np.ndarray]=None) -> Tuple[Dict[str,List[int]], float, Dict[str,Any]]:
    """
    Ejecuta el GA y devuelve la mejor solución encontrada.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # preparar datos
    if D_mat is None:
        D_mat, ids = construir_matriz_distancias(deposito_coord, df_clients)
    else:
        _, ids = construir_matriz_distancias(deposito_coord, df_clients)

    n = len(df_clients)
    VEH = []
    for idx, vr in df_vehicles.reset_index(drop=True).iterrows():
        vid = str(vr.get("StandardizedID", vr.get("VehicleID", f"V{idx+1:03d}")))
        VEH.append(vid)

    capacities = {str(vr.get("StandardizedID", vr.get("VehicleID", f"V{idx+1:03d}"))): float(vr.get("Capacity", vr.get("capacity", 0.0)))
                  for idx, vr in df_vehicles.reset_index(drop=True).iterrows()}

    demandas = df_clients.reset_index(drop=True)["Demand"].astype(float).tolist()

    # población inicial: permutaciones aleatorias
    population = [random.sample(list(range(1, n+1)), n) for _ in range(pop_size)]

    # evaluar
    scores = []
    penalties = 1e7
    for ind in population:
        rutas, leftovers = decode_permutation_to_routes(ind, VEH, capacities, demandas)
        # intentar reparar inserciones
        rutas, remaining = repair_insert(leftovers, rutas, VEH, capacities, demandas, D_mat, df_clients, params_por_veh, velocidades)
        cost, detalle = fitness_from_routes(rutas, D_mat, ids, df_clients, params_por_veh, velocidades, capacities, penalty_unassigned=penalties)
        # penalizar clientes no asignados
        cost += len(remaining) * penalties
        scores.append(cost)

    best_idx = int(np.argmin(scores))
    best_cost = scores[best_idx]
    best_perm = population[best_idx][:]
    LOG.info(f"Inicial: mejor coste {best_cost:.2f}")

    history: List[float] = []
    for gen in range(generations):
        new_pop: List[List[int]] = []
        new_scores: List[float] = []

        # elitismo
        ranked = sorted(range(len(population)), key=lambda i: scores[i])
        for i in ranked[:elitism]:
            new_pop.append(population[i][:])
            new_scores.append(scores[i])

        # generar el resto
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, scores, k=3)
            p2 = tournament_select(population, scores, k=3)
            if random.random() < crossover_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = mutate_swap(c1, mutation_prob)
            c1 = mutate_inversion(c1, mutation_prob*0.5)
            c2 = mutate_swap(c2, mutation_prob)
            c2 = mutate_inversion(c2, mutation_prob*0.5)

            for child in (c1, c2):
                if len(new_pop) >= pop_size:
                    break
                rutas, leftovers = decode_permutation_to_routes(child, VEH, capacities, demandas)
                rutas, remaining = repair_insert(leftovers, rutas, VEH, capacities, demandas, D_mat, df_clients, params_por_veh, velocidades)
                cost, detalle = fitness_from_routes(rutas, D_mat, ids, df_clients, params_por_veh, velocidades, capacities, penalty_unassigned=penalties)
                cost += len(remaining) * penalties
                new_pop.append(child)
                new_scores.append(cost)

        population = new_pop
        scores = new_scores

        gen_best_idx = int(np.argmin(scores))
        gen_best_cost = scores[gen_best_idx]
        history.append(gen_best_cost)
        if gen_best_cost + 1e-6 < best_cost:
            best_cost = gen_best_cost
            best_perm = population[gen_best_idx][:]
            LOG.info(f"Gen {gen}: nuevo mejor {best_cost:.2f}")

    # reconstruir rutas finales para el mejor individuo
    best_routes, leftovers = decode_permutation_to_routes(best_perm, VEH, capacities, demandas)
    best_routes, remaining = repair_insert(leftovers, best_routes, VEH, capacities, demandas, D_mat, df_clients, params_por_veh, velocidades)
    best_cost, detalle = fitness_from_routes(best_routes, D_mat, ids, df_clients, params_por_veh, velocidades, capacities, penalty_unassigned=penalties)
    best_cost += len(remaining) * penalties

    resumen = {
        "best_cost": best_cost,
        "unassigned_after_repair": len(remaining),
        "generations": generations,
        "population": pop_size
    }

    resumen["history"] = history
    return best_routes, best_cost, resumen


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import json
    import matplotlib.pyplot as plt
    import pandas as pd

    OUT_DIR = Path("verificaciones") / "GA"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PICKLE = Path("cleaned_data/almacenamiento_datos.pkl")
    if not PICKLE.exists():
        LOG.error("No se encuentra el pickle de datos: cleaned_data/almacenamiento_datos.pkl")
        raise SystemExit(1)

    data = pickle.load(open(PICKLE, "rb"))
    caso = data.get("Proyecto_Caso_Base", data)

    clients = None
    vehicles = None
    depots = None
    for k, v in caso.items():
        if isinstance(v, pd.DataFrame):
            k_low = str(k).lower()
            if "client" in k_low:
                clients = v.copy().reset_index(drop=True)
            elif "vehicle" in k_low:
                vehicles = v.copy().reset_index(drop=True)
            elif "depot" in k_low or "depots" in k_low:
                depots = v.copy().reset_index(drop=True)

    if clients is None or vehicles is None or depots is None:
        LOG.error("No se encontraron las tablas clients/vehicles/depots en el pickle.")
        raise SystemExit(1)

    if "StandardizedID" not in clients.columns:
        clients["StandardizedID"] = clients.index.to_series().apply(lambda i: f"C{i+1:03d}")
    clients["Latitude"] = clients["Latitude"].astype(float)
    clients["Longitude"] = clients["Longitude"].astype(float)
    clients["Demand"] = clients["Demand"].astype(float)

    dep = depots.iloc[0]
    deposito = (float(dep.get("Latitude", dep.get("Lat", 0.0))), float(dep.get("Longitude", dep.get("Long", 0.0))))

    # preparar params por veh
    params_por_veh: Dict[str, Dict[str, float]] = {}
    velocidades: Dict[str, float] = {}
    for i, vr in vehicles.reset_index(drop=True).iterrows():
        vid = str(vr.get("StandardizedID", vr.get("VehicleID", f"V{i+1:03d}")))
        params_por_veh[vid] = {
            "Rv": float(vr.get("FuelEfficiency", 30.0)),
            "Cv": float(vr.get("FixedCost", 50000.0)),
            "Mv": float(vr.get("M", 500.0)),
            "Ct": 3000.0,
            "Pf": 16300.0,
            "Ctime": 0.0,
        }
        velocidades[vid] = float(vr.get("Speed", 40.0))

    LOG.info("Ejecutando GA (metaheuristica_ga.run_ga)...")
    # parámetros moderados que puedes ajustar
    best_routes, best_cost, resumen = run_ga(deposito, clients, vehicles, params_por_veh, velocidades,
                                            pop_size=40, generations=80, crossover_prob=0.9,
                                            mutation_prob=0.2, elitism=2, random_seed=1)

    LOG.info(f"GA finalizado. Mejor coste: {best_cost}")

    # reconstruir métricas por vehículo
    D_mat, ids = construir_matriz_distancias(deposito, clients)
    filas: List[Dict[str, Any]] = []
    total_cost = 0.0
    total_distance = 0.0
    for v, ruta in best_routes.items():
        if not ruta:
            continue
        met = calcular_metricas_ruta(ruta, D_mat, ids, clients, params_por_veh[v], velocidades[v])
        fila = {
            "VehicleId": v,
            "InitialLoad": int(met["InitialLoad"]),
            "RouteSequence": met["RouteSequence"],
            "ClientsServed": int(met["ClientsServed"]),
            "DemandsSatisfied": met.get("DemandSatisfied", met.get("DemandsSatisfied", "")),
            "TotalDistance": float(met["TotalDistance"]),
            "TotalTime": float(met["TotalTime"]),
            "FuelCost": int(met["FuelCost"]),
            "OperationCost": int(round(float(met["OperationCost"])))
        }
        filas.append(fila)
        total_cost += float(met["OperationCost"])
        total_distance += float(met["TotalDistance"])

    # CSV y JSON output
    inst_name = "caso_base"
    csv_path = OUT_DIR / f"verificacion_metaheuristica_GA_{inst_name}.csv"
    json_path = OUT_DIR / f"verificacion_metaheuristica_GA_{inst_name}.json"
    df_out = pd.DataFrame(filas, columns=["VehicleId","InitialLoad","RouteSequence","ClientsServed","DemandsSatisfied","TotalDistance","TotalTime","FuelCost","OperationCost"])
    df_out.to_csv(csv_path, index=False)

    resumen_out = {
        "routes": filas,
        "total_cost": int(round(total_cost)),
        "total_distance": round(total_distance, 3),
        "vehicle_count": len(filas),
        "solver_used": "GA",
        "ga_summary": resumen
    }
    with open(json_path, "w", encoding="utf8") as fh:
        json.dump(resumen_out, fh, indent=2, ensure_ascii=False)

    LOG.info(f"Guardados CSV {csv_path} y JSON {json_path}")

    # GRAFICOS: convergencia
    history = resumen.get("history", [])
    if history:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(history)+1), history, marker='o')
        plt.xlabel('Generación')
        plt.ylabel('Mejor coste (por generación)')
        plt.title('Convergencia GA - Proyecto_Caso_Base')
        conv_path = OUT_DIR / f"metaheuristica_GA_{inst_name}_convergence.png"
        plt.grid(True); plt.tight_layout(); plt.savefig(conv_path, dpi=200)
        LOG.info(f"Guardada curva de convergencia: {conv_path}")

    # GRAFICO: rutas finales sobre mapa simple
    plt.figure(figsize=(10,8))
    clientes = clients.reset_index(drop=True)
    depo_lat, depo_lon = deposito[0], deposito[1]
    plt.scatter([depo_lon], [depo_lat], marker='s', s=120, c='black', label='Depot CD01')
    plt.scatter(clientes['Longitude'], clientes['Latitude'], marker='o', s=40, c='gray', label='Clientes')
    for idx, r in clientes.iterrows():
        plt.text(r['Longitude']+0.00015, r['Latitude']+0.00015, r['StandardizedID'], fontsize=8)

    from matplotlib import cm
    cmap = cm.get_cmap('tab10')
    color_i = 0
    for v, ruta in best_routes.items():
        color = cmap(color_i % 10)
        color_i += 1
        pts = ["CD01"] + [clients.loc[c-1, 'StandardizedID'] for c in ruta] + ["CD01"]
        xs = []
        ys = []
        for token in pts:
            if token == 'CD01':
                xs.append(depo_lon); ys.append(depo_lat)
            else:
                latlon = clientes.loc[clientes['StandardizedID'] == token, ['Latitude','Longitude']].values
                if latlon.shape[0] == 0:
                    continue
                lat, lon = latlon[0]
                xs.append(lon); ys.append(lat)
        plt.plot(xs, ys, linestyle='-', linewidth=2, color=color, label=f"{v} ({len(ruta)})")
        plt.scatter(xs[1:-1], ys[1:-1], s=60, color=color)

    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title('Rutas GA - Proyecto_Caso_Base')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True); plt.tight_layout()
    routes_path = OUT_DIR / f"metaheuristica_GA_{inst_name}_rutas.png"
    plt.savefig(routes_path, dpi=200)
    LOG.info(f"Guardado plot de rutas: {routes_path}")

    # GRAFICOS
    loads = [int(row['InitialLoad']) for row in filas]
    lengths = [float(row['TotalDistance']) for row in filas]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.hist(loads, bins=10, color='C0', edgecolor='k')
    plt.title('Histograma de cargas por vehículo')
    plt.xlabel('Carga (unidades)')
    plt.ylabel('Frecuencia')
    plt.subplot(1,2,2)
    plt.boxplot([loads, lengths], labels=['Cargas','Longitudes'])
    plt.title('Boxplot: cargas y longitudes')
    plt.tight_layout()
    hist_path = OUT_DIR / f"metaheuristica_GA_{inst_name}_histograms.png"
    plt.savefig(hist_path, dpi=200)
    LOG.info(f"Guardadas distribuciones: {hist_path}")

    LOG.info('Ejecución completa.')
