"""
 1) Cargar datos 
 2) Usar heurística (fleet-aware) para simular rutas por vehículo
    y construir matriz de costos estimados C_est[(c,v)] y dist_est[(c,v)]
 3) Resolver pequeño Pyomo (Z,Y) con esos costos (se fuerza GLPK)
 4) Reconstruir rutas reales por vehículo (NN)
 5) Optimizar rutas: 2-OPT + recolocaciones intra-ruta + recolocaciones inter-rutas
 6) Calcular costos reales, guardar CSV/JSON y graficar rutas

Salida (en carpeta verificaciones/caso1/):
 - verificacion_caso1.csv
 - verificacion_caso1_resumen.json
 - verificacion_caso1_rutas.png
"""
from pathlib import Path
import pickle
import math
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Rutas de salida y constantes
ALMACEN_PICKLE = Path("cleaned_data/almacenamiento_datos.pkl")
OUT_DIR = Path("verificaciones/caso1")
OUT_CSV = OUT_DIR / "verificacion_caso1.csv"
OUT_JSON = OUT_DIR / "verificacion_caso1_resumen.json"
OUT_PNG = OUT_DIR / "verificacion_caso1_rutas.png"

# Valores por defecto (según enunciado)
PF_DEF = 16300.0       # COP / gallon
CT_DEF = 3000.0        # COP / km transporte
CV_DEF = 50000.0       # COP costo fijo vehículo
MV_DEF = 500.0         # COP / km mantenimiento
RV_DEF = 30.0          # km / gallon
SPEED_DEF = 40.0       # km/h
CTIME_DEF = 0.0        # COP / hora
EPS = 1e-9
COST_INF = 1e12

# UTILIDADES GEOGRÁFICAS Y DE E/S
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia Haversine (km) entre dos coordenadas en grados."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def buscar_df_por_nombre(contenedor: Dict[str, Any], patron: str) -> List[pd.DataFrame]:
    """
    Busca recursivamente DataFrames dentro de un diccionario anidado por nombre o clave
    que contenga el 'patron' en su key.
    """
    encontrados: List[pd.DataFrame] = []
    def _recorrer(nodo):
        if isinstance(nodo, dict):
            for k, v in nodo.items():
                if isinstance(v, pd.DataFrame):
                    if patron.lower() in str(k).lower():
                        encontrados.append(v)
                else:
                    _recorrer(v)
    _recorrer(contenedor)
    return encontrados

def cargar_pickle(path: Path) -> Dict[str, Any]:
    """Carga el pickle con las tablas (raise si no existe)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta antes el pipeline de limpieza.")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    logging.info("Almacenamiento cargado desde pickle.")
    return data

# MATRIZ DE DISTANCIAS Y HEURÍSTICAS BASE
def construir_matriz_distancias(deposito: Tuple[float,float], clientes_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Construye matriz de distancia indexada 0..n, donde 0 = deposito,
    1..n = clientes. Retorna (D, ids) donde ids = ["CD01","C001",...].
    """
    clientes = clientes_df.reset_index(drop=True)
    n = len(clientes)
    ids = ["CD01"] + clientes["StandardizedID"].astype(str).tolist()
    coords = [(deposito[0], deposito[1])] + list(zip(clientes["Latitude"].astype(float), clientes["Longitude"].astype(float)))
    D = np.zeros((n+1, n+1), dtype=float)
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                D[i,j] = 0.0
            else:
                D[i,j] = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return D, ids

def heuristica_flotilla_semilla(deposito: Tuple[float,float], clientes_df: pd.DataFrame, semilla_cliente: int, capacidad: float) -> List[int]:
    """
    Heurística simple que construye una ruta iniciando por un cliente semilla
    y llenando por cercanía respetando capacidad.
    Devuelve lista de índices de clientes (1..n).
    """
    clientes = clientes_df.reset_index(drop=True)
    demandas = clientes["Demand"].astype(float).tolist()
    n = len(clientes)
    D, ids = construir_matriz_distancias(deposito, clientes_df)
    servidos = [False] * n
    ruta: List[int] = []
    cap_rest = capacidad

    # comprobar semilla viable
    if demandas[semilla_cliente - 1] <= cap_rest + 1e-9:
        ruta.append(semilla_cliente)
        servidos[semilla_cliente - 1] = True
        cap_rest -= demandas[semilla_cliente - 1]
        actual = semilla_cliente
    else:
        return []

    while True:
        candidatos = []
        for idx in range(1, n+1):
            if (not servidos[idx - 1]) and (demandas[idx - 1] <= cap_rest + 1e-9):
                candidatos.append((D[actual, idx], idx))
        if not candidatos:
            break
        candidatos.sort(key=lambda x: x[0])
        _, elegido = candidatos[0]
        ruta.append(elegido)
        servidos[elegido - 1] = True
        cap_rest -= demandas[elegido - 1]
        actual = elegido
    return ruta

# CÁLCULO DE MÉTRICAS DE RUTA
def calcular_metricas_ruta(ruta_idxs: List[int],
                           D: np.ndarray,
                           ids: List[str],
                           clientes_df: pd.DataFrame,
                           params_veh: Dict[str, float],
                           velocidad_kmh: float) -> Dict[str, Any]:
    """
    Calcula distancia, tiempo, costo de combustible y costo operativo total
    para una ruta dada (lista de indices de clientes 1..n).
    Retorna diccionario con claves:
      RouteSequence, ClientsServed, DemandSatisfied, TotalDistance, TotalTime,
      InitialLoad, FuelCost, OperationCost
    """
    seq_nodos = [0] + ruta_idxs + [0]
    distancia_total = 0.0
    for a,b in zip(seq_nodos[:-1], seq_nodos[1:]):
        distancia_total += float(D[a,b])
    demandas = clientes_df.reset_index(drop=True)["Demand"].astype(float).tolist()
    carga = sum(demandas[i-1] for i in ruta_idxs) if ruta_idxs else 0.0
    demandas_satisfechas = "-".join(str(int(demandas[i-1])) for i in ruta_idxs) if ruta_idxs else ""
    tiempo_min = distancia_total / max(velocidad_kmh, 1e-6) * 60.0
    tiempo_horas = tiempo_min / 60.0

    Ct = params_veh.get("Ct", CT_DEF)
    Pf = params_veh.get("Pf", PF_DEF)
    Rv = params_veh.get("Rv", RV_DEF)
    Mv = params_veh.get("Mv", MV_DEF)
    Cv = params_veh.get("Cv", CV_DEF)
    Ctime = params_veh.get("Ctime", CTIME_DEF)

    consumo_galon = distancia_total / max(Rv, 1e-6)
    costo_combustible = consumo_galon * Pf

    costo_operativo = Cv + distancia_total * (Ct + Mv) + costo_combustible + tiempo_horas * Ctime

    return {
        "RouteSequence": " - ".join([ids[i] for i in seq_nodos]),
        "ClientsServed": len(ruta_idxs),
        "DemandSatisfied": demandas_satisfechas,
        "TotalDistance": round(float(distancia_total), 3),
        "TotalTime": round(float(tiempo_min), 2),
        "InitialLoad": int(carga),
        "FuelCost": int(round(costo_combustible)),
        "OperationCost": float(costo_operativo)
    }

# MEJORADORES DE RUTA  2-OPT
def distancia_ruta(ruta: List[int], D: np.ndarray) -> float:
    seq = [0] + ruta + [0]
    d = 0.0
    for a,b in zip(seq[:-1], seq[1:]):
        d += D[a,b]
    return d

def dos_opt_ruta(ruta: List[int], D: np.ndarray) -> List[int]:
    """
    Aplica búsqueda local 2-OPT para mejorar la secuencia de una ruta.
    """
    if len(ruta) < 3:
        return ruta[:]
    mejor = ruta[:]
    mejoro = True
    while mejoro:
        mejoro = False
        n = len(mejor)
        dist_mejor = distancia_ruta(mejor, D)
        for i in range(0, n - 1):
            for j in range(i+1, n):
                if j - i == 0:
                    continue
                nueva = mejor[:]
                nueva[i:j+1] = reversed(nueva[i:j+1])
                dist_nueva = distancia_ruta(nueva, D)
                if dist_nueva + 1e-6 < dist_mejor:
                    mejor = nueva
                    dist_mejor = dist_nueva
                    mejoro = True
        # Repetir hasta convergencia
    return mejor

def recolocar_intra_ruta(ruta: List[int], D: np.ndarray) -> List[int]:
    """
    Intenta mover un cliente dentro de la misma ruta a otra posición si mejora distancia.
    """
    if len(ruta) < 3:
        return ruta[:]
    mejor = ruta[:]
    mejoro = True
    while mejoro:
        mejoro = False
        n = len(mejor)
        dist_mejor = distancia_ruta(mejor, D)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                nueva = mejor[:]
                elem = nueva.pop(i)
                nueva.insert(j, elem)
                dist_nueva = distancia_ruta(nueva, D)
                if dist_nueva + 1e-6 < dist_mejor:
                    mejor = nueva
                    dist_mejor = dist_nueva
                    mejoro = True
                    break
            if mejoro:
                break
    return mejor

def recolocar_inter_rutas(rutas: Dict[str, List[int]],
                          lista_veh: List[str],
                          D: np.ndarray,
                          clientes_df: pd.DataFrame,
                          capacidades: Dict[str, float],
                          params_por_veh: Dict[str, Dict[str,float]],
                          velocidades: Dict[str, float]) -> Tuple[Dict[str,List[int]], bool]:
    """
    Intenta mover un cliente de la ruta de un vehículo a la ruta de otro vehículo
    si respeta capacidad y reduce el costo total (suma costo operativo de ambas rutas).
    Retorna (rutas_modificadas, hubo_mejora).
    """
    hubo_mejora = False
    demandas = clientes_df["Demand"].astype(float).tolist()

    # lista de vehículos con ruta no vacía
    vehiculos_activos = [v for v in lista_veh if rutas.get(v)]

    def costo_ruta_local(ruta, veh_id):
        if not ruta:
            return 0.0
        met = calcular_metricas_ruta(ruta, D, ["CD01"] + clientes_df["StandardizedID"].tolist(), clientes_df, params_por_veh[veh_id], velocidades[veh_id])
        return float(met["OperationCost"])

    # intentar mejoras de manera greedy
    for i in range(len(vehiculos_activos)):
        for j in range(len(vehiculos_activos)):
            if i == j:
                continue
            v_origen = vehiculos_activos[i]
            v_destino = vehiculos_activos[j]
            ruta_origen = rutas[v_origen][:]
            ruta_destino = rutas[v_destino][:]
            if not ruta_origen:
                continue
            carga_origen = sum(demandas[c-1] for c in ruta_origen)
            carga_destino = sum(demandas[c-1] for c in ruta_destino)
            # probar cada cliente de la ruta_origen
            for pos, cliente in enumerate(ruta_origen):
                dem_c = demandas[cliente - 1]
                if carga_destino + dem_c > capacidades[v_destino] + 1e-9:
                    continue
                nueva_origen = ruta_origen[:]
                nueva_origen.pop(pos)
                mejor_ganancia_local = 1e-6
                mejor_origen = None
                mejor_destino = None
                costo_antes = costo_ruta_local(ruta_origen, v_origen) + costo_ruta_local(ruta_destino, v_destino)
                # intentar insertar en todas las posiciones de la ruta_destino
                for pos_ins in range(0, len(ruta_destino) + 1):
                    nueva_destino = ruta_destino[:]
                    nueva_destino.insert(pos_ins, cliente)
                    costo_despues = costo_ruta_local(nueva_origen, v_origen) + costo_ruta_local(nueva_destino, v_destino)
                    if costo_despues + 1e-6 < costo_antes and (costo_antes - costo_despues) > mejor_ganancia_local:
                        mejor_ganancia_local = costo_antes - costo_despues
                        mejor_origen = nueva_origen[:]
                        mejor_destino = nueva_destino[:]
                if mejor_origen is not None:
                    # aceptar el movimiento
                    rutas[v_origen] = mejor_origen
                    rutas[v_destino] = mejor_destino
                    hubo_mejora = True
                    # recomputar vehiculos_activos y salir para reiniciar proceso
                    vehiculos_activos = [v for v in lista_veh if rutas.get(v)]
                    break
            if hubo_mejora:
                break
        if hubo_mejora:
            break

    return rutas, hubo_mejora

# PRINCIPAL
def ejecutar_caso1(nombre_caso: Optional[str] = "Proyecto_Caso_Base"):
    """
    Ejecución completa del flujo:
     - Carga de pickle
     - Precomputo de costos estimados por semilla
     - Construcción y resolución Pyomo (Z, Y)
     - Reconstrucción de rutas (NN)
     - Optimización local (2-opt, recolocar)
     - Cálculo final de costos exactos
     - Guarda CSV/JSON y grafica las rutas
    """
    # cargar datos
    almacen = cargar_pickle(ALMACEN_PICKLE)
    contenedor = almacen.get(nombre_caso, almacen) if nombre_caso else almacen

    # extraer tablas
    f_clients = buscar_df_por_nombre(contenedor, "clients")
    f_depots  = buscar_df_por_nombre(contenedor, "depots")
    f_vehicles= buscar_df_por_nombre(contenedor, "vehicles")
    if not f_clients or not f_depots or not f_vehicles:
        raise ValueError("No se encontraron las tablas clients/depots/vehicles en el pickle.")
    df_clients = f_clients[0].copy().reset_index(drop=True)
    df_depots = f_depots[0].copy().reset_index(drop=True)
    df_vehicles = f_vehicles[0].copy().reset_index(drop=True)

    # parámetros globales opcionales
    parametros_globales: Dict[str, float] = {}
    for dfp in buscar_df_por_nombre(contenedor, "param"):
        cols = [c.lower() for c in dfp.columns]
        if "parameter" in cols and "value" in cols:
            col_p = dfp.columns[cols.index("parameter")]
            col_v = dfp.columns[cols.index("value")]
            for _, r in dfp.iterrows():
                try:
                    parametros_globales[str(r[col_p])] = None if pd.isna(r[col_v]) else float(r[col_v])
                except:
                    parametros_globales[str(r[col_p])] = None

    # normalizaciones básicas
    if "StandardizedID" not in df_clients.columns:
        df_clients["StandardizedID"] = df_clients.index.to_series().apply(lambda i: f"C{i+1:03d}")
    df_clients["Latitude"] = df_clients["Latitude"].astype(float)
    df_clients["Longitude"] = df_clients["Longitude"].astype(float)
    df_clients["Demand"] = df_clients["Demand"].astype(float)

    # coordenadas del depósito
    dep = df_depots.iloc[0]
    lat_depo = float(dep.get("Latitude", dep.get("Lat", dep.get("latitude", 0.0))))
    lon_depo = float(dep.get("Longitude", dep.get("Long", dep.get("longitude", 0.0))))
    deposito_coord = (lat_depo, lon_depo)

    # construir matriz de distancias
    D_mat, ids = construir_matriz_distancias(deposito_coord, df_clients)

    # preparar flota y parámetros por vehículo
    VEH: List[str] = []
    capacidades: Dict[str, float] = {}
    autonomia: Dict[str, float] = {}
    rendimiento: Dict[str, float] = {}
    costo_fijo: Dict[str, float] = {}
    costo_mantenimiento: Dict[str, float] = {}
    velocidad_por_veh: Dict[str, float] = {}
    params_por_veh: Dict[str, Dict[str,float]] = {}

    for idx, vr in df_vehicles.reset_index(drop=True).iterrows():
        vid = str(vr.get("StandardizedID", vr.get("VehicleID", f"V{idx+1:03d}")))
        VEH.append(vid)
        capacidades[vid] = float(vr.get("Capacity", vr.get("capacity", 0.0)))
        autonomia[vid] = float(vr.get("Range", vr.get("Autonomy", vr.get("range", 300.0))))
        rendimiento[vid] = float(vr.get("FuelEfficiency", parametros_globales.get("Rv", RV_DEF)))
        costo_fijo[vid] = float(vr.get("FixedCost", parametros_globales.get("Cv", CV_DEF)))
        costo_mantenimiento[vid] = float(vr.get("M", parametros_globales.get("Mv", MV_DEF)))
        velocidad_por_veh[vid] = float(vr.get("Speed", vr.get("speed", SPEED_DEF)))
        params_por_veh[vid] = {
            "Av": autonomia[vid],
            "Rv": rendimiento[vid],
            "Cv": costo_fijo[vid],
            "Mv": costo_mantenimiento[vid],
            "Ct": float(parametros_globales.get("Ct", CT_DEF)),
            "Pf": float(parametros_globales.get("Pf", PF_DEF)),
            "Ctime": float(parametros_globales.get("Ctime", CTIME_DEF))
        }

    Ct = float(parametros_globales.get("Ct", CT_DEF))
    Pf = float(parametros_globales.get("Pf", PF_DEF))

    logging.info(f"Clientes: {len(df_clients)}, Vehículos: {len(VEH)}")
    logging.info(f"Capacidades: {[capacidades[v] for v in VEH]}")
    
    # simular rutas semilla para estimar costos por asignación cliente->vehículo
    n = len(df_clients)
    C_est = {(c,v): COST_INF for c in range(1, n+1) for v in VEH}
    dist_est = {(c,v): 0.0 for c in range(1, n+1) for v in VEH}

    logging.info("Precomputando costos estimados C_est[(c,v)] usando heurística semilla por cliente...")
    for v in VEH:
        cap_v = capacidades[v]
        params_v = params_por_veh[v]
        for c in range(1, n+1):
            # saltar semillas inviables por demanda
            if df_clients.loc[c-1, "Demand"] > cap_v + 1e-9:
                C_est[(c,v)] = COST_INF
                continue
            ruta_sim = heuristica_flotilla_semilla(deposito_coord, df_clients, semilla_cliente=c, capacidad=cap_v)
            if not ruta_sim:
                C_est[(c,v)] = COST_INF
                continue
            met = calcular_metricas_ruta(ruta_sim, D_mat, ids, df_clients, params_v, velocidad_por_veh[v])
            costo_ruta = float(met["OperationCost"])
            dist_ruta = float(met["TotalDistance"])
            # repartir costo por cliente como aproximación simple
            compartido_costo = costo_ruta / max(len(ruta_sim), 1)
            compartido_dist = dist_ruta / max(len(ruta_sim), 1)
            for cliente_en_ruta in ruta_sim:
                key = (cliente_en_ruta, v)
                if compartido_costo < C_est[key]:
                    C_est[key] = compartido_costo
                    dist_est[key] = compartido_dist
    logging.info("Precomputación completada.")

    # MODELO PYOMO (ASIGNACIÓN Z, Y)
    modelo = pyo.ConcreteModel()
    modelo.C = pyo.Set(initialize=list(range(1, n+1)))
    modelo.V = pyo.Set(initialize=VEH)

    modelo.q = pyo.Param(modelo.C, initialize=lambda m,c: float(df_clients.loc[c-1,"Demand"]))
    modelo.cap = pyo.Param(modelo.V, initialize=lambda m,v: capacidades[v])
    modelo.Av = pyo.Param(modelo.V, initialize=lambda m,v: autonomia[v])
    modelo.Cv = pyo.Param(modelo.V, initialize=lambda m,v: costo_fijo[v])

    modelo.Z = pyo.Var(((c,v) for c in modelo.C for v in modelo.V), domain=pyo.Binary)
    modelo.Y = pyo.Var(modelo.V, domain=pyo.Binary)

    def regla_asignar_una_vez(m, c):
        return sum(m.Z[c, v] for v in m.V) == 1
    modelo.AsignarUnaVez = pyo.Constraint(modelo.C, rule=regla_asignar_una_vez)

    def regla_capacidad(m, v):
        return sum(m.q[c] * m.Z[c, v] for c in m.C) <= m.cap[v] * m.Y[v]
    modelo.Capacidad = pyo.Constraint(modelo.V, rule=regla_capacidad)

    def regla_autonomia(m, v):
        return sum(dist_est[(c,v)] * m.Z[c, v] for c in m.C) <= m.Av[v] * m.Y[v]
    modelo.Autonomia = pyo.Constraint(modelo.V, rule=regla_autonomia)

    def funcion_objetivo(m):
        fijo = sum(m.Cv[v] * m.Y[v] for v in m.V)
        var = sum(C_est[(c,v)] * m.Z[c, v] for c in m.C for v in m.V)
        return fijo + var
    modelo.OBJ = pyo.Objective(rule=funcion_objetivo, sense=pyo.minimize)

    # RESOLVER 
    solver_name = "glpk"
    solver = pyo.SolverFactory(solver_name)
    if not (solver and solver.available()):
        raise RuntimeError("GLPK no está disponible. Instala con: conda install -c conda-forge glpk")

    logging.info("Resolviendo modelo Pyomo (asignación) con GLPK...")
    resultado = solver.solve(modelo, tee=False)
    logging.info(f"Resultado solver: {resultado.solver.status}, terminación: {resultado.solver.termination_condition}")

    # EXTRAER ASIGNACIONES Z, Y
    Z_val = {(c,v): int(round(pyo.value(modelo.Z[c,v]))) for c in modelo.C for v in modelo.V}
    Y_val = {v: int(round(pyo.value(modelo.Y[v]))) for v in modelo.V}

    asignados_por_veh = {v: [] for v in VEH}
    for (c,v), val in Z_val.items():
        if val == 1:
            asignados_por_veh[v].append(c)

    # RECONSTRUIR RUTAS INICIALES (Nearest Neighbor)
    rutas: Dict[str, List[int]] = {}
    for v in VEH:
        clientes_v = asignados_por_veh[v]
        if not clientes_v:
            rutas[v] = []
            continue
        seq: List[int] = []
        resto = set(clientes_v)
        cur = 0
        while resto:
            nxt = min(resto, key=lambda c: D_mat[cur, c])
            seq.append(nxt)
            resto.remove(nxt)
            cur = nxt
        rutas[v] = seq

    # MEJORAS LOCALES: iterar 2-OPT + recolocar intra + recolocar inter
    MAX_PASADAS = 10
    logging.info("Optimizando rutas con 2-OPT + recolocaciones intra/inter...")
    for pasada in range(MAX_PASADAS):
        mejoro_global = False
        # intra-ruta
        for v in VEH:
            r = rutas.get(v, [])
            if not r:
                continue
            antes = distancia_ruta(r, D_mat)
            r2 = dos_opt_ruta(r, D_mat)
            r2 = recolocar_intra_ruta(r2, D_mat)
            if distancia_ruta(r2, D_mat) + 1e-6 < antes:
                rutas[v] = r2
                mejoro_global = True
        # recolocaciones inter-rutas
        rutas, imp2 = recolocar_inter_rutas(rutas, VEH, D_mat, df_clients, capacidades, params_por_veh, velocidad_por_veh)
        if imp2:
            mejoro_global = True
        if not mejoro_global:
            break
    logging.info("Optimización local finalizada.")

    # CÁLCULO FINAL DE COSTOS 
    filas_verif: List[Dict[str, Any]] = []
    costo_total = 0.0
    distancia_total = 0.0
    fuel_total = 0.0
    fixed_total = 0.0
    variable_total = 0.0
    advertencias: List[str] = []

    for v in VEH:
        seq = rutas.get(v, [])
        if not seq:
            continue
        met = calcular_metricas_ruta(seq, D_mat, ids, df_clients, params_por_veh[v], velocidad_por_veh[v])
        costo_op = float(met["OperationCost"])
        dist = float(met["TotalDistance"])
        fuel_cost = float(met["FuelCost"])
        carga = int(met["InitialLoad"])

        if carga > capacidades[v] + 1e-6:
            advertencias.append(f"Veh {v} carga {carga} > cap {capacidades[v]}")
        if dist > autonomia[v] + 1e-6:
            advertencias.append(f"Veh {v} dist {dist:.2f} km > Av {autonomia[v]} km")

        fila = {
            "VehicleId": v,
            "InitialLoad": carga,
            "RouteSequence": met["RouteSequence"],
            "ClientsServed": met["ClientsServed"],
            "DemandsSatisfied": met["DemandSatisfied"],
            "TotalDistance": met["TotalDistance"],
            "TotalTime": met["TotalTime"],
            "FuelCost": met["FuelCost"],
            "OperationCost": int(round(costo_op))
        }
        filas_verif.append(fila)
        costo_total += costo_op
        distancia_total += dist
        fuel_total += fuel_cost
        fixed_total += costo_fijo[v]
        variable_total += dist * (Ct + costo_mantenimiento[v])  # Mv por vehículo

    # crear carpeta de salida
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # guardar CSV y JSON
    df_out = pd.DataFrame(filas_verif, columns=[
        "VehicleId","InitialLoad","RouteSequence","ClientsServed","DemandsSatisfied",
        "TotalDistance","TotalTime","FuelCost","OperationCost"
    ])
    df_out.to_csv(OUT_CSV, index=False)

    resumen = {
        "routes": filas_verif,
        "total_cost": int(round(costo_total)),
        "total_distance": round(distancia_total, 3),
        "vehicle_count": len(filas_verif),
        "total_fuel_cost": int(round(fuel_total)),
        "total_fixed_cost": int(round(fixed_total)),
        "total_variable_cost": int(round(variable_total)),
        "solver_used": solver_name,
        "warnings": advertencias
    }
    with open(OUT_JSON, "w", encoding="utf8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)

    logging.info(f"Guardados CSV {OUT_CSV} y JSON {OUT_JSON}")
    logging.info(f"Resumen final: vehículos usados={len(filas_verif)}, distancia total={distancia_total:.2f} km, costo total={int(costo_total)} COP")

    # GRAFICAR RUTAS 
    plt.figure(figsize=(10,8))
    cmap = plt.get_cmap("tab10")
    clientes = df_clients.reset_index(drop=True)
    depo_lat, depo_lon = deposito_coord[0], deposito_coord[1]
    plt.scatter([depo_lon], [depo_lat], marker="s", s=120, c="black", label="Depot CD01")
    plt.scatter(clientes["Longitude"], clientes["Latitude"], marker="o", s=40, c="gray", label="Clientes")
    for idx, r in clientes.iterrows():
        plt.text(r["Longitude"] + 0.00015, r["Latitude"] + 0.00015, r["StandardizedID"], fontsize=8)

    color_i = 0
    for fila in filas_verif:
        veh = fila["VehicleId"]
        color = cmap(color_i % 10)
        color_i += 1
        pts = [p.strip() for p in fila["RouteSequence"].split("-") if p.strip()]
        xs: List[float] = []
        ys: List[float] = []
        for token in pts:
            if token == "CD01":
                xs.append(depo_lon); ys.append(depo_lat)
            else:
                latlon = clientes.loc[clientes["StandardizedID"] == token, ["Latitude","Longitude"]].values
                if latlon.shape[0] == 0:
                    continue
                lat, lon = latlon[0]
                xs.append(lon); ys.append(lat)
        plt.plot(xs, ys, linestyle="-", linewidth=2, color=color, label=f"{veh} ({fila['ClientsServed']})")
        plt.scatter(xs[1:-1], ys[1:-1], s=60, color=color)

    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Rutas CVRP - Caso 1 (preheurística -> Pyomo -> rutas optimizadas)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    try:
        plt.show()
    except Exception:
        pass

    # mostrar advertencias
    if advertencias:
        logging.warning("Advertencias detectadas:")
        for w in advertencias:
            logging.warning(" - " + w)

    return df_out, resumen

# main
if __name__ == "__main__":
    df_result, resumen_final = ejecutar_caso1(nombre_caso="Proyecto_Caso_Base")
    print("Resumen:", resumen_final)
