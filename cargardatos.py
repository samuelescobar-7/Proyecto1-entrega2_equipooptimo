"""
Lee, valida y limpia datos para los proyectos:
 - Proyecto_Caso_Base
 - project_b

Salida:
 - reports/integrity_report.json
 - reports/integrity_report_summary.csv
 - cleaned_data
 - además guarda en memoria `almacenamiento_datos` (diccionario) con los DataFrames cargados,
   manteniendo la jerarquía por carpeta y subcarpeta.

Requisitos:
 pip install pandas openpyxl
"""
import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Extensiones de diferentes archivos
EXTENSIONES_SOPORTADAS = [".csv", ".xlsx", ".xls", ".json"]


# Funciones para localizar carpetas y archivos
def encontrar_carpetas_caso(raiz: Path) -> List[Path]:
    """
    Busca en la carpeta raíz las carpetas de interés:
    'Proyecto_Caso_Base' y 'project_b' (acepta variaciones de nombre y mayúsculas).
    Devuelve una lista de Path únicas.
    """
    candidatos = []
    for p in raiz.iterdir():
        if p.is_dir():
            nombre = p.name.lower()
            if "caso_base" in nombre or "proyecto_caso_base" in nombre:
                candidatos.append(p)
            if "project_b" in nombre or "proyecto_b" in nombre or "project-b" in nombre:
                candidatos.append(p)

    # Si no encontró nada en primer nivel, busca recursivamente por patrón
    if not candidatos:
        for p in raiz.rglob("*"):
            if p.is_dir() and re.search(r"proyecto[_\- ]?caso[_\- ]?base|project[_\- ]?b|proyecto[_\- ]?b",
                                       p.name, re.IGNORECASE):
                candidatos.append(p)

    # Desduplicar y ordenar por nombre
    unico = sorted({str(p.resolve()): p for p in candidatos}.values(), key=lambda x: x.name)
    logging.info(f"Carpetas de caso detectadas: {[p.name for p in unico]}")
    return unico


def encontrar_archivos_datos(carpeta_caso: Path) -> List[Path]:
    """
    Busca recursivamente archivos con extensiones soportadas dentro de una carpeta de caso.
    """
    archivos = []
    for ext in EXTENSIONES_SOPORTADAS:
        archivos.extend(carpeta_caso.rglob(f"*{ext}"))
    archivos = [f for f in archivos if ".git" not in str(f) and "__pycache__" not in str(f)]
    return sorted(archivos)


def cargar_tabla(ruta: Path) -> Tuple[pd.DataFrame, str]:
    """
    Intenta cargar un archivo como CSV, Excel o JSON.
    Devuelve el DataFrame y un string con el formato detectado.
    """
    try:
        if ruta.suffix.lower() == ".csv":
            df = pd.read_csv(ruta)
            return df, "csv"
        if ruta.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(ruta)
            return df, "excel"
        if ruta.suffix.lower() == ".json":
            df = pd.read_json(ruta)
            return df, "json"
    except Exception as e:
        logging.warning(f"Fallo al leer {ruta}: {e}")
        try:
            df = pd.read_csv(ruta, encoding="latin1", on_bad_lines="skip")
            return df, "csv_fallback"
        except Exception:
            raise
    raise ValueError(f"Formato no soportado o fallo al leer: {ruta}")

# Limpieza de datos numéricos
def intentar_parsear_numero(texto):
    """
    Intenta extraer un número desde una cadena.
    - Maneja comas decimales (si es seguro).
    - Extrae el primer patrón numérico si existe ('10 km' -> 10.0).
    - Devuelve float o None si no se puede parsear.
    """
    if pd.isna(texto):
        return None
    s = str(texto).strip()
    if s == "":
        return None
    s = s.replace("−", "-")  # reemplazar signo menos unicode
    s = s.replace(" ", "")   # quitar espacios internos que rompen parseo

    # Regla: convertir '1,5' -> '1.5' solo si no hay puntos para evitar confundir miles
    if s.count(",") == 1 and s.count(".") == 0 and re.search(r"\d+,\d+$", s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        pass

    # Extraer primer patrón numérico de la cadena
    m = re.search(r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?", s)
    if m:
        numstr = m.group(0).replace(",", ".")
        try:
            return float(numstr)
        except Exception:
            return None
    return None


def convertir_serie_numerica_con_imputacion(serie: pd.Series, estrategia_relleno: str = "median") -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Intenta convertir una serie a numérica:
    - Usa intentar_parsear_numero para cada valor.
    - Si quedan NA, imputa según estrategia: 'median' (mediana) o 'zero' (0).
    - Retorna la serie numérica y una lista de cambios realizados (muestra para reporte).
    """
    cambios = []
    parseados = serie.apply(intentar_parsear_numero)

    # Registrar coerciones y fallos 
    for idx, (orig, p) in enumerate(zip(serie.fillna("").tolist(), parseados.tolist())):
        if pd.isna(p):
            if str(orig).strip() != "":
                cambios.append({"fila": idx, "original": orig, "nuevo": None, "razon": "no_parseable"})
        else:
            try:
                if str(orig).strip() != str(p):
                    cambios.append({"fila": idx, "original": orig, "nuevo": p, "razon": "coercion"})
            except Exception:
                cambios.append({"fila": idx, "original": orig, "nuevo": p, "razon": "coercion"})

    serie_num = pd.to_numeric(parseados, errors="coerce")

    # Seleccionar valor de relleno
    if serie_num.isna().all():
        valor_relleno = 0.0
    else:
        if estrategia_relleno == "median":
            valor_relleno = float(serie_num.median(skipna=True))
        else:
            valor_relleno = 0.0

    faltantes_antes = int(serie_num.isna().sum())
    if faltantes_antes > 0:
        serie_num = serie_num.fillna(valor_relleno)
        cambios.append({"valor_imputado": valor_relleno, "n_imputados": faltantes_antes})

    return serie_num, cambios

# Funciones para estandarizar IDs
def estandarizar_id_cliente(valor_id) -> str:
    """
    Genera IDs estandarizados para clientes: C###.
    Si no hay número extraíble genera hash corto.
    """
    try:
        n = int(valor_id)
        return f"C{n:03d}"
    except Exception:
        s = str(valor_id)
        dig = re.findall(r"\d+", s)
        if dig:
            return f"C{int(dig[0]):03d}"
        return "C" + (abs(hash(s)) % 1000).__format__("03d")


def estandarizar_id_vehiculo(valor_id) -> str:
    """
    Genera IDs estandarizados para vehículos: V###.
    """
    try:
        n = int(valor_id)
        return f"V{n:03d}"
    except Exception:
        s = str(valor_id)
        dig = re.findall(r"\d+", s)
        if dig:
            return f"V{int(dig[0]):03d}"
        return "V" + (abs(hash(s)) % 1000).__format__("03d")


def estandarizar_id_deposito(valor_id) -> str:
    """
    Genera IDs estandarizados para depósitos: CD##.
    """
    try:
        n = int(valor_id)
        return f"CD{n:02d}"
    except Exception:
        s = str(valor_id)
        dig = re.findall(r"\d+", s)
        if dig:
            return f"CD{int(dig[0]):02d}"
        return "CD" + (abs(hash(s)) % 100).__format__("02d")

# Limpieza y validación por tipo de archivo
def limpiar_validar_clientes(df: pd.DataFrame, ruta_archivo: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Limpia y valida un DataFrame de clientes:
    - Crea StandardizedID si falta.
    - Convierte Latitude/Longitude a numérico e intenta detectar intercambio lat/lon.
    - Convierte Demand a numérico y corrige negativos (los hace absolutos).
    - Normaliza TimeWindow (si existe).
    Devuelve un dict con información para el reporte y el DataFrame limpio.
    """
    info = {"archivo": str(ruta_archivo), "tipo": "clientes", "n_filas": len(df)}
    # Normalizar nombres de columnas (quitar espacios)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Crear StandardizedID si no existe
    if "StandardizedID" not in df.columns and "ClientID" in df.columns:
        df["StandardizedID"] = df["ClientID"].apply(estandarizar_id_cliente)
        info["standardized_ids_creados"] = True
    elif "StandardizedID" not in df.columns:
        df["StandardizedID"] = [estandarizar_id_cliente(i) for i in range(1, len(df) + 1)]
        info["standardized_ids_creados"] = True

    # Coordenadas: convertir y verificar rangos
    if "Latitude" in df.columns and "Longitude" in df.columns:
        lat, cambios_lat = convertir_serie_numerica_con_imputacion(df["Latitude"], estrategia_relleno="median")
        lon, cambios_lon = convertir_serie_numerica_con_imputacion(df["Longitude"], estrategia_relleno="median")
        df["Latitude"] = lat
        df["Longitude"] = lon

        # Detectar si hay muchos lat fuera de [-90,90] -> posible intercambio
        lat_fuera = int(((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum())
        lon_fuera = int(((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum())
        if lat_fuera > 0 and lon_fuera == 0:
            # intentar intercambio
            lat_temp = df["Latitude"].copy()
            lon_temp = df["Longitude"].copy()
            df["Latitude"], df["Longitude"] = lon_temp, lat_temp
            lat_fuera2 = int(((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum())
            lon_fuera2 = int(((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum())
            if lat_fuera2 <= lat_fuera:
                info["coords_intercambiadas"] = True
                logging.info(f"Intercambio lat/lon aplicado en {ruta_archivo}")
            else:
                # revertir
                df["Latitude"], df["Longitude"] = lat_temp, lon_temp

        #rangos válidos
        df["Latitude"] = df["Latitude"].clip(-90, 90)
        df["Longitude"] = df["Longitude"].clip(-180, 180)
    else:
        info.setdefault("avisos", []).append("Faltan columnas Latitude/Longitude")

    # convertir y arreglar negativos
    if "Demand" in df.columns:
        demand, cambios_demanda = convertir_serie_numerica_con_imputacion(df["Demand"], estrategia_relleno="median")
        df["Demand"] = demand
        n_neg = int((df["Demand"] < 0).sum())
        if n_neg > 0:
            df.loc[df["Demand"] < 0, "Demand"] = df.loc[df["Demand"] < 0, "Demand"].abs()
            info["demand_negativos_arreglados"] = n_neg
    else:
        info.setdefault("avisos", []).append("Falta columna 'Demand'")

    # TimeWindow: normalizar formato (si existe)
    if "TimeWindow" in df.columns:
        patron = re.compile(r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$")
        tw_fijados = 0
        for i, val in df["TimeWindow"].fillna("").items():
            if str(val).strip() == "":
                df.at[i, "TimeWindow"] = "00:00-23:59"
                tw_fijados += 1
            elif not patron.match(str(val)):
                df.at[i, "TimeWindow"] = "00:00-23:59"
                tw_fijados += 1
        if tw_fijados > 0:
            info["timewindows_normalizados"] = tw_fijados

    # Evitar duplicados en StandardizedID: hacerlos únicos si aparecen
    if "StandardizedID" in df.columns:
        if df["StandardizedID"].duplicated().any():
            vistos = {}
            for idx, val in df["StandardizedID"].items():
                if val in vistos:
                    vistos[val] += 1
                    df.at[idx, "StandardizedID"] = f"{val}_{vistos[val]}"
                else:
                    vistos[val] = 1

    info["issues"] = []  # tras limpieza no reportamos issues a nivel sintáctico
    info["columnas"] = list(df.columns)
    return info, df


def limpiar_validar_vehiculos(df: pd.DataFrame, ruta_archivo: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Limpia y valida un DataFrame de vehículos:
    - Crea StandardizedID si falta.
    - Convierte Capacity, Range, Speed a numérico e imputa/ajusta valores no positivos.
    """
    info = {"archivo": str(ruta_archivo), "tipo": "vehiculos", "n_filas": len(df)}
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    if "StandardizedID" not in df.columns and "VehicleID" in df.columns:
        df["StandardizedID"] = df["VehicleID"].apply(estandarizar_id_vehiculo)
        info["standardized_ids_creados"] = True
    elif "StandardizedID" not in df.columns:
        df["StandardizedID"] = [estandarizar_id_vehiculo(i) for i in range(1, len(df) + 1)]
        info["standardized_ids_creados"] = True

    for columna in ["Capacity", "Range", "Speed"]:
        if columna in df.columns:
            serie_num, cambios = convertir_serie_numerica_con_imputacion(df[columna], estrategia_relleno="median")
            df[columna] = serie_num
            # Reemplazar valores no positivos por la mediana si aparecen
            malos = int((df[columna] <= 0).sum())
            if malos > 0:
                med = float(df[columna].median())
                df.loc[df[columna] <= 0, columna] = med if med > 0 else 1.0
                info[f"{columna}_no_positivos_arreglados"] = malos

    # Hacer únicos los StandardizedID repetidos
    if "StandardizedID" in df.columns:
        if df["StandardizedID"].duplicated().any():
            vistos = {}
            for idx, val in df["StandardizedID"].items():
                if val in vistos:
                    vistos[val] += 1
                    df.at[idx, "StandardizedID"] = f"{val}_{vistos[val]}"
                else:
                    vistos[val] = 1

    info["issues"] = []
    info["columnas"] = list(df.columns)
    return info, df


def limpiar_validar_depositos(df: pd.DataFrame, ruta_archivo: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Limpia y valida un DataFrame de depósitos:
    - Crea StandardizedID si falta.
    - Asegura columnas Latitude/Longitude (intenta renombrar columnas alternativas).
    - Convierte y clampa coordenadas.
    """
    info = {"archivo": str(ruta_archivo), "tipo": "depositos", "n_filas": len(df)}
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    if "StandardizedID" not in df.columns and "DepotID" in df.columns:
        df["StandardizedID"] = df["DepotID"].apply(estandarizar_id_deposito)
        info["standardized_ids_creados"] = True
    elif "StandardizedID" not in df.columns:
        df["StandardizedID"] = [estandarizar_id_deposito(i) for i in range(1, len(df) + 1)]
        info["standardized_ids_creados"] = True

    if "Latitude" in df.columns and "Longitude" in df.columns:
        lat, cambios_lat = convertir_serie_numerica_con_imputacion(df["Latitude"], estrategia_relleno="median")
        lon, cambios_lon = convertir_serie_numerica_con_imputacion(df["Longitude"], estrategia_relleno="median")
        df["Latitude"] = lat
        df["Longitude"] = lon

        lat_malos = int(((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum())
        lon_malos = int(((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum())
        if lat_malos > 0 and lon_malos == 0:
            # intentar intercambio
            df["Latitude"], df["Longitude"] = df["Longitude"].copy(), df["Latitude"].copy()
            info["coords_intercambiadas"] = True

        df["Latitude"] = df["Latitude"].clip(-90, 90)
        df["Longitude"] = df["Longitude"].clip(-180, 180)
    else:
        # intentar detectar nombres alternativos (lon, lat, latitud, longitud)
        cols_lower = [c.lower() for c in df.columns]
        if "longitude" in cols_lower and "latitude" not in cols_lower:
            for alt_lon in ["lon", "long", "longitud"]:
                if alt_lon in cols_lower:
                    idx = cols_lower.index(alt_lon)
                    df = df.rename(columns={df.columns[idx]: "Longitude"})
        if "latitude" in cols_lower and "longitude" not in cols_lower:
            for alt_lat in ["lat", "latitud"]:
                if alt_lat in cols_lower:
                    idx = cols_lower.index(alt_lat)
                    df = df.rename(columns={df.columns[idx]: "Latitude"})
        # si aún faltan, crear columnas con 0.0 
        if "Latitude" not in df.columns:
            df["Latitude"] = 0.0
            info["latitude_creada"] = True
        if "Longitude" not in df.columns:
            df["Longitude"] = 0.0
            info["longitude_creada"] = True

    info["issues"] = []
    info["columnas"] = list(df.columns)
    return info, df


def limpiar_validar_parametros(df: pd.DataFrame, ruta_archivo: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Convierte la columna 'Value' a numérico, intenta extraer números desde textos,
    e imputa faltantes con la mediana para dejar el archivo sin problemas numéricos.
    """
    info = {"archivo": str(ruta_archivo), "tipo": "parametros", "n_filas": len(df)}
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if "Value" not in df.columns:
        info.setdefault("avisos", []).append("Falta columna 'Value'")
        info["issues"] = info.get("avisos", [])
        info["columnas"] = list(df.columns)
        return info, df

    serie_num, cambios = convertir_serie_numerica_con_imputacion(df["Value"], estrategia_relleno="median")
    df["Value"] = serie_num
    info["muestra_cambios"] = cambios[:10]
    info["issues"] = []
    info["columnas"] = list(df.columns)
    return info, df


def limpiar_validar_generico(df: pd.DataFrame, ruta_archivo: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Validador genérico: elimina filas totalmente vacías y quita duplicados exactos.
    """
    info = {"archivo": str(ruta_archivo), "tipo": "tabla", "n_filas": len(df)}
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    if df.duplicated().any():
        df = df.drop_duplicates().reset_index(drop=True)
    info["issues"] = []
    info["columnas"] = list(df.columns)
    return info, df

#estructura de datos almacenamiento_datos

def insertar_en_estructura_anidada(base: Dict[str, Any], partes: List[str], valor: Any) -> None:
    nodo = base
    for p in partes[:-1]:
        if p not in nodo or not isinstance(nodo[p], dict):
            nodo[p] = {}
        nodo = nodo[p]
    nodo[partes[-1]] = valor

# Procesamiento principal por carpeta de caso
def procesar_carpeta_caso(carpeta_caso: Path, lista_reportes: List[Dict], almacenamiento_datos: Dict[str, Dict[str, Any]]) -> None:
    """
    Procesa todos los archivos en una carpeta de caso dada:
    - Carga cada archivo
    - Limpia/valida según tipo
    - Guarda copia limpia
    - Añade información para reporte y almacena DataFrames en almacenamiento_datos
    """
    archivos = encontrar_archivos_datos(carpeta_caso)
    logging.info(f"Encontrados {len(archivos)} archivos en {carpeta_caso}")
    carpeta_limpia = Path("cleaned_data") / carpeta_caso.name
    carpeta_limpia.mkdir(parents=True, exist_ok=True)

    clave_caso = carpeta_caso.name
    # inicializar contenedor para este caso                     
    if clave_caso not in almacenamiento_datos:
        almacenamiento_datos[clave_caso] = {}

    for arc in archivos:
        try:
            df, fmt = cargar_tabla(arc)
        except Exception as e:
            logging.error(f"No se pudo leer {arc}: {e}")
            lista_reportes.append({"archivo": str(arc), "error": str(e)})
            continue

        # Normalizar nombres columnas
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        nombre_low = arc.name.lower()

        # --- limpieza/validación según tipo ---
        if "client" in nombre_low:
            info, df_limpio = limpiar_validar_clientes(df, arc)
        elif "vehicle" in nombre_low:
            info, df_limpio = limpiar_validar_vehiculos(df, arc)
        elif "depot" in nombre_low or "depots" in nombre_low:
            info, df_limpio = limpiar_validar_depositos(df, arc)
        elif "parameter" in nombre_low or nombre_low.startswith("parameters"):
            info, df_limpio = limpiar_validar_parametros(df, arc)
        else:
            info, df_limpio = limpiar_validar_generico(df, arc)

        lista_reportes.append(info)


        # Guardar copia limpia 
        try:
           
            ruta_relativa = arc.relative_to(carpeta_caso)  
            ruta_salida = carpeta_limpia / ruta_relativa
            ruta_salida.parent.mkdir(parents=True, exist_ok=True)
            # guardar como CSV 
            df_limpio.to_csv(ruta_salida.with_suffix(".csv"), index=False)
        except Exception as e:
            logging.warning(f"Fallo guardando copia limpia de {arc}: {e}")


        # Insertar DataFrame en almacenamiento_datos 
        try:
            partes = list(ruta_relativa.parts)  
            insertar_en_estructura_anidada(almacenamiento_datos[clave_caso], partes, df_limpio)
        except Exception as e:
            logging.warning(f"No se pudo insertar {arc} en almacenamiento_datos: {e}")



# Guardado de reportes
def guardar_reportes(reportes: List[Dict], carpeta_salida: Path = Path("reports")) -> None:
    """
    Escribe el JSON completo y un CSV resumen fácil de leer.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta_salida / "integrity_report.json"
    ruta_csv = carpeta_salida / "integrity_report_summary.csv"
    with ruta_json.open("w", encoding="utf8") as fh:
        json.dump(reportes, fh, indent=2, ensure_ascii=False)
    filas = []
    for r in reportes:
        if isinstance(r, dict):
            filas.append({
                "archivo": r.get("archivo", r.get("file", "")),
                "tipo": r.get("tipo", r.get("type", "")),
                "n_filas": r.get("n_filas", r.get("n_rows", "")),
                "n_columnas": len(r.get("columnas", [])) if r.get("columnas") else "",
                "issues": " | ".join(r.get("issues", [])) if r.get("issues") else "[]"
            })
    df_resumen = pd.DataFrame(filas)
    df_resumen.to_csv(ruta_csv, index=False)
    logging.info(f"Reportes guardados en {ruta_json} y {ruta_csv}")

# Función principal                  
def ejecutar_principal(ruta_raiz: str = "."):
    """
    Punto de entrada principal:
    - Localiza carpetas de caso en la ruta dada
    - Procesa cada carpeta
    - Guarda reportes y una copia pickled del almacenamiento_datos
    - Retorna almacenamiento_datos para uso programático
    """
    raiz = Path(ruta_raiz).resolve()
    if not raiz.exists():
        logging.error(f"Ruta no encontrada: {raiz}")
        return

    carpetas = encontrar_carpetas_caso(raiz)
    if not carpetas:
        logging.warning("No se detectaron carpetas 'Proyecto_Caso_Base' ni 'project_b' en la ruta. Buscando subcarpetas...")
        carpetas = [p for p in raiz.rglob("*") if p.is_dir() and re.search(
            r"proyecto[_\- ]?caso[_\- ]?base|project[_\- ]?b|proyecto[_\- ]?b", p.name, re.IGNORECASE)]
        carpetas = list(dict.fromkeys(carpetas))
    if not carpetas:
        logging.error("No se encontraron carpetas de casos. Asegúrate de que las carpetas estén en la ruta proporcionada.")
        return

    reportes = []
    # Almacenamiento en memoria
    almacenamiento_datos: Dict[str, Dict[str, Any]] = {}

    for c in carpetas:
        procesar_carpeta_caso(c, reportes, almacenamiento_datos)

    # Guardar reportes en disco
    guardar_reportes(reportes)

    # Guardar almacenamiento_datos como pickle para uso posterior (local)
    try:
        import pickle
        Path("cleaned_data").mkdir(parents=True, exist_ok=True)
        with open("cleaned_data/almacenamiento_datos.pkl", "wb") as fh:
            pickle.dump(almacenamiento_datos, fh)
        logging.info("Estructura 'almacenamiento_datos' guardada en cleaned_data/almacenamiento_datos.pkl")
    except Exception as e:
        logging.warning(f"No se pudo guardar almacenamiento_datos: {e}")

    logging.info("Proceso terminado.")
    return almacenamiento_datos


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "."
    ejecutar_principal(ruta)
