
import io
import json
import os
import math
from datetime import datetime, timedelta, time
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Planeación de Laboratorio", page_icon="🧪", layout="wide")

DAY_START = time(8, 0)
DAY_END = time(18, 0)

def normalize_yes_no(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def load_data(default_path: str, uploaded: bytes | None):
    if uploaded is not None:
        xls = pd.ExcelFile(uploaded)
    else:
        xls = pd.ExcelFile(default_path)

    # Leer hoja 'Foliar' si existe; si no, crear DataFrame vacío con columnas esperadas
    foliar = None
    try:
        if "Foliar" in xls.sheet_names:
            foliar = pd.read_excel(xls, sheet_name="Foliar")
        else:
            # Crear DF vacío con columnas mínimas para evitar errores posteriores
            foliar = pd.DataFrame(columns=["Aplican", "Cumple", "Entrega dias", "Fecha solicitud", "Fecha de entrega"]) 
    except Exception:
        # En caso de cualquier error, devolver DF vacío similar
        foliar = pd.DataFrame(columns=["Aplican", "Cumple", "Entrega dias", "Fecha solicitud", "Fecha de entrega"]) 
    # Prueba (aceptar nombre alternativo 'digestor' si se renombró la hoja)
    prueba = None
    try:
        candidates = ["Prueba", "prueba", "digestor", "Digestor"]
        for c in candidates:
            if c in xls.sheet_names:
                prueba = pd.read_excel(xls, sheet_name=c)
                break
        # Si no se encontró, intentar coincidencias parciales (por si el usuario cambió el nombre ligeramente)
        if prueba is None:
            for s in xls.sheet_names:
                lname = s.lower()
                if "pru" in lname or "dige" in lname:
                    prueba = pd.read_excel(xls, sheet_name=s)
                    break
        # Si aún no hay 'prueba', crear DataFrame vacío con columnas esperadas
        if prueba is None:
            prueba = pd.DataFrame(columns=["Registro", "Tipo de analisis", "No muestras", "Fecha solicitud", "Fecha de entrega"])
    except Exception:
        prueba = pd.DataFrame(columns=["Registro", "Tipo de analisis", "No muestras", "Fecha solicitud", "Fecha de entrega"])

    # Tiempo: si la hoja no existe, crear DataFrame vacío con columnas esperadas
    try:
        if "Tiempo" in xls.sheet_names:
            tiempo = pd.read_excel(xls, sheet_name="Tiempo")
        else:
            tiempo = pd.DataFrame(columns=["Grupo", "Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"])
    except Exception:
        tiempo = pd.DataFrame(columns=["Grupo", "Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"])

    # Capacidad: si falta, usare un DataFrame vacío y la función load_data aplicará valor por defecto
    try:
        if "Capacidad" in xls.sheet_names:
            capacidad = pd.read_excel(xls, sheet_name="Capacidad")
        else:
            capacidad = pd.DataFrame()
    except Exception:
        capacidad = pd.DataFrame()

    foliar.columns = [c.strip() for c in foliar.columns]
    prueba.columns = [c.strip() for c in prueba.columns]
    tiempo.columns = [c.strip() for c in tiempo.columns]
    capacidad.columns = [c.strip() for c in capacidad.columns]

    for c in ["Fecha solicitud", "Fecha de entrega"]:
        if c in foliar.columns:
            try:
                foliar[c] = pd.to_datetime(foliar[c], errors="coerce")
            except Exception:
                foliar[c] = pd.Series([], dtype='datetime64[ns]')
        if c in prueba.columns:
            prueba[c] = pd.to_datetime(prueba[c], errors="coerce")

    if "Tipo de analisis" in prueba.columns:
        prueba["Tipo de analisis"] = prueba["Tipo de analisis"].astype(str).str.strip()
    if "Grupo" in tiempo.columns:
        tiempo["Grupo"] = tiempo["Grupo"].astype(str).str.strip()
    if "Aplican" in foliar.columns:
        foliar["Aplican"] = normalize_yes_no(foliar["Aplican"])
    if "Cumple" in foliar.columns:
        foliar["Cumple"] = normalize_yes_no(foliar["Cumple"])

    # Capacidad DIARIA (muestras/día) - Por defecto 38
    daily_cap = 38  # Valor por defecto
    cap_col = [c for c in capacidad.columns if "Capacidad" in c]
    if cap_col and not capacidad[cap_col[0]].dropna().empty:
        try:
            val = float(pd.to_numeric(capacidad[cap_col[0]], errors="coerce").dropna().iloc[0])
            if val > 0:
                daily_cap = int(val)
        except Exception:
            daily_cap = 38  # Mantener valor por defecto en caso de error

    return foliar, prueba, tiempo, daily_cap

def expand_analyses(prueba: pd.DataFrame) -> pd.DataFrame:
    df = prueba.copy()
    df["Tipo de analisis"] = df["Tipo de analisis"].astype(str).str.replace(" ", "")
    rows = []
    for _, r in df.iterrows():
        ty = r["Tipo de analisis"]
        grupos = [g for g in ty.split(",") if g] if ("," in ty) else [ty]
        for g in grupos:
            rr = r.copy()
            rr["Tipo de analisis"] = g
            rr["ID analisis"] = f"{r['Registro']}-{g}"
            rows.append(rr)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def split_large_records(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Divide filas con 'No muestras' mayores a `cap` en fragmentos de tamaño `cap`.
    - Cada fragmento mantiene los mismos metadatos (Registro, Fecha solicitud, Tipo de analisis, etc.).
    - Se genera un nuevo `ID analisis` con sufijos `-p1`, `-p2`, ... para los fragmentos.
    - El fragmento final puede quedar con menos de `cap` y será tratado como registro pequeño
      (permitiendo que el scheduler lo combine con otros ≤cap).
    """
    if df is None or df.empty:
        return df.copy()

    rows = []
    for _, r in df.reset_index(drop=True).iterrows():
        try:
            nm = int(r.get("No muestras", 0) or 0)
        except Exception:
            nm = 0

        base = r.to_dict()
        base_id = str(base.get("ID analisis", base.get("Registro", "")))

        if nm <= cap or nm <= 0:
            # No es necesario fragmentar
            rows.append(base)
            continue

        # Cuantos fragmentos completos de tamaño `cap`
        full_parts = nm // cap
        remainder = nm % cap
        seq = 1

        for i in range(full_parts):
            new = base.copy()
            new["No muestras"] = cap
            new["ID analisis"] = f"{base_id}-p{seq}"
            rows.append(new)
            seq += 1

        if remainder > 0:
            new = base.copy()
            new["No muestras"] = remainder
            new["ID analisis"] = f"{base_id}-p{seq}"
            rows.append(new)

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


# -------------------------
# Persistent configuration
# -------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def read_config() -> dict:
    """Leer configuración persistente si existe (capacidad y tiempos por grupo)."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # Valores por defecto
    return {
        "daily_cap": 38,
        "times": {
            "A": {"pre": 2.0, "proc": 6.0},
            "B": {"pre": 2.0, "proc": 6.0},
            "C": {"pre": 2.0, "proc": 6.0},
            "D": {"pre": 2.0, "proc": 6.0}
        }
    }

def write_config(conf: dict) -> bool:
    """Guardar configuración persistente en `config.json`. Devuelve True si tuvo éxito."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# NOTE: La función de cálculo de tasas de llegada históricas fue eliminada por petición del usuario.
# Si se requiere más adelante, puede reimplementarse con la función `expand_analyses` como base.

# NOTE: La función de análisis de capacidad vs tasas históricas fue eliminada por petición del usuario.
    # NOTE: La función de análisis de capacidad vs tasas históricas fue eliminada por petición del usuario.

def compute_kpis(foliar: pd.DataFrame) -> Dict[str, float]:
    subset = foliar.copy()
    if "Aplican" in subset.columns:
        subset = subset[subset["Aplican"] == "SI"]
    total = len(subset)
    kpis = {}
    if total and "Cumple" in subset.columns:
        cumple_si = (subset["Cumple"] == "SI").sum()
        kpis["% Cumplimiento (solo Aplican=SI)"] = (cumple_si / total) * 100
    else:
        kpis["% Cumplimiento (solo Aplican=SI)"] = 0.0
    if "Entrega dias" in subset.columns and total:
        kpis["Tiempo promedio de entrega (días)"] = subset["Entrega dias"].dropna().astype(float).mean()
    else:
        kpis["Tiempo promedio de entrega (días)"] = 0.0
    kpis["Casos considerados (Aplican=SI)"] = float(total)
    return kpis

def get_week_days(selected_date: datetime):
    monday = selected_date - timedelta(days=selected_date.weekday())
    return [datetime.combine((monday + timedelta(days=i)).date(), DAY_START) for i in range(4)]

def consolidate_blocks(df_proc: pd.DataFrame) -> pd.DataFrame:
    """Une bloques contiguos (Fin == Inicio siguiente) con mismo Registro+Grupo (misma fecha)."""
    if df_proc.empty:
        return df_proc.copy()
    dfp = df_proc.sort_values(["Fecha","Registro","Grupo","Inicio"]).reset_index(drop=True)
    out_rows = []
    cur = None
    for _, r in dfp.iterrows():
        if cur is None:
            cur = r.to_dict()
            continue
        if (r["Registro"] == cur["Registro"] and r["Grupo"] == cur["Grupo"]
            and r["Fecha"] == cur["Fecha"] and r["Inicio"] == cur["Fin"]):
            cur["Fin"] = r["Fin"]
            cur["Muestras"] += r["Muestras"]
        else:
            out_rows.append(cur)
            cur = r.to_dict()
    if cur is not None:
        out_rows.append(cur)
    return pd.DataFrame(out_rows)

def gantt_user_format_from_blocks(blocks: pd.DataFrame, totals_by_reg: pd.DataFrame, accum: Dict[str,int]):
    """Devuelve df(Tarea,Inicio,Fin,Progreso) + acumulador de progreso por Registro."""
    rows = []
    if blocks.empty:
        return pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso"]), accum
    blocks = blocks.sort_values("Inicio")
    for _, r in blocks.iterrows():
        reg = str(r["Registro"])
        total_req = int(totals_by_reg.loc[totals_by_reg["Registro"] == reg, "TotalRegistro"].iloc[0]) if (totals_by_reg["Registro"] == reg).any() else 0
        accum[reg] = accum.get(reg, 0) + int(r["Muestras"])
        progreso = int(round(100 * min(accum[reg], total_req) / total_req)) if total_req > 0 else 0
        rows.append({"Tarea": reg, "Inicio": r["Inicio"], "Fin": r["Fin"], "Progreso": progreso})
    return pd.DataFrame(rows), accum

def plan_week_by_day(prueba: pd.DataFrame, tiempo: pd.DataFrame, selected_date: datetime, daily_cap: int):
    """Planifica L–V con **entrega completa de registros**:
       - Un registro se entrega SOLO cuando se completan TODOS sus grupos (A, B, C, D)
       - No se pueden dejar registros a medias de procesamiento
       - Prioriza FIFO con urgencia 20 días
       - LUNES: Solo grupos A y D (D no se combina con otros)
       - MARTES: Solo grupos B y C (sí se pueden combinar)
       - MIÉRCOLES y JUEVES: Todos los grupos disponibles
       - Grupo D: NUNCA se combina con otros grupos
       - Optimiza capacidad diaria sin fragmentar registros ≤38 muestras
       - Registros >38 se fragmentan inteligentemente evaluando: antigüedad, porcentaje procesable, resto significativo
       - Trata de cumplir umbral mínimo del 75% de capacidad (29 muestras/día)
    """
    if daily_cap is None or daily_cap <= 0:
        daily_cap = 38  # Valor por defecto si hay algún problema
        st.warning(f"No se encontró capacidad diaria válida en la hoja 'Capacidad'. Usando valor por defecto: {daily_cap}")

    st.info(f"📊 **Capacidad diaria configurada: {daily_cap} muestras/día**")

    # Nota: eliminación de análisis histórico/optimización — la planificación se ejecuta sin cálculos basados en historiales.

    # Map de tiempos desde Excel
    tiempo = tiempo.copy()
    tiempo.columns = [c.strip() for c in tiempo.columns]
    tiempo["Grupo"] = tiempo["Grupo"].astype(str).str.strip()
    t_map = tiempo.set_index("Grupo")[["Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"]].to_dict("index")

    # Pedidos expandidos (datos originales, sin fragmentar) - usados para totales
    df_expanded = expand_analyses(prueba)
    df_expanded = df_expanded.sort_values(by=["Fecha solicitud", "Registro"]).reset_index(drop=True)
    df_expanded["Tipo de analisis"] = df_expanded["Tipo de analisis"].astype(str).str.strip()
    df_expanded["Registro"] = df_expanded["Registro"].astype(str)

    # División proactiva SOLO para el estado de procesamiento: fragmentar registros con más muestras que la capacidad diaria
    df_original = split_large_records(df_expanded, daily_cap)
    df_original = df_original.sort_values(by=["Fecha solicitud", "Registro"]).reset_index(drop=True)
    df_original["Pendiente"] = df_original["No muestras"].fillna(0).astype(int)
    df_original["Tipo de analisis"] = df_original["Tipo de analisis"].astype(str).str.strip()
    df_original["Registro"] = df_original["Registro"].astype(str)

    # NUEVO: Análisis de registros completos para entrega
    # Agrupar por registro para verificar qué grupos tiene cada uno
    registros_info = df_original.groupby("Registro").agg({
        "Tipo de analisis": lambda x: set(x),
        "Fecha solicitud": "first",
        "No muestras": "first",
        "Pendiente": "sum"
    }).reset_index()
    
    registros_info["Grupos_requeridos"] = registros_info["Tipo de analisis"].apply(lambda x: sorted(list(x)))
    registros_info["Total_grupos"] = registros_info["Grupos_requeridos"].apply(len)

    # Totales por Registro para % progreso acumulado (usar datos originales sin fragmentar)
    totals_by_reg = df_expanded.groupby("Registro", as_index=False)["No muestras"].sum().rename(columns={"No muestras":"TotalRegistro"})

    # Días de la semana
    days = get_week_days(selected_date)

    schedule_rows = []
    daily_utilization = []
    gantt_per_day = {}
    accum_progress: Dict[str,int] = {}
    
    # Estado global de muestras pendientes (se mantiene entre días)
    df_state = df_original.copy()
    # Capacidad/tasas históricas fueron removidas; usar dict vacío como fallback
    capacity_analysis: Dict[str, Dict] = {}
    
    def verificar_registro_completo(registro: str, df_estado: pd.DataFrame) -> tuple:
        """Verifica si un registro tiene todos sus grupos completados para entrega"""
        reg_data = df_estado[df_estado["Registro"] == registro]
        if reg_data.empty:
            return False, []
        
        grupos_pendientes = reg_data[reg_data["Pendiente"] > 0]["Tipo de analisis"].tolist()
        grupos_completados = reg_data[reg_data["Pendiente"] == 0]["Tipo de analisis"].tolist()
        
        esta_completo = len(grupos_pendientes) == 0
        return esta_completo, grupos_pendientes
    
    def obtener_registros_entregables(df_estado: pd.DataFrame) -> list:
        """Obtiene lista de registros listos para entrega (todos los grupos completados)"""
        registros_unicos = df_estado["Registro"].unique()
        entregables = []
        
        for registro in registros_unicos:
            completo, pendientes = verificar_registro_completo(registro, df_estado)
            if completo:
                # Calcular total de muestras del registro
                total_muestras = df_estado[df_estado["Registro"] == registro]["No muestras"].iloc[0]
                entregables.append((registro, total_muestras))
        
        return entregables
    
    def get_available_groups_sorted(df_temp: pd.DataFrame, current_day: datetime) -> List[Tuple[str, float, int]]:
        """Retorna grupos disponibles ordenados por prioridad FIFO + urgencia 20 días + tasas de llegada"""
        candidates = df_temp[df_temp["Pendiente"] > 0].copy()
        if candidates.empty:
            return []
        
        # Calcular prioridad por grupo usando FIFO + urgencia + tasas de llegada
        group_priority = []
        for grupo in candidates["Tipo de analisis"].unique():
            group_data = candidates[candidates["Tipo de analisis"] == grupo]
            
            # Calcular días de antigüedad (FIFO)
            try:
                fechas_solicitud = group_data["Fecha solicitud"].apply(
                    lambda x: pd.to_datetime(x) if pd.notna(x) else current_day
                )
                # Usar la fecha MÁS ANTIGUA del grupo (FIFO estricto)
                fecha_mas_antigua = fechas_solicitud.min()
                dias_antiguedad = (current_day.date() - fecha_mas_antigua.date()).days
            except:
                dias_antiguedad = 0
                
            muestras_grupo = group_data["Pendiente"].sum()
            
            # NUEVO: Criterio directo basado en tasas de llegada
            tasa_factor = 1.0  # Factor neutro por defecto
            tasa_llegada_diaria = 0.0
            capacidad_grupo = daily_cap / 4  # Capacidad equitativa por grupo (38/4 = 9.5)
            
            if capacity_analysis and grupo in capacity_analysis:
                grupo_analysis = capacity_analysis[grupo]
                tasa_llegada_diaria = grupo_analysis.get('tasa_llegada', 0.0)
                saturacion = grupo_analysis.get('saturacion', 1.0)
                
                # CRITERIO 1: Ajuste por tasa de llegada vs capacidad base
                ratio_tasa = tasa_llegada_diaria / capacidad_grupo if capacidad_grupo > 0 else 0
                
                if ratio_tasa > 1.5:  # Tasa muy alta vs capacidad base
                    tasa_factor = 2.5  # Prioridad muy alta
                elif ratio_tasa > 1.0:  # Tasa alta vs capacidad base  
                    tasa_factor = 1.8  # Prioridad alta
                elif ratio_tasa > 0.7:  # Tasa moderada
                    tasa_factor = 1.3  # Prioridad moderada
                elif ratio_tasa > 0.3:  # Tasa baja
                    tasa_factor = 1.0  # Prioridad normal
                else:  # Tasa muy baja
                    tasa_factor = 0.7  # Prioridad reducida
                
                # CRITERIO 2: Ajuste adicional por saturación del sistema
                if saturacion > 1.2:  # Sistema saturado para este grupo
                    tasa_factor *= 1.5  # Multiplicador adicional
                elif saturacion > 0.8:  # Sistema con alta carga
                    tasa_factor *= 1.2  # Multiplicador moderado
            
            # PRIORIDAD COMPUESTA: FIFO + Urgencia 20 días + Tasa de Llegada
            base_priority = 0
            if dias_antiguedad >= 20:
                # URGENTE: Solicitudes ≥20 días tienen máxima prioridad
                base_priority = 1000 + dias_antiguedad  # Base alta + días adicionales
                urgencia_msg = "🚨 URGENTE"
            else:
                # NORMAL: FIFO + Criterio de tasa de llegada
                base_priority = dias_antiguedad + (tasa_llegada_diaria * 10)  # Boost por tasa
                urgencia_msg = "📅 Normal"
            
            # Aplicar factor de tasa de llegada
            priority_score = base_priority * tasa_factor
            
            # Determinar mensaje de prioridad combinada
            if tasa_factor >= 2.0:
                urgencia_msg += " + 🔥 ALTA DEMANDA"
            elif tasa_factor >= 1.5:
                urgencia_msg += " + ⚡ DEMANDA ELEVADA"
            elif tasa_factor >= 1.2:
                urgencia_msg += " + ⚠️ DEMANDA MEDIA"
            
            if grupo in t_map:  # Solo considerar grupos con tiempos definidos
                group_priority.append((grupo, priority_score, int(muestras_grupo), dias_antiguedad, urgencia_msg))
        
        # Ordenar por prioridad descendente (más urgente/antiguo/saturado primero)
        sorted_groups = sorted(group_priority, key=lambda x: x[1], reverse=True)
        
        # Retornar formato esperado (grupo, score, muestras)
        return [(g, s, m) for g, s, m, _, _ in sorted_groups]

    # INICIO DEL PROCESAMIENTO SEMANAL
    for day_idx, d in enumerate(days):
        day_name = ["Lunes","Martes","Miércoles","Jueves"][day_idx]
        pendientes_inicio = (df_state["Pendiente"] > 0).sum()
        muestras_pendientes_total = df_state["Pendiente"].sum()
        
        day_start = d
        day_end = datetime.combine(d.date(), DAY_END)
        day_cursor = day_start
        used_seconds = 0.0
        remaining_daily = int(daily_cap)
        day_samples_processed = 0

        # BUCLE PRINCIPAL: Maximizar uso de máquina con fallback automático
        session_attempted = False
        while day_cursor < day_end and (df_state["Pendiente"] > 0).any() and remaining_daily > 0:
            hours_left = (day_end - day_cursor).total_seconds() / 3600.0
            
            # Obtener grupos disponibles ordenados por prioridad
            available_groups = get_available_groups_sorted(df_state, d)
            if not available_groups:
                break
            
            # ESTRATEGIA DE FALLBACK: Probar grupos en orden de prioridad
            grupo_procesado = None
            fallback_attempts = []
            
            # NUEVAS RESTRICCIONES POR DÍA:
            # LUNES: Solo A y D (D no se combina)
            # MARTES: Solo B y C (sí se combinan)
            # MIÉRCOLES y JUEVES: Todos los grupos
            if day_name == "Lunes":
                # En lunes, SOLO grupos A y D permitidos
                available_groups_original = available_groups.copy()
                available_groups = [(g, s, m) for g, s, m in available_groups if g in ["A", "D"]]
                
                if not available_groups:
                    continue
                    
            elif day_name == "Martes":
                # En martes, SOLO grupos B y C permitidos (sí se combinan)
                available_groups_original = available_groups.copy()
                available_groups = [(g, s, m) for g, s, m in available_groups if g in ["B", "C"]]
                
                if not available_groups:
                    continue
                    
                # Reorganizar para procesar B y C juntos en martes
                bc_groups = [(g, s, m) for g, s, m in available_groups if g in ["B", "C"]]
                available_groups = bc_groups
            
            # MIÉRCOLES y JUEVES: Todos los grupos disponibles
            elif day_name in ["Miércoles", "Jueves"]:
                pass  # Usar todos los grupos disponibles
            
            for grupo, priority_score, muestras_disponibles in available_groups:
                tiempos = t_map.get(grupo, {})
                t_proc = float(tiempos.get("Tiempo procesamiento (horas)", 0) or 0.0)
                
                total_time_needed = t_proc  # Solo tiempo de procesamiento
                
                fallback_attempts.append(f"{grupo}({total_time_needed:.1f}h)")
                
                # VALIDACIÓN: ¿Cabe en el tiempo disponible?
                if hours_left >= total_time_needed and t_proc > 0:
                    grupo_procesado = grupo
                    break
                elif t_proc <= 0:
                    continue
            
            # Si no se encontró ningún grupo que quepa, terminar el día
            if grupo_procesado is None:
                break
            
            # PROCESAMIENTO DEL GRUPO SELECCIONADO (SIN PREALISTAMIENTO)
            grupo = grupo_procesado
            tiempos = t_map[grupo]
            t_proc = float(tiempos.get("Tiempo procesamiento (horas)", 0) or 0.0)

            # Seleccionar registros del grupo por FIFO estricto + urgencia
            grp_df = df_state[(df_state["Pendiente"] > 0) & (df_state["Tipo de analisis"] == grupo)].copy()
            if grp_df.empty:
                continue
            
            # FIFO ESTRICTO: Ordenar por fecha de solicitud (más antiguas primero), luego por registro
            grp_df["Fecha solicitud"] = pd.to_datetime(grp_df["Fecha solicitud"], errors='coerce')
            grp_df = grp_df.sort_values(by=["Fecha solicitud", "Registro"])
            
            # Marcar solicitudes urgentes (≥20 días)
            grp_df["Dias_antiguedad"] = (pd.Timestamp(d.date()) - grp_df["Fecha solicitud"]).dt.days
            urgentes = grp_df[grp_df["Dias_antiguedad"] >= 20]
            pend_grp_total = int(grp_df["Pendiente"].sum())
            
            # REGLA ANTI-FRAGMENTACIÓN OPTIMIZADA: Respetar reglas estrictas
            session_take = 0
            registros_a_procesar = []
            
            # RESTRICCIÓN ESPECIAL GRUPO D: No se combina con otros
            if grupo == "D":
                # Para grupo D, usar toda la capacidad disponible sin combinar
                grp_df_d = grp_df.copy()
                
                # Solo procesar registros de grupo D, uno a la vez
                for _, row in grp_df_d.iterrows():
                    pend_registro = int(row["Pendiente"])
                    if pend_registro <= 0:
                        continue
                        
                    # Tomar hasta la capacidad disponible
                    take_d = min(pend_registro, remaining_daily - session_take)
                    if take_d > 0:
                        session_take += take_d
                        registros_a_procesar.append((row.name, take_d, "D"))
                        
                        # Para grupo D, procesar solo un registro por sesión
                        break
            else:
                # Lógica normal para grupos A, B, C (pueden combinarse)
                # Separar registros por tamaño ANTES de procesamiento
                # Para los grupos B y C permitimos combinar registros de ambos grupos (aunque sean de distinto Registro)
                combinable = False
                if grupo in ["B", "C"]:
                    combinable = True

                if combinable:
                    pool_df = df_state[(df_state["Pendiente"] > 0) & (df_state["Tipo de analisis"].isin(["B", "C"]))].copy()
                else:
                    pool_df = grp_df.copy()

                registros_pequenos = []  # ≤38 muestras - NO se pueden fragmentar
                registros_grandes = []   # >38 muestras - SE pueden fragmentar con decisión inteligente

                # Clasificar registros por tamaño (manteniendo el grupo de cada fila)
                for _, row in pool_df.iterrows():
                    pend_registro = int(row["Pendiente"])
                    if pend_registro <= 0:
                        continue
                    if pend_registro <= 38:
                        registros_pequenos.append((row.name, pend_registro, row["Tipo de analisis"]))
                    else:
                        registros_grandes.append((row.name, pend_registro, row["Tipo de analisis"]))

                # PASO 1: Procesar registros pequeños completos (≤38) - Sin fragmentación
                # Ordenar por tamaño (más grandes primero) para mejor llenado
                registros_pequenos.sort(key=lambda x: x[1], reverse=True)

                for idx, muestras, grp_of in registros_pequenos:
                    if session_take + muestras <= remaining_daily:
                        session_take += muestras
                        registros_a_procesar.append((idx, muestras, grp_of))
                    # Si no cabe completo, se deja para otro día (no fragmentar)

                # VERIFICACIÓN DE UMBRAL MÍNIMO (75% de capacidad)
                umbral_minimo = int(daily_cap * 0.75)
                necesita_mas_muestras = session_take < umbral_minimo

                # PASO 2: Si queda espacio O no se alcanzó el umbral, procesar registros grandes
                espacio_restante = remaining_daily - session_take
                if (espacio_restante > 0 or necesita_mas_muestras) and registros_grandes:
                    mejor_opcion = None

                    for idx, muestras, grp_of in registros_grandes:
                        reg_data = df_state.iloc[idx]

                        # OPCIÓN 1: ¿Cabe completo?
                        if muestras <= espacio_restante:
                            mejor_opcion = (idx, muestras, grp_of, "completo")
                            break  # Completo es siempre mejor

                        # OPCIÓN 2: Evaluación inteligente para fragmentar
                        grupos_completos = espacio_restante // 38
                        if grupos_completos > 0:
                            fragmento = min(grupos_completos * 38, muestras)

                            dias_antiguedad = (datetime.now() - pd.to_datetime(reg_data["Fecha solicitud"])) .days
                            porcentaje_procesable = fragmento / muestras
                            resto_significativo = (muestras - fragmento) >= 38
                            ayuda_umbral = (session_take + fragmento) >= umbral_minimo

                            debe_fragmentar = (
                                dias_antiguedad >= 15 or
                                porcentaje_procesable >= 0.6 or
                                (espacio_restante >= 76 and resto_significativo) or
                                not resto_significativo or
                                (necesita_mas_muestras and ayuda_umbral)
                            )

                            if debe_fragmentar and fragmento >= 38:
                                if mejor_opcion is None:
                                    mejor_opcion = (idx, fragmento, grp_of, "fragmentado_inteligente")

                    # PASO 3: Si aún no se alcanza el umbral del 75%, buscar cualquier fragmento viable
                    if necesita_mas_muestras and mejor_opcion is None and espacio_restante > 0:
                        for idx, muestras, grp_of in registros_grandes:
                            reg_data = df_state.iloc[idx]
                            dias_antiguedad = (datetime.now() - pd.to_datetime(reg_data["Fecha solicitud"])) .days

                            if dias_antiguedad >= 10 and espacio_restante >= 20:
                                fragmento_minimo = min(espacio_restante, muestras)
                                if (session_take + fragmento_minimo) >= umbral_minimo:
                                    mejor_opcion = (idx, fragmento_minimo, grp_of, "umbral_60")
                                    break

                    # Aplicar la mejor opción encontrada
                    if mejor_opcion:
                        idx, muestras, grp_of, tipo = mejor_opcion
                        session_take += muestras
                        registros_a_procesar.append((idx, muestras, grp_of))
            
            if session_take <= 0:
                break

            # Sesión de procesamiento directo
            proc_start = day_cursor
            proc_end = proc_start + timedelta(hours=t_proc)

            # Asignar muestras según registros_a_procesar (respeta no-fragmentación)
            session_registros = []
            
            for item in registros_a_procesar:
                # item puede ser (idx, take, grupo_real)
                if len(item) == 3:
                    idx, take, grupo_real = item
                else:
                    idx, take = item
                    grupo_real = grupo

                pend = int(df_state.loc[idx, "Pendiente"])
                
                schedule_rows.append({
                    "Fecha": proc_start.date(),
                    "Inicio": proc_start,
                    "Fin": proc_end,
                    "Registro": str(df_state.loc[idx, "Registro"]),
                    "Grupo": grupo_real,
                    "Tipo": "Procesamiento",
                    "Muestras": take,
                    "Fecha_Solicitud": df_state.loc[idx, "Fecha solicitud"]
                })
                df_state.loc[idx, "Pendiente"] = pend - take
                session_registros.append(f"{df_state.loc[idx, 'Registro']}({take})")

            used_seconds += (proc_end - proc_start).total_seconds()
            day_cursor = proc_end
            remaining_daily -= session_take
            day_samples_processed += session_take

        # Fin del procesamiento del día
        
        # Si no se procesó nada y hay muestras pendientes, reportar
        if day_samples_processed == 0 and (df_state["Pendiente"] > 0).any():
            pass  # Silenciar mensaje
        elif day_samples_processed == 0 and muestras_pendientes_total == 0:
            pass  # Silenciar mensaje

        # KPIs de utilización diaria mejorados
        day_total_seconds = (day_end - day_start).total_seconds()
        time_util = used_seconds / day_total_seconds if day_total_seconds > 0 else 0.0
        
        # Calcular utilización basada en muestras (más importante que tiempo)
        capacity_util = day_samples_processed / daily_cap if daily_cap > 0 else 0.0
        
        # Contar prealistamientos anticipados hechos hoy para mañana
        prep_anticipados = len([r for r in schedule_rows if r["Fecha"] == d.date() and "para mañana" in r.get("Registro", "")])
        
        daily_utilization.append({
            "Fecha": d.date(), 
            "Utilización (%)": round(capacity_util * 100, 1), 
            "Muestras procesadas": day_samples_processed,
            "Capacidad restante": remaining_daily,
            "Prep. anticipados": prep_anticipados,
            "Tiempo usado (h)": round(used_seconds / 3600, 1)
        })

        # Generar Gantt del día con progreso acumulado
        day_blocks = pd.DataFrame(schedule_rows)
        if not day_blocks.empty:
            day_blocks = day_blocks[(day_blocks["Fecha"] == d.date()) & (day_blocks["Tipo"] == "Procesamiento")]
        else:
            day_blocks = pd.DataFrame(columns=["Fecha","Inicio","Fin","Registro","Grupo","Tipo","Muestras","Fecha_Solicitud"])
            
        day_blocks_cons = consolidate_blocks(day_blocks)
        
        # Calcular progreso acumulado hasta este día
        gantt_day_rows = []
        for _, r in day_blocks_cons.iterrows():
            reg = str(r["Registro"])
            total_req = int(totals_by_reg.loc[totals_by_reg["Registro"] == reg, "TotalRegistro"].iloc[0]) if (totals_by_reg["Registro"] == reg).any() else 0
            accum_progress[reg] = accum_progress.get(reg, 0) + int(r["Muestras"])
            progreso = int(round(100 * min(accum_progress[reg], total_req) / total_req)) if total_req > 0 else 0
            
            # Obtener fecha de solicitud del registro
            fecha_sol = r.get("Fecha_Solicitud", "N/A")
            if pd.notna(fecha_sol):
                fecha_sol_str = pd.to_datetime(fecha_sol).strftime('%Y-%m-%d')
            else:
                fecha_sol_str = "N/A"
            
            gantt_day_rows.append({
                "Tarea": reg, 
                "Inicio": r["Inicio"], 
                "Fin": r["Fin"], 
                "Progreso": progreso,
                "Muestras_dia": int(r["Muestras"]),
                "Acumulado": accum_progress[reg],
                "Grupo": r["Grupo"],
                "Fecha_Solicitud": fecha_sol_str
            })
        
        gantt_per_day[d.date()] = pd.DataFrame(gantt_day_rows)
        
        pendientes_fin = (df_state["Pendiente"] > 0).sum()
        muestras_restantes = df_state["Pendiente"].sum()
        eficiencia_dia = round(capacity_util * 100, 1)
        
        # NUEVO: Verificar registros listos para entrega
        registros_entregables = obtener_registros_entregables(df_state)

    # Resultado semanal
    schedule_week = pd.DataFrame(schedule_rows).sort_values("Inicio").reset_index(drop=True)

    # Pendientes usando el estado final actualizado
    pend_base = df_state.copy()
    pend_base["Grupo"] = pend_base["Tipo de analisis"]
    pend_base["Solicitadas"] = pend_base["No muestras"]
    pend_base["Procesadas"] = pend_base["No muestras"] - pend_base["Pendiente"]
    pendientes = pend_base[["ID analisis","Registro","Tipo de analisis","Solicitadas","Procesadas","Pendiente"]]

    util_df = pd.DataFrame(daily_utilization)
    # Asegurar que todos los días laborales (get_week_days) estén representados.
    try:
        expected_dates = [d.date() for d in days]
        if util_df.empty:
            # Crear filas con ceros para cada día
            util_df = pd.DataFrame([{
                "Fecha": ed,
                "Utilización (%)": 0.0,
                "Muestras procesadas": 0,
                "Capacidad restante": daily_cap,
                "Prep. anticipados": 0,
                "Tiempo usado (h)": 0.0
            } for ed in expected_dates])
        else:
            existing = set(pd.to_datetime(util_df["Fecha"]).dt.date.tolist())
            missing = [ed for ed in expected_dates if ed not in existing]
            for ed in missing:
                util_df = pd.concat([util_df, pd.DataFrame([{
                    "Fecha": ed,
                    "Utilización (%)": 0.0,
                    "Muestras procesadas": 0,
                    "Capacidad restante": daily_cap,
                    "Prep. anticipados": 0,
                    "Tiempo usado (h)": 0.0
                }])], ignore_index=True)
        # Ordenar por Fecha
        util_df["Fecha"] = pd.to_datetime(util_df["Fecha"]).dt.date
        util_df = util_df.sort_values("Fecha").reset_index(drop=True)
    except Exception:
        # En caso de error inesperado, mantener lo calculado sin fallo
        pass

    # Gantt semanal (concatenación de días)
    # CREAR TABLA DE PROGRESO POR REGISTRO Y CATEGORÍA
    progreso_registros = []
    registros_unicos = df_state["Registro"].unique()
    
    for registro in registros_unicos:
        reg_data = df_state[df_state["Registro"] == registro]
        
        # Información base del registro
        fecha_solicitud = reg_data["Fecha solicitud"].iloc[0]
        # Usar total original por registro (no fragmentado)
        if (totals_by_reg["Registro"] == registro).any():
            total_muestras = int(totals_by_reg.loc[totals_by_reg["Registro"] == registro, "TotalRegistro"].iloc[0])
        else:
            # Fallback: sumar las filas en el estado (si no se encuentra en totals_by_reg)
            total_muestras = int(reg_data["No muestras"].sum())
        
        # Inicializar contadores por categoría (incluyendo Grupo D)
        progreso_reg = {
            "Registro": registro,
            "Fecha Solicitud": pd.to_datetime(fecha_solicitud).strftime('%Y-%m-%d'),
            "Total Muestras": total_muestras,
            "A_Procesadas": 0,
            "A_Pendientes": 0, 
            "B_Procesadas": 0,
            "B_Pendientes": 0,
            "C_Procesadas": 0,
            "C_Pendientes": 0,
            "D_Procesadas": 0,
            "D_Pendientes": 0,
            "Grupos_Requeridos": "",
            "Grupos_Completados": "",
            "Estado_Entrega": ""
        }
        
        # Calcular progreso por cada grupo (sumando fragmentos si existen)
        grupos_requeridos = []
        grupos_completados = []

        if not reg_data.empty:
            group_sums = reg_data.groupby("Tipo de analisis").agg({"No muestras": "sum", "Pendiente": "sum"})
            grupos_requeridos = sorted([g for g in group_sums.index.astype(str)])

            for g in grupos_requeridos:
                total_g = int(group_sums.loc[g, "No muestras"]) if g in group_sums.index else 0
                pend_g = int(group_sums.loc[g, "Pendiente"]) if g in group_sums.index else 0
                proc_g = total_g - pend_g

                if g == "A":
                    progreso_reg["A_Procesadas"] += proc_g
                    progreso_reg["A_Pendientes"] += pend_g
                elif g == "B":
                    progreso_reg["B_Procesadas"] += proc_g
                    progreso_reg["B_Pendientes"] += pend_g
                elif g == "C":
                    progreso_reg["C_Procesadas"] += proc_g
                    progreso_reg["C_Pendientes"] += pend_g
                elif g == "D":
                    progreso_reg["D_Procesadas"] += proc_g
                    progreso_reg["D_Pendientes"] += pend_g

                if pend_g == 0:
                    grupos_completados.append(g)
        
        # Determinar estado de entrega
        progreso_reg["Grupos_Requeridos"] = ", ".join(sorted(grupos_requeridos))
        progreso_reg["Grupos_Completados"] = ", ".join(sorted(grupos_completados))
        
        if len(grupos_completados) == len(grupos_requeridos):
            progreso_reg["Estado_Entrega"] = "✅ LISTO PARA ENTREGAR"
        elif len(grupos_completados) > 0:
            grupos_faltantes = [g for g in grupos_requeridos if g not in grupos_completados]
            progreso_reg["Estado_Entrega"] = f"⏳ Faltan: {', '.join(grupos_faltantes)}"
        else:
            progreso_reg["Estado_Entrega"] = "🔄 Pendiente de iniciar"
        
        progreso_registros.append(progreso_reg)
    
    # Crear DataFrame de progreso
    df_progreso = pd.DataFrame(progreso_registros)
    
    if len(gantt_per_day):
        gantt_week_df = pd.concat([g for g in gantt_per_day.values()], ignore_index=True)
    else:
        gantt_week_df = pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso"])

    return schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day, df_progreso, registros_info

def plot_gantt_user(df: pd.DataFrame, title: str, registros_info: pd.DataFrame = None):
    if df.empty:
        st.info("Sin bloques para mostrar.")
        return
    
    # Preparar datos para el Gantt mejorado
    df_plot = df.copy()
    if "Muestras_dia" in df_plot.columns and "Acumulado" in df_plot.columns:
        df_plot["Info"] = df_plot.apply(
            lambda x: f"{x['Tarea']}<br>Día: {x['Muestras_dia']} muestras<br>Acumulado: {x['Acumulado']}<br>Progreso: {x['Progreso']}%", 
            axis=1
        )
        hover_data = ["Info"]
    else:
        hover_data = ["Progreso"]
    
    fig = px.timeline(
        df_plot, 
        x_start="Inicio", 
        x_end="Fin", 
        y="Tarea", 
        color="Grupo" if "Grupo" in df_plot.columns else "Progreso",
        title=title,
        text="Muestras_dia" if "Muestras_dia" in df_plot.columns else None,  # Agregar texto con muestras
        color_discrete_map={
            "A": "#FFD700",  # Amarillo para grupo A
            "B": "#FF4444",  # Rojo para grupo B  
            "C": "#00CC66",  # Verde para grupo C
            "D": "#9966FF",  # Púrpura para grupo D
            "B_C_CONJUNTO": "#FF8C00"  # Naranja para conjunto B+C
        } if "Grupo" in df_plot.columns else None,
        color_continuous_scale="RdYlGn" if "Grupo" not in df_plot.columns else None,
        range_color=[0, 100] if "Grupo" not in df_plot.columns else None,
        hover_data=["Grupo", "Muestras_dia"] if "Grupo" in df_plot.columns and "Muestras_dia" in df_plot.columns else None
    )
    
    # Configurar y mostrar el gráfico
    fig.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        automargin=True,  # Ajuste automático de márgenes
        tickfont=dict(size=11)
    )
    
    # Mejorar configuración del eje X
    fig.update_xaxes(
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        tickfont=dict(size=11),
        tickangle=0  # Etiquetas horizontales para mejor legibilidad
    )
    
    # Configurar eje X para Gantt semanal
    if "Semanal" in title:
        # Para Gantt semanal, usar solo días laborales (Lunes a Jueves)
        dias_laborales = ['Lunes', 'Martes', 'Miércoles', 'Jueves']
        fecha_range = pd.date_range(
            start=df_plot['Inicio'].min().date(),
            periods=4,  # Solo 4 días laborales
            freq='D'
        )
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=fecha_range,
                ticktext=dias_laborales,
                tickangle=0,
                showgrid=True,
                gridcolor='lightgray',
                range=[fecha_range[0] - pd.Timedelta(hours=2), 
                       fecha_range[-1] + pd.Timedelta(hours=10)]  # Ampliar un poco el rango para mejor visualización
            )
        )
    
    # Configurar texto en las barras
    if "Muestras_dia" in df_plot.columns:
        fig.update_traces(
            textposition="inside",
            textfont=dict(size=11, color="white", family="Arial Black"),
            texttemplate="<b>%{text}</b>",
            marker=dict(line=dict(width=1, color='rgba(0,0,0,0.2)'))  # Borde sutil
        )
    
    fig.update_layout(
        xaxis_title="📅 Días de la Semana" if "Semanal" in title else "📅 Tiempo", 
        yaxis_title="📋 Actividades", 
        height=max(450, len(df_plot) * 45 + 120),  # Más espacio vertical
        font=dict(size=12),
        legend_title_text="🎯 Grupos de Análisis",  # Título más claro para la leyenda
        margin=dict(l=30, r=30, t=70, b=50),  # Márgenes más equilibrados
        showlegend=True,
        legend=dict(
            orientation="h",  # Leyenda horizontal
            yanchor="bottom",
            y=1.02,
            xanchor="center",  # Centrar la leyenda
            x=0.5,
            font=dict(size=11)
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        paper_bgcolor='rgba(0,0,0,0)'  # Fondo del papel transparente
    )
    st.plotly_chart(fig, use_container_width=True)

# NOTE: La implementación del algoritmo FIFO simple (plan_fifo_simple) fue eliminada
# por petición del usuario. Si se requiere una versión simplificada en el futuro,
# puede reimplementarse usando `expand_analyses` y reglas FIFO básicas.

def to_excel_download(schedule: pd.DataFrame, pendientes: pd.DataFrame, util_df: pd.DataFrame,
                      gantt_week: pd.DataFrame, gantt_per_day: Dict, prueba: pd.DataFrame = None, 
                      tiempo: pd.DataFrame = None, selected_date: datetime = None, daily_cap: int = None) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        # Verificar y exportar schedule
        if isinstance(schedule, pd.DataFrame) and not schedule.empty:
            schedule.to_excel(writer, sheet_name="Plan", index=False)
        else:
            # Crear DataFrame vacío como fallback
            pd.DataFrame().to_excel(writer, sheet_name="Plan", index=False)
            
        # Verificar y exportar pendientes
        if isinstance(pendientes, pd.DataFrame) and not pendientes.empty:
            pendientes.to_excel(writer, sheet_name="Pendiente", index=False)
        else:
            # Crear DataFrame vacío como fallback
            pd.DataFrame().to_excel(writer, sheet_name="Pendiente", index=False)
            
        # Verificar y exportar util_df
        if isinstance(util_df, pd.DataFrame) and not util_df.empty:
            util_df.to_excel(writer, sheet_name="Utilizacion", index=False)
        else:
            # Crear DataFrame vacío como fallback
            pd.DataFrame().to_excel(writer, sheet_name="Utilizacion", index=False)
            
        # Verificar y exportar gantt_week
        if isinstance(gantt_week, pd.DataFrame) and not gantt_week.empty:
            gantt_week.to_excel(writer, sheet_name="Gantt_semana", index=False)
        else:
            # Crear DataFrame vacío como fallback
            pd.DataFrame().to_excel(writer, sheet_name="Gantt_semana", index=False)
            
        # Verificar y exportar gantt_per_day
        if isinstance(gantt_per_day, dict):
            for k, v in gantt_per_day.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    sheet = f"Gantt_{k.strftime('%a')}"
                    v.to_excel(writer, sheet_name=sheet, index=False)
        
        # Eliminado: generación de hoja "Optimizacion" y ejecución del modelo FIFO/optimizado
        # por petición del usuario. El Excel exporta ahora solo: Plan, Pendiente, Utilizacion y Gantt.
        
        # NOTE: Eliminado el bloque de 'Tasas_Llegada' (análisis históricos) por petición.
    
    return out.getvalue()

# =============================
# UI
# =============================
# Leer configuración persistente de forma silenciosa (sin exponerla en la UI)
config = read_config()

# Nota: La edición/visualización de `config.json` fue removida de la interfaz.
# La configuración se aplica automáticamente como fallback cuando no se sube
# un Excel o cuando falta la hoja 'Tiempo'.

default_path = "Insumo_Planeacion.xlsx"
uploaded = st.sidebar.file_uploader("Cargar Excel personalizado", type=["xlsx", "xlsm"])

# Cargar datos desde Excel (si el usuario sube uno). Si no se sube pero se selecciona usar persistente,
# usaremos la capacidad y tiempos guardados para sobreescribir lo cargado desde Excel.
foliar, prueba, tiempo, daily_cap = load_data(default_path, uploaded.getvalue() if uploaded else None)

# Aplicar configuración persistente automáticamente solo si NO se subió un Excel
# (la edición de config.json fue removida de la UI)
if uploaded is None and config:
    cfg = config
    try:
        daily_cap = int(cfg.get("daily_cap", daily_cap))
    except Exception:
        pass

    # Construir DataFrame `tiempo` a partir de la configuración
    times = cfg.get("times", {})
    tiempo_rows = []
    for g, vals in times.items():
        tiempo_rows.append({
            "Grupo": g,
            "Tiempo de prealistamiento (horas)": float(vals.get("pre", 0.0)),
            "Tiempo procesamiento (horas)": float(vals.get("proc", 0.0))
        })
    try:
        tiempo = pd.DataFrame(tiempo_rows)
    except Exception:
        # si falla, dejar el tiempo tal como lo cargó load_data
        pass

# Si la hoja 'Tiempo' no existía en el Excel y no contiene datos, aplicar config como fallback
if (tiempo is None) or ("Grupo" not in tiempo.columns) or tiempo.empty:
    cfg = config
    times = cfg.get("times", {})
    tiempo_rows = []
    for g, vals in times.items():
        tiempo_rows.append({
            "Grupo": g,
            "Tiempo de prealistamiento (horas)": float(vals.get("pre", 0.0)),
            "Tiempo procesamiento (horas)": float(vals.get("proc", 0.0))
        })
    try:
        tiempo = pd.DataFrame(tiempo_rows)
    except Exception:
        pass
    try:
        daily_cap = int(cfg.get("daily_cap", daily_cap))
    except Exception:
        pass

tab = st.container()

with tab:
    st.subheader("Parámetros de planeación (Lun–Vie)")
    today = datetime.today()
    selected_date = st.date_input("Selecciona cualquier día de la semana a planificar", value= today)
    cap_txt = f"{daily_cap} muestras/día" if daily_cap is not None else "sin tope diario (solo limitado por tiempo)"
    st.caption(f"""
    **🚀 Planeación Optimizada** - Lunes a Jueves, 8:00–18:00, capacidad diaria = {cap_txt}
    """)
    if st.button("Planificar semana (capacidad diaria)"):
        schedule, pendientes, util_df, gantt_week, gantt_per_day, df_progreso, registros_info = plan_week_by_day(prueba, tiempo, datetime.combine(selected_date, DAY_START), daily_cap)
        if schedule.empty and len(gantt_per_day) == 0:
            st.stop()
        st.success("Planeación generada (toda la semana).")

        # NUEVA TABLA: Progreso por registro y categoría
        st.subheader("📊 Progreso de Muestras por Registro y Categoría")
        
        # Mostrar tabla con formato (incluyendo Grupo D)
        st.dataframe(
            df_progreso[["Registro", "Fecha Solicitud", "Total Muestras", 
                        "A_Procesadas", "A_Pendientes", 
                        "B_Procesadas", "B_Pendientes", 
                        "C_Procesadas", "C_Pendientes",
                        "D_Procesadas", "D_Pendientes", 
                        "Grupos_Requeridos", "Estado_Entrega"]],
            use_container_width=True,
            column_config={
                "Registro": st.column_config.TextColumn("📋 Registro"),
                "Fecha Solicitud": st.column_config.TextColumn("📅 Fecha"),
                "Total Muestras": st.column_config.NumberColumn("🧪 Total", format="%d"),
                "A_Procesadas": st.column_config.NumberColumn("🟡 A Proc.", format="%d"),
                "A_Pendientes": st.column_config.NumberColumn("🟡 A Pend.", format="%d"),
                "B_Procesadas": st.column_config.NumberColumn("🔴 B Proc.", format="%d"),
                "B_Pendientes": st.column_config.NumberColumn("🔴 B Pend.", format="%d"),
                "C_Procesadas": st.column_config.NumberColumn("🟢 C Proc.", format="%d"),
                "C_Pendientes": st.column_config.NumberColumn("🟢 C Pend.", format="%d"),
                "D_Procesadas": st.column_config.NumberColumn("🟣 D Proc.", format="%d"),
                "D_Pendientes": st.column_config.NumberColumn("🟣 D Pend.", format="%d"),
                "Grupos_Requeridos": st.column_config.TextColumn("📊 Grupos Req."),
                "Estado_Entrega": st.column_config.TextColumn("🎯 Estado Entrega")
            }
        )
        
        # Resumen estadístico (actualizado para incluir Grupo D)
        col1, col2, col3, col4 = st.columns(4)
        
        registros_listos = len([r for _, r in df_progreso.iterrows() if "LISTO PARA ENTREGAR" in r["Estado_Entrega"]])
        registros_en_progreso = len([r for _, r in df_progreso.iterrows() if "Faltan:" in r["Estado_Entrega"]])
        registros_pendientes = len([r for _, r in df_progreso.iterrows() if "Pendiente de iniciar" in r["Estado_Entrega"]])
        total_muestras_procesadas = (df_progreso["A_Procesadas"] + df_progreso["B_Procesadas"] + df_progreso["C_Procesadas"] + df_progreso["D_Procesadas"]).sum()
        
        with col1:
            st.metric("✅ Listos para Entrega", registros_listos)
        with col2:
            st.metric("⏳ En Progreso", registros_en_progreso)
        with col3:
            st.metric("🔄 Pendientes", registros_pendientes)
        with col4:
            st.metric("🧪 Muestras Procesadas", total_muestras_procesadas)

        # Resumen ejecutivo
        st.markdown("### 📊 Resumen Ejecutivo de la Planeación")
        total_muestras = util_df["Muestras procesadas"].sum() if not util_df.empty else 0
        muestras_pendientes_total = registros_en_progreso + registros_pendientes  # Suma de En Progreso + Pendientes
        utilizacion_promedio = util_df["Utilización (%)"].mean() if not util_df.empty else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧪 Muestras Procesadas", total_muestras)
        with col2:
            st.metric("⏳ Muestras Pendientes", muestras_pendientes_total)
        with col3:
            st.metric("⚡ Utilización Promedio", f"{utilizacion_promedio:.1f}%")

        st.markdown("### 📅 Gantt Diario - Progreso Acumulado")
        days = sorted(list(gantt_per_day.keys()))
        names = ["Lunes","Martes","Miércoles","Jueves"]
        
        # Pestañas para cada día
        tabs = st.tabs([f"{names[i]} ({d.strftime('%m/%d')})" for i, d in enumerate(days)])
        
        for i, (d, tab) in enumerate(zip(days, tabs)):
            with tab:
                day_data = gantt_per_day[d]
                if not day_data.empty:
                    # Métricas del día
                    muestras_dia = day_data["Muestras_dia"].sum() if "Muestras_dia" in day_data.columns else 0
                    registros_dia = len(day_data)
                    progreso_promedio = day_data["Progreso"].mean() if not day_data.empty else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Muestras {names[i]}", muestras_dia)
                    with col2:
                        st.metric("Registros Procesados", registros_dia)
                    with col3:
                        st.metric("Progreso Promedio", f"{progreso_promedio:.1f}%")
                    
                    # Gantt del día
                    plot_gantt_user(day_data, title=f"Programación {names[i]} - {d.strftime('%Y-%m-%d')}", registros_info=registros_info)
                    
                    # Tabla detalle del día
                    if "Muestras_dia" in day_data.columns:
                        st.markdown("**Detalle del día:**")
                        detalle_dia = day_data[["Tarea", "Muestras_dia", "Acumulado", "Progreso"]].rename(columns={
                            "Tarea": "Registro",
                            "Muestras_dia": "Muestras Procesadas",
                            "Acumulado": "Total Acumulado",
                            "Progreso": "% Completado"
                        })
                        st.dataframe(detalle_dia, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No hay actividades programadas para {names[i]}")

        st.markdown("### 🗓️ Vista Semanal Consolidada")
        plot_gantt_user(gantt_week, title="Gantt Semanal - Progreso Acumulativo", registros_info=registros_info)

        # Tabla de información de muestras por registro (solo para vista semanal)
        if not gantt_week.empty and "Muestras_dia" in gantt_week.columns:
            st.caption("📊 Información de muestras por registro:")
            
            # Crear tabla mejorada con fecha de solicitud y grupos
            tabla_mejorada = []
            registros_procesados = set()
            
            for _, row in gantt_week.iterrows():
                registro = row['Tarea']
                
                # Evitar duplicados
                if registro in registros_procesados:
                    continue
                registros_procesados.add(registro)
                
                # Sumar todas las muestras de este registro en la semana
                muestras_total = gantt_week[gantt_week['Tarea'] == registro]['Muestras_dia'].sum()
                fecha_sol = row.get('Fecha_Solicitud', 'N/A')
                
                # Obtener grupos solicitados desde registros_info si está disponible
                if registros_info is not None and not registros_info.empty:
                    reg_info = registros_info[registros_info['Registro'] == registro]
                    if not reg_info.empty:
                        grupos_solicitados = reg_info['Grupos_requeridos'].iloc[0]
                        grupos_str = ", ".join(grupos_solicitados) if isinstance(grupos_solicitados, list) else str(grupos_solicitados)
                    else:
                        grupos_str = "N/A"
                else:
                    # Fallback: obtener grupos del DataFrame actual
                    mismo_registro = gantt_week[gantt_week['Tarea'] == registro]
                    grupos_total = mismo_registro['Grupo'].unique().tolist() if 'Grupo' in mismo_registro.columns else ["N/A"]
                    grupos_str = ", ".join(sorted(grupos_total))
                
                # Grupos procesados en la semana
                grupos_semana = gantt_week[gantt_week['Tarea'] == registro]['Grupo'].unique().tolist()
                grupos_semana_str = ", ".join(sorted(grupos_semana))
                
                tabla_mejorada.append({
                    '📋 Registro': registro,
                    '🧪 Muestras Procesadas': muestras_total,
                    '📅 Fecha Solicitud': fecha_sol,
                    '🎯 Grupos Solicitados': grupos_str,
                    '⚗️ Grupos Procesados': grupos_semana_str
                })
            
            df_tabla = pd.DataFrame(tabla_mejorada)
            st.dataframe(df_tabla, use_container_width=True, hide_index=True)

        # Detalles técnicos en expanders
        
        # Eliminado: análisis de tasas de llegada (históricos) por petición del usuario.

        with st.expander("📋 Detalle Completo de la Planeación"):
            st.dataframe(schedule, use_container_width=True)
            
        with st.expander("⏰ Análisis de Pendientes"):
            # NUEVA TABLA: Registros pendientes por procesar (detalle por grupo)
            st.markdown("**📋 Registros Pendientes por Procesar:**")
            
            # Crear tabla detallada de registros pendientes
            registros_pendientes_detalle = []
            
            # Filtrar TODOS los registros que NO están listos para entrega
            registros_no_listos = df_progreso[
                ~df_progreso["Estado_Entrega"].str.contains("✅ LISTO PARA ENTREGAR", na=False)
            ]
            
            for _, reg in registros_no_listos.iterrows():
                registro = reg["Registro"]
                fecha_sol = reg["Fecha Solicitud"]
                
                # Analizar cada grupo individualmente
                grupos_analizar = [
                    ("A", reg["A_Pendientes"]),
                    ("B", reg["B_Pendientes"]),
                    ("C", reg["C_Pendientes"]),
                    ("D", reg["D_Pendientes"])
                ]
                
                for grupo, muestras_pend in grupos_analizar:
                    if muestras_pend > 0:  # Solo agregar si hay muestras pendientes
                        # TODOS los registros no listos son considerados PENDIENTES
                        estado_grupo = "🔄 Pendiente por Procesar"
                        
                        registros_pendientes_detalle.append({
                            "📋 Registro": registro,
                            "📅 Fecha Solicitud": fecha_sol,
                            "🎯 Grupo": grupo,
                            "🧪 Muestras Pendientes": int(muestras_pend),
                            "📊 Estado": estado_grupo
                        })
            
            if registros_pendientes_detalle:
                df_pendientes_detalle = pd.DataFrame(registros_pendientes_detalle)
                # Ordenar por fecha de solicitud y registro
                df_pendientes_detalle = df_pendientes_detalle.sort_values(["📅 Fecha Solicitud", "📋 Registro", "🎯 Grupo"])
                st.dataframe(df_pendientes_detalle, use_container_width=True, hide_index=True)
                
                # Resumen de pendientes
                total_muestras_pendientes = df_pendientes_detalle["🧪 Muestras Pendientes"].sum()
                total_registros_pendientes = df_pendientes_detalle["📋 Registro"].nunique()
                total_grupos_pendientes = len(df_pendientes_detalle)
                st.error(f"⚠️ **PENDIENTES POR PROCESAR**: {total_registros_pendientes} registros, {total_grupos_pendientes} grupos, {total_muestras_pendientes} muestras SIN PROCESAR")
                
                st.markdown("---")
                
                # NUEVA TABLA: Sugerencia de programación para la siguiente semana
                st.markdown("**📅 Sugerencia de Programación - Próxima Semana:**")
                
                # Crear sugerencia de programación
                sugerencia_programacion = []
                
                # Calcular prioridad basada en días de antigüedad
                fecha_actual = datetime.now().date()
                
                for _, row in df_pendientes_detalle.iterrows():
                    registro = row["📋 Registro"]
                    fecha_sol_str = row["📅 Fecha Solicitud"]
                    grupo = row["🎯 Grupo"]
                    muestras = row["🧪 Muestras Pendientes"]
                    
                    # Calcular días de antigüedad
                    try:
                        fecha_sol = pd.to_datetime(fecha_sol_str).date()
                        dias_antiguedad = (fecha_actual - fecha_sol).days
                    except:
                        dias_antiguedad = 0
                    
                    # Determinar prioridad
                    if dias_antiguedad >= 20:
                        prioridad = "🚨 CRÍTICA"
                        prioridad_num = 1
                    elif dias_antiguedad >= 10:
                        prioridad = "⚠️ ALTA"
                        prioridad_num = 2
                    elif dias_antiguedad >= 5:
                        prioridad = "📋 MEDIA"
                        prioridad_num = 3
                    else:
                        prioridad = "📅 NORMAL"
                        prioridad_num = 4
                    
                    # Sugerir día de procesamiento basado en restricciones
                    if grupo == "A":
                        dia_sugerido = "Lunes"
                    elif grupo == "D":
                        dia_sugerido = "Lunes"
                    elif grupo in ["B", "C"]:
                        dia_sugerido = "Martes"
                    else:
                        dia_sugerido = "Miércoles/Jueves"
                    
                    sugerencia_programacion.append({
                        "📋 Registro": registro,
                        "🧪 Muestras": muestras,
                        "🎯 Grupo": grupo,
                        "🚨 Prioridad": prioridad,
                        "📅 Día Sugerido": dia_sugerido,
                        "⏰ Días de Antigüedad": dias_antiguedad,
                        "_prioridad_num": prioridad_num  # Para ordenamiento
                    })
                
                if sugerencia_programacion:
                    df_sugerencia = pd.DataFrame(sugerencia_programacion)
                    # Ordenar por prioridad (más crítico primero) y luego por días de antigüedad
                    df_sugerencia = df_sugerencia.sort_values(["_prioridad_num", "⏰ Días de Antigüedad"], ascending=[True, False])
                    
                    # Remover columna auxiliar de ordenamiento
                    df_sugerencia_display = df_sugerencia.drop("_prioridad_num", axis=1)
                    
                    st.dataframe(df_sugerencia_display, use_container_width=True, hide_index=True)
                    
                    # Resumen de la programación sugerida
                    criticas = len(df_sugerencia[df_sugerencia["🚨 Prioridad"] == "🚨 CRÍTICA"])
                    altas = len(df_sugerencia[df_sugerencia["🚨 Prioridad"] == "⚠️ ALTA"])
                    medias = len(df_sugerencia[df_sugerencia["🚨 Prioridad"] == "📋 MEDIA"])
                    normales = len(df_sugerencia[df_sugerencia["🚨 Prioridad"] == "📅 NORMAL"])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🚨 Críticas", criticas)
                    with col2:
                        st.metric("⚠️ Altas", altas)
                    with col3:
                        st.metric("📋 Medias", medias)
                    with col4:
                        st.metric("📅 Normales", normales)
                
            else:
                st.success("🎉 ¡No hay registros pendientes por procesar!")
            
            st.markdown("---")
            
            # Análisis de pendientes por grupo (solo si hay datos técnicos)
            if isinstance(pendientes, pd.DataFrame) and not pendientes.empty:
                pend_por_grupo = pendientes.groupby("Tipo de analisis").agg({
                    "Pendiente": "sum",
                    "Registro": "count"
                }).rename(columns={"Registro": "Num_Registros"}).reset_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pendientes por Tipo de Análisis:**")
                    st.dataframe(pend_por_grupo, use_container_width=True, hide_index=True)
                with col2:
                    st.markdown("**Todos los Pendientes (detalle técnico):**")
                    st.dataframe(pendientes, use_container_width=True, hide_index=True)
            else:
                st.info("📊 No hay datos técnicos de pendientes disponibles")

        with st.expander("⚡ KPIs de Utilización Diaria Optimizada"):
            st.markdown("**Métricas mejoradas con prealistamiento anticipado:**")
            st.dataframe(util_df, use_container_width=True, hide_index=True)
            
            # Gráficos de utilización mejorados
            if not util_df.empty:
                fig_util = px.bar(util_df, x="Fecha", y="Utilización (%)", 
                                title="Utilización Diaria del Laboratorio",
                                color="Utilización (%)",
                                color_continuous_scale="RdYlGn",
                                text="Utilización (%)")
                fig_util.update_layout(xaxis_title="Día", yaxis_title="% Utilización")
                fig_util.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig_util, use_container_width=True)
                
                # Resumen de optimización
                total_muestras = util_df["Muestras procesadas"].sum()
                avg_utilizacion = util_df["Utilización (%)"].mean()
                total_prep_anticipados = util_df["Prep. anticipados"].sum()
                
                st.markdown("### 📈 Resumen de Optimización")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🧪 Total Muestras", total_muestras)
                with col2:
                    st.metric("⚡ Utilización Promedio", f"{avg_utilizacion:.1f}%")
                with col3:
                    st.metric("🌅 Prealist. Anticipados", total_prep_anticipados)

        xls_bytes = to_excel_download(schedule, pendientes, util_df, gantt_week, gantt_per_day)
        st.download_button(
            label="⬇️ Descargar plan (semana)",
            data=xls_bytes,
            file_name="planeacion_semana.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Eliminado: pestaña y UI de 'Análisis Comparativo de Optimización' por petición del usuario.
# Si se requiere volver a añadir comparación, podemos reimplementar una versión mínima sin dependencias históricas.
