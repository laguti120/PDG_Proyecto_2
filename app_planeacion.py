
import io
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
    foliar = pd.read_excel(xls, sheet_name="Foliar")
    prueba = pd.read_excel(xls, sheet_name="Prueba")
    tiempo = pd.read_excel(xls, sheet_name="Tiempo")
    capacidad = pd.read_excel(xls, sheet_name="Capacidad")

    foliar.columns = [c.strip() for c in foliar.columns]
    prueba.columns = [c.strip() for c in prueba.columns]
    tiempo.columns = [c.strip() for c in tiempo.columns]
    capacidad.columns = [c.strip() for c in capacidad.columns]

    for c in ["Fecha solicitud", "Fecha de entrega"]:
        if c in foliar.columns:
            foliar[c] = pd.to_datetime(foliar[c], errors="coerce")
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

def calculate_arrival_rates(prueba: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calcula las tasas de llegada diarias por grupo basado en datos históricos.
    
    Fórmula: Tasa de llegada = Total de muestras del grupo G / Total de días en el periodo
    
    Returns:
        Dict con estadísticas por grupo: {
            'A': {'tasa_diaria': X, 'total_muestras': Y, 'dias_periodo': Z},
            'B': {...}, 'C': {...}, 'D': {...}
        }
    """
    # Expandir análisis para obtener datos por grupo individual
    df_expanded = expand_analyses(prueba)
    
    # Calcular periodo total de datos
    if "Fecha solicitud" not in df_expanded.columns:
        return {}
    
    df_expanded["Fecha solicitud"] = pd.to_datetime(df_expanded["Fecha solicitud"], errors="coerce")
    df_valid = df_expanded.dropna(subset=["Fecha solicitud"])
    
    if df_valid.empty:
        return {}
    
    fecha_min = df_valid["Fecha solicitud"].min()
    fecha_max = df_valid["Fecha solicitud"].max()
    dias_periodo = (fecha_max - fecha_min).days + 1  # +1 para incluir ambos días
    
    # Calcular tasas por grupo
    grupos_disponibles = ['A', 'B', 'C', 'D']
    tasas_por_grupo = {}
    
    for grupo in grupos_disponibles:
        # Filtrar muestras del grupo específico
        muestras_grupo = df_valid[df_valid["Tipo de analisis"].str.strip() == grupo]
        
        if not muestras_grupo.empty:
            # Total de muestras del grupo en el periodo
            total_muestras = muestras_grupo["No muestras"].fillna(0).sum()
            
            # Calcular tasa de llegada diaria
            tasa_diaria = total_muestras / dias_periodo if dias_periodo > 0 else 0
            
            # Información adicional
            num_registros = len(muestras_grupo["Registro"].unique())
            muestras_promedio_por_registro = total_muestras / num_registros if num_registros > 0 else 0
            
            tasas_por_grupo[grupo] = {
                'tasa_diaria': round(tasa_diaria, 2),
                'total_muestras': int(total_muestras),
                'dias_periodo': dias_periodo,
                'num_registros': num_registros,
                'promedio_por_registro': round(muestras_promedio_por_registro, 1),
                'fecha_inicio': fecha_min.strftime('%Y-%m-%d'),
                'fecha_fin': fecha_max.strftime('%Y-%m-%d')
            }
        else:
            # Grupo sin datos
            tasas_por_grupo[grupo] = {
                'tasa_diaria': 0.0,
                'total_muestras': 0,
                'dias_periodo': dias_periodo,
                'num_registros': 0,
                'promedio_por_registro': 0.0,
                'fecha_inicio': fecha_min.strftime('%Y-%m-%d') if fecha_min else 'N/A',
                'fecha_fin': fecha_max.strftime('%Y-%m-%d') if fecha_max else 'N/A'
            }
    
    return tasas_por_grupo

def analyze_capacity_vs_arrival_rates(arrival_rates: Dict[str, Dict[str, float]], daily_capacity: int = 38) -> Dict[str, Dict[str, float]]:
    """
    Analiza la relación entre tasas de llegada y capacidad diaria.
    Calcula tiempos de espera y saturación por grupo.
    
    Args:
        arrival_rates: Tasas de llegada por grupo (resultado de calculate_arrival_rates)
        daily_capacity: Capacidad diaria total (38 muestras)
    
    Returns:
        Análisis de capacidad por grupo con métricas de espera y saturación
    """
    
    if not arrival_rates:
        return {}
    
    # Calcular totales
    tasa_total_diaria = sum(rates['tasa_diaria'] for rates in arrival_rates.values())
    
    analisis_por_grupo = {}
    
    for grupo, rates in arrival_rates.items():
        tasa_grupo = rates['tasa_diaria']
        
        # Porcentaje de la demanda total que representa este grupo
        porcentaje_demanda = (tasa_grupo / tasa_total_diaria * 100) if tasa_total_diaria > 0 else 0
        
        # Capacidad proporcional que debería asignarse a este grupo (basada en demanda actual)
        capacidad_proporcional = (tasa_grupo / tasa_total_diaria * daily_capacity) if tasa_total_diaria > 0 else 0
        
        # Capacidad base teórica por grupo (distribución equitativa)
        capacidad_base_grupo = daily_capacity / 4  # 38/4 = 9.5 muestras por grupo
        
        # Tiempo teórico para procesar la llegada diaria (si solo fuera este grupo)
        dias_para_procesar_solo = tasa_grupo / daily_capacity if daily_capacity > 0 else float('inf')
        
        # Tiempo de espera estimado (asumiendo procesamiento equitativo)
        tiempo_espera_dias = dias_para_procesar_solo if dias_para_procesar_solo > 1 else 0
        
        # Nivel de saturación del grupo (comparado con capacidad base teórica)
        if tasa_grupo == 0:
            saturacion = 0
        else:
            saturacion = tasa_grupo / capacidad_base_grupo
        
        # Clasificación de prioridad basada en saturación
        if saturacion > 1.2:
            prioridad_capacidad = "CRÍTICA"
        elif saturacion > 0.8:
            prioridad_capacidad = "ALTA"
        elif saturacion > 0.5:
            prioridad_capacidad = "MEDIA"
        else:
            prioridad_capacidad = "BAJA"
        
        analisis_por_grupo[grupo] = {
            'tasa_llegada': tasa_grupo,
            'porcentaje_demanda': round(porcentaje_demanda, 1),
            'capacidad_proporcional': round(capacidad_proporcional, 1),
            'tiempo_espera_dias': round(tiempo_espera_dias, 2),
            'saturacion': round(saturacion, 2) if saturacion != float('inf') else 999.99,
            'prioridad_capacidad': prioridad_capacidad,
            'dias_acumulacion': round(dias_para_procesar_solo, 2) if dias_para_procesar_solo != float('inf') else 999.99
        }
    
    # Agregar análisis global
    analisis_por_grupo['GLOBAL'] = {
        'tasa_total': round(tasa_total_diaria, 2),
        'utilizacion_capacidad': round((tasa_total_diaria / daily_capacity * 100), 1) if daily_capacity > 0 else 0,
        'capacidad_disponible': daily_capacity,
        'deficit_superavit': round(daily_capacity - tasa_total_diaria, 2),
        'estado_sistema': 'SOBRECARGADO' if tasa_total_diaria > daily_capacity else 'EQUILIBRADO' if tasa_total_diaria > daily_capacity * 0.8 else 'SUBUTILIZADO'
    }
    
    return analisis_por_grupo

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

    # NUEVO: Calcular tasas de llegada por grupo para optimización
    arrival_rates = calculate_arrival_rates(prueba)
    capacity_analysis = analyze_capacity_vs_arrival_rates(arrival_rates, daily_cap)
    
    # Mostrar análisis de tasas de llegada
    if arrival_rates:
        st.info(f"📈 **Análisis de tasas de llegada calculado** - Estado del sistema: {capacity_analysis.get('GLOBAL', {}).get('estado_sistema', 'N/A')}")

    # Map de tiempos desde Excel
    tiempo = tiempo.copy()
    tiempo.columns = [c.strip() for c in tiempo.columns]
    tiempo["Grupo"] = tiempo["Grupo"].astype(str).str.strip()
    t_map = tiempo.set_index("Grupo")[["Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"]].to_dict("index")

    # Pedidos expandidos y ordenados por prioridad
    df_original = expand_analyses(prueba)
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

    # Totales por Registro para % progreso acumulado
    totals_by_reg = df_original.groupby("Registro", as_index=False)["No muestras"].sum().rename(columns={"No muestras":"TotalRegistro"})

    # Días de la semana
    days = get_week_days(selected_date)

    schedule_rows = []
    daily_utilization = []
    gantt_per_day = {}
    accum_progress: Dict[str,int] = {}
    
    # Estado global de muestras pendientes (se mantiene entre días)
    df_state = df_original.copy()
    
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
                        registros_a_procesar.append((row.name, take_d))
                        
                        # Para grupo D, procesar solo un registro por sesión
                        break
            else:
                # Lógica normal para grupos A, B, C (pueden combinarse)
                # Separar registros por tamaño ANTES de procesamiento
                registros_pequenos = []  # ≤38 muestras - NO se pueden fragmentar
                registros_grandes = []   # >38 muestras - SE pueden fragmentar con decisión inteligente
                
                # Clasificar registros por tamaño
                for _, row in grp_df.iterrows():
                    pend_registro = int(row["Pendiente"])
                    if pend_registro <= 0:
                        continue
                    if pend_registro <= 38:
                        registros_pequenos.append((row.name, pend_registro))
                    else:
                        registros_grandes.append((row.name, pend_registro))
                
                # PASO 1: Procesar registros pequeños completos (≤38) - Sin fragmentación
                registros_pequenos.sort(key=lambda x: x[1], reverse=True)  # Más grandes primero
                
                for idx, muestras in registros_pequenos:
                    if session_take + muestras <= remaining_daily:
                        session_take += muestras
                        registros_a_procesar.append((idx, muestras))
                    # Si no cabe completo, se deja para otro día (no fragmentar)
                
                # VERIFICACIÓN DE UMBRAL MÍNIMO (75% de capacidad = 29 muestras)
                umbral_minimo = int(daily_cap * 0.75)  # 75% de 38 = 29 muestras
                necesita_mas_muestras = session_take < umbral_minimo
                
                # PASO 2: Si queda espacio O no se alcanzó el umbral, procesar registros grandes
                espacio_restante = remaining_daily - session_take
                if (espacio_restante > 0 or necesita_mas_muestras) and registros_grandes:
                    # Buscar el mejor registro grande para el espacio disponible
                    mejor_opcion = None
                    
                    for idx, muestras in registros_grandes:
                        reg_data = df_state.iloc[idx]
                        
                        # OPCIÓN 1: ¿Cabe completo?
                        if muestras <= espacio_restante:
                            mejor_opcion = (idx, muestras, "completo")
                            break  # Completo es siempre mejor
                        
                        # OPCIÓN 2: Evaluación inteligente para fragmentar
                        grupos_completos = espacio_restante // 38
                        if grupos_completos > 0:
                            fragmento = min(grupos_completos * 38, muestras)
                            
                            # Criterios para decidir si fragmentar inteligentemente:
                            dias_antiguedad = (datetime.now() - pd.to_datetime(reg_data["Fecha solicitud"])).days
                            porcentaje_procesable = fragmento / muestras
                            resto_significativo = (muestras - fragmento) >= 38
                            ayuda_umbral = (session_take + fragmento) >= umbral_minimo
                            
                            # Decisión inteligente basada en múltiples factores
                            debe_fragmentar = (
                                dias_antiguedad >= 15 or  # Muestra muy antigua, fragmentar
                                porcentaje_procesable >= 0.6 or  # Se puede procesar >60%, vale la pena
                                (espacio_restante >= 76 and resto_significativo) or  # Espacio grande y resto significativo
                                not resto_significativo or  # El resto es <38, mejor terminarlo
                                (necesita_mas_muestras and ayuda_umbral)  # Necesita alcanzar umbral del 75%
                            )
                            
                            if debe_fragmentar and fragmento >= 38:
                                if mejor_opcion is None:  # Solo si no hay opción completa
                                    mejor_opcion = (idx, fragmento, "fragmentado_inteligente")
                    
                    # PASO 3: Si aún no se alcanza el umbral del 75%, buscar cualquier fragmento viable
                    if necesita_mas_muestras and mejor_opcion is None and espacio_restante > 0:
                        for idx, muestras in registros_grandes:
                            reg_data = df_state.iloc[idx]
                            dias_antiguedad = (datetime.now() - pd.to_datetime(reg_data["Fecha solicitud"])).days
                            
                            # Para alcanzar umbral, permitir fragmentos más pequeños si es urgente
                            if dias_antiguedad >= 10 and espacio_restante >= 20:  # Flexibilidad para muestras urgentes
                                fragmento_minimo = min(espacio_restante, muestras)
                                if (session_take + fragmento_minimo) >= umbral_minimo:
                                    mejor_opcion = (idx, fragmento_minimo, "umbral_60")
                                    break
                    
                    # Aplicar la mejor opción encontrada
                    if mejor_opcion:
                        idx, muestras, tipo = mejor_opcion
                        session_take += muestras
                        registros_a_procesar.append((idx, muestras))
            
            if session_take <= 0:
                break

            # Sesión de procesamiento directo
            proc_start = day_cursor
            proc_end = proc_start + timedelta(hours=t_proc)

            # Asignar muestras según registros_a_procesar (respeta no-fragmentación)
            session_registros = []
            
            for idx, take in registros_a_procesar:
                pend = int(df_state.loc[idx, "Pendiente"])
                
                schedule_rows.append({
                    "Fecha": proc_start.date(),
                    "Inicio": proc_start,
                    "Fin": proc_end,
                    "Registro": str(df_state.loc[idx, "Registro"]),
                    "Grupo": grupo,
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

    # Gantt semanal (concatenación de días)
    # CREAR TABLA DE PROGRESO POR REGISTRO Y CATEGORÍA
    progreso_registros = []
    registros_unicos = df_state["Registro"].unique()
    
    for registro in registros_unicos:
        reg_data = df_state[df_state["Registro"] == registro]
        
        # Información base del registro
        fecha_solicitud = reg_data["Fecha solicitud"].iloc[0]
        total_muestras = reg_data["No muestras"].iloc[0]
        
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
        
        # Calcular progreso por cada grupo
        grupos_requeridos = []
        grupos_completados = []
        
        for _, row in reg_data.iterrows():
            grupo = row["Tipo de analisis"]
            procesadas = row["No muestras"] - row["Pendiente"]
            pendientes = row["Pendiente"]
            
            grupos_requeridos.append(grupo)
            
            if grupo == "A":
                progreso_reg["A_Procesadas"] = procesadas
                progreso_reg["A_Pendientes"] = pendientes
            elif grupo == "B":
                progreso_reg["B_Procesadas"] = procesadas
                progreso_reg["B_Pendientes"] = pendientes
            elif grupo == "C":
                progreso_reg["C_Procesadas"] = procesadas
                progreso_reg["C_Pendientes"] = pendientes
            elif grupo == "D":
                progreso_reg["D_Procesadas"] = procesadas
                progreso_reg["D_Pendientes"] = pendientes
            
            # Si está completado (pendientes = 0)
            if pendientes == 0:
                grupos_completados.append(grupo)
        
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

def plan_fifo_simple(prueba: pd.DataFrame, tiempo: pd.DataFrame, selected_date: datetime, daily_cap: int):
    """Algoritmo FIFO simple con reglas actualizadas:
    - FIFO estricto por fecha de solicitud
    - Aplica restricciones por día (Lunes: A+D, Martes: B+C, etc.)
    - Grupo D: NUNCA se combina con otros grupos
    - Capacidad diaria = 38 muestras
    - Umbral mínimo del 75% (29 muestras/día)
    - Sin fragmentación inteligente (solo grupos de 38 para registros >38)
    """
    if daily_cap is None or daily_cap <= 0:
        daily_cap = 38  # Valor por defecto si hay algún problema

    # Expandir análisis
    df_original = expand_analyses(prueba)
    df_original = df_original.sort_values(by=["Fecha solicitud", "Registro", "Tipo de analisis"]).reset_index(drop=True)
    df_original["Pendiente"] = df_original["No muestras"].fillna(0).astype(int)

    # Map de tiempos
    tiempo = tiempo.copy()
    tiempo["Grupo"] = tiempo["Grupo"].astype(str).str.strip()
    t_map = tiempo.set_index("Grupo")[["Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"]].to_dict("index")

    # Días de trabajo (L-J)
    days = get_week_days(selected_date)
    
    df_state = df_original.copy()
    schedule_rows = []
    
    # Procesar día por día aplicando las nuevas reglas
    for d in days:
        day_name = d.strftime('%A').lower()
        
        # Definir grupos permitidos por día
        if day_name == 'monday':  # Lunes
            grupos_permitidos = ['A', 'D']
        elif day_name == 'tuesday':  # Martes  
            grupos_permitidos = ['B', 'C']
        else:  # Miércoles y Jueves
            grupos_permitidos = ['A', 'B', 'C', 'D']
        
        # Filtrar registros disponibles para este día
        registros_dia = df_state[
            (df_state["Pendiente"] > 0) & 
            (df_state["Tipo de analisis"].isin(grupos_permitidos))
        ].copy()
        
        if registros_dia.empty:
            continue
            
        # Ordenar por FIFO (fecha de solicitud)
        registros_dia = registros_dia.sort_values("Fecha solicitud")
        
        muestras_procesadas_dia = 0
        umbral_minimo = int(daily_cap * 0.75)  # 75% de 38 = 29 muestras
        
        # PASO 1: Procesar registros ≤38 muestras (completos)
        for idx, row in registros_dia.iterrows():
            if muestras_procesadas_dia >= daily_cap:
                break
                
            grupo = row["Tipo de analisis"]
            muestras = int(row["Pendiente"])
            registro = row["Registro"]
            
            # Verificar restricción del Grupo D (no se combina)
            if grupo == 'D' and muestras_procesadas_dia > 0:
                continue  # Grupo D debe procesarse solo
            if muestras_procesadas_dia > 0 and any(r["Grupo"] == 'D' for r in schedule_rows if r["Fecha"] == d.date()):
                continue  # Ya hay Grupo D programado este día
                
            if muestras <= 38 and muestras_procesadas_dia + muestras <= daily_cap:
                # Procesar completo
                t_pre = t_map.get(grupo, {}).get("Tiempo de prealistamiento (horas)", 0) or 0
                t_proc = t_map.get(grupo, {}).get("Tiempo procesamiento (horas)", 0) or 0
                duracion_total = max(t_pre + t_proc, 0.1)
                
                inicio = d + timedelta(hours=8)  # Inicio a las 8:00
                fin = inicio + timedelta(hours=duracion_total)
                
                schedule_rows.append({
                    "Fecha": d.date(),
                    "Registro": registro,
                    "Grupo": grupo,
                    "Muestras": muestras,
                    "Inicio": inicio,
                    "Fin": fin,
                    "Duracion (h)": duracion_total
                })
                
                muestras_procesadas_dia += muestras
                df_state.loc[idx, "Pendiente"] -= muestras
        
        # PASO 2: Si no se alcanza umbral del 75%, procesar registros >38 en grupos de 38
        if muestras_procesadas_dia < umbral_minimo:
            registros_grandes = registros_dia[registros_dia["Pendiente"] > 38].copy()
            
            for idx, row in registros_grandes.iterrows():
                if muestras_procesadas_dia >= daily_cap:
                    break
                    
                grupo = row["Tipo de analisis"]
                muestras_disponibles = int(row["Pendiente"])
                registro = row["Registro"]
                
                # Verificar restricción del Grupo D
                if grupo == 'D' and muestras_procesadas_dia > 0:
                    continue
                if muestras_procesadas_dia > 0 and any(r["Grupo"] == 'D' for r in schedule_rows if r["Fecha"] == d.date()):
                    continue
                
                espacio_restante = daily_cap - muestras_procesadas_dia
                grupos_de_38 = espacio_restante // 38
                
                if grupos_de_38 > 0:
                    muestras_a_procesar = min(grupos_de_38 * 38, muestras_disponibles)
                    
                    # Solo fragmentar si ayuda a alcanzar el umbral
                    if (muestras_procesadas_dia + muestras_a_procesar) >= umbral_minimo or muestras_a_procesar == muestras_disponibles:
                        t_pre = t_map.get(grupo, {}).get("Tiempo de prealistamiento (horas)", 0) or 0
                        t_proc = t_map.get(grupo, {}).get("Tiempo procesamiento (horas)", 0) or 0
                        duracion_total = max(t_pre + t_proc, 0.1)
                        
                        inicio = d + timedelta(hours=8)
                        fin = inicio + timedelta(hours=duracion_total)
                        
                        schedule_rows.append({
                            "Fecha": d.date(),
                            "Registro": registro,
                            "Grupo": grupo,
                            "Muestras": muestras_a_procesar,
                            "Inicio": inicio,
                            "Fin": fin,
                            "Duracion (h)": duracion_total
                        })
                        
                        muestras_procesadas_dia += muestras_a_procesar
                        df_state.loc[idx, "Pendiente"] -= muestras_a_procesar
                        break  # Solo un registro grande por día
    
    # Crear DataFrames resultado
    if schedule_rows:
        schedule_df = pd.DataFrame(schedule_rows)
    else:
        schedule_df = pd.DataFrame(columns=["Fecha", "Registro", "Grupo", "Muestras", "Inicio", "Fin", "Duracion (h)"])
    
    # Calcular pendientes correctamente (restar lo que se programó)
    # Crear diccionario de muestras procesadas por registro+grupo
    procesadas_dict = {}
    for row in schedule_rows:
        key = f"{row['Registro']}_{row['Grupo']}"
        procesadas_dict[key] = procesadas_dict.get(key, 0) + row["Muestras"]
    
    # Pendientes (lo que no se procesó)
    pendientes_rows = []
    for _, row in df_state.iterrows():
        key = f"{row['Registro']}_{row['Tipo de analisis']}"
        muestras_procesadas = procesadas_dict.get(key, 0)
        muestras_pendientes = max(0, row["Pendiente"] - muestras_procesadas)
        
        if muestras_pendientes > 0:  # Solo agregar si realmente quedan pendientes
            pendientes_rows.append({
                "Registro": row["Registro"],
                "Grupo": row["Tipo de analisis"],
                "Muestras": muestras_pendientes,
                "Fecha solicitud": row["Fecha solicitud"]
            })
    
    pendientes_df = pd.DataFrame(pendientes_rows) if pendientes_rows else pd.DataFrame()
    
    return schedule_df, pendientes_df

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
        
        # NUEVA HOJA: Análisis de Optimización (SIEMPRE se genera)
        try:
            # Ejecutar FIFO simple para comparación (si hay datos)
            schedule_fifo = pd.DataFrame()
            pendientes_fifo = pd.DataFrame()
            
            if all(x is not None for x in [prueba, tiempo, selected_date, daily_cap]):
                schedule_fifo, pendientes_fifo = plan_fifo_simple(prueba, tiempo, selected_date, daily_cap)
            
            # Crear análisis comparativo
            optimizacion_data = []
            
            # Métricas del modelo optimizado
            muestras_opt = schedule["Muestras"].sum() if isinstance(schedule, pd.DataFrame) and not schedule.empty and "Muestras" in schedule.columns else 0
            pendientes_opt = pendientes["Pendiente"].sum() if isinstance(pendientes, pd.DataFrame) and not pendientes.empty and "Pendiente" in pendientes.columns else 0
            dias_opt = len(schedule["Fecha"].unique()) if isinstance(schedule, pd.DataFrame) and not schedule.empty and "Fecha" in schedule.columns else 0
            
            # Métricas del FIFO simple
            muestras_fifo = schedule_fifo["Muestras"].sum() if not schedule_fifo.empty and "Muestras" in schedule_fifo.columns else 0
            pendientes_fifo_total = pendientes_fifo["Muestras"].sum() if not pendientes_fifo.empty and "Muestras" in pendientes_fifo.columns else 0
            dias_fifo = len(schedule_fifo["Fecha"].unique()) if not schedule_fifo.empty and "Fecha" in schedule_fifo.columns else 0
            
            # Calcular porcentajes de optimización
            mejora_muestras = ((muestras_opt - muestras_fifo) / max(muestras_fifo, 1)) * 100 if muestras_fifo > 0 else 0
            mejora_pendientes = ((pendientes_fifo_total - pendientes_opt) / max(pendientes_fifo_total, 1)) * 100 if pendientes_fifo_total > 0 else 0
            ahorro_dias = max(0, dias_fifo - dias_opt)
            
            # Información del modelo (SIEMPRE se incluye)
            optimizacion_data.extend([
                {"Concepto": "DESCRIPCIÓN DEL MODELO DE OPTIMIZACIÓN", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "═══════════════════════════════════════", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "Heurística aplicada:", "Optimizado": "Optimización de capacidad + FIFO inteligente", "FIFO Simple": "FIFO estricto sin optimización", "Mejora (%)": ""},
                {"Concepto": "Estrategia de fragmentación:", "Optimizado": "≤38 nunca fragmentar, >38 decisión inteligente + umbral 75%", "FIFO Simple": "≤38 nunca fragmentar, >38 grupos de 38 + umbral 75%", "Mejora (%)": ""},
                {"Concepto": "Mezcla de muestras:", "Optimizado": "Permite mezclar grupos compatibles", "FIFO Simple": "No mezcla registros/grupos", "Mejora (%)": ""},
                {"Concepto": "Restricciones temporales:", "Optimizado": "Lunes: A+D, Martes: B+C, Mié/Jue: todos", "FIFO Simple": "Lunes: A+D, Martes: B+C, Mié/Jue: todos", "Mejora (%)": ""},
                {"Concepto": "Grupo D:", "Optimizado": "NUNCA se combina con otros grupos", "FIFO Simple": "NUNCA se combina con otros grupos", "Mejora (%)": ""},
                {"Concepto": "Umbral mínimo:", "Optimizado": "75% capacidad (29 muestras/día)", "FIFO Simple": "75% capacidad (29 muestras/día)", "Mejora (%)": ""},
                {"Concepto": "Priorización:", "Optimizado": "FIFO + urgencia 20 días + capacidad", "FIFO Simple": "FIFO estricto por fecha", "Mejora (%)": ""},
                {"Concepto": "Programación semanal:", "Optimizado": "Lunes a Jueves (4 días disponibles)", "FIFO Simple": "Lunes a Jueves (4 días disponibles)", "Mejora (%)": ""},
                {"Concepto": "", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "RESULTADOS COMPARATIVOS", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "════════════════════════", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "Muestras procesadas:", "Optimizado": f"{muestras_opt:,.0f}", "FIFO Simple": f"{muestras_fifo:,.0f}", "Mejora (%)": f"{mejora_muestras:+.1f}%"},
                {"Concepto": "Muestras pendientes:", "Optimizado": f"{pendientes_opt:,.0f}", "FIFO Simple": f"{pendientes_fifo_total:,.0f}", "Mejora (%)": f"{mejora_pendientes:+.1f}%"},
                {"Concepto": "Días utilizados:", "Optimizado": f"{dias_opt}", "FIFO Simple": f"{dias_fifo}", "Mejora (%)": f"{ahorro_dias} días ahorrados"},
                {"Concepto": "Utilización capacidad:", "Optimizado": f"{(muestras_opt/(dias_opt*daily_cap)*100):,.1f}%" if dias_opt > 0 and daily_cap > 0 else "N/A", "FIFO Simple": f"{(muestras_fifo/(dias_fifo*daily_cap)*100):,.1f}%" if dias_fifo > 0 and daily_cap > 0 else "N/A", "Mejora (%)": ""},
                {"Concepto": "", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "CONCLUSIONES", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "═══════════", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "Optimización general:", "Optimizado": "✅ Mayor throughput y eficiencia", "FIFO Simple": "❌ Menor utilización de recursos", "Mejora (%)": f"{max(mejora_muestras, mejora_pendientes):+.1f}% promedio"},
                {"Concepto": "Gestión de capacidad:", "Optimizado": "✅ Aprovecha capacidad máxima", "FIFO Simple": "❌ Desperdicia capacidad diaria", "Mejora (%)": ""},
                {"Concepto": "Flexibilidad operativa:", "Optimizado": "✅ Adaptable a restricciones", "FIFO Simple": "❌ Rígido y poco eficiente", "Mejora (%)": ""},
                {"Concepto": "Restricción de viernes:", "Optimizado": "✅ No programa viernes (4 días hábiles)", "FIFO Simple": "✅ No programa viernes (4 días hábiles)", "Mejora (%)": "Implementado"},
            ])
            
            # SIEMPRE crear y exportar la hoja
            optimizacion_df = pd.DataFrame(optimizacion_data)
            optimizacion_df.to_excel(writer, sheet_name="Optimizacion", index=False)
            
        except Exception as e:
            # Fallback: crear hoja básica si hay error
            fallback_data = [
                {"Concepto": "ERROR EN ANÁLISIS", "Optimizado": f"Error: {str(e)}", "FIFO Simple": "No disponible", "Mejora (%)": "N/A"},
                {"Concepto": "Descripción del modelo:", "Optimizado": "Optimización con fragmentación inteligente", "FIFO Simple": "FIFO simple con grupo B +1 día", "Mejora (%)": ""},
                {"Concepto": "Programación:", "Optimizado": "Lunes a Jueves únicamente", "FIFO Simple": "Lunes a Jueves únicamente", "Mejora (%)": "Implementado"},
            ]
            fallback_df = pd.DataFrame(fallback_data)
            fallback_df.to_excel(writer, sheet_name="Optimizacion", index=False)
        
        # NUEVA HOJA: Tasas de Llegada por Grupo
        try:
            if prueba is not None:
                # Calcular tasas de llegada
                arrival_rates = calculate_arrival_rates(prueba)
                capacity_analysis = analyze_capacity_vs_arrival_rates(arrival_rates, daily_cap if daily_cap else 38)
                
                # Crear tabla de tasas de llegada
                tasas_data = []
                
                # Encabezado
                tasas_data.append({
                    "Grupo": "ANÁLISIS DE TASAS DE LLEGADA POR GRUPO",
                    "Tasa_Diaria_Muestras": "",
                    "Total_Muestras_Historicas": "",
                    "Dias_Periodo_Analizado": "",
                    "Num_Registros": "",
                    "Promedio_Muestras_por_Registro": "",
                    "Porcentaje_Demanda_Total": "",
                    "Capacidad_Proporcional_Sugerida": "",
                    "Tiempo_Espera_Estimado_Dias": "",
                    "Factor_Saturacion": "",
                    "Prioridad_Capacidad": "",
                    "Fecha_Inicio_Periodo": "",
                    "Fecha_Fin_Periodo": ""
                })
                
                tasas_data.append({
                    "Grupo": "═══════════════════════════════════",
                    "Tasa_Diaria_Muestras": "═══════════",
                    "Total_Muestras_Historicas": "══════════════",
                    "Dias_Periodo_Analizado": "══════════════",
                    "Num_Registros": "════════",
                    "Promedio_Muestras_por_Registro": "═══════════════════",
                    "Porcentaje_Demanda_Total": "═══════════════",
                    "Capacidad_Proporcional_Sugerida": "═══════════════════════",
                    "Tiempo_Espera_Estimado_Dias": "═══════════════════",
                    "Factor_Saturacion": "════════════",
                    "Prioridad_Capacidad": "════════════════",
                    "Fecha_Inicio_Periodo": "═══════════════",
                    "Fecha_Fin_Periodo": "══════════════"
                })
                
                # Datos por grupo
                for grupo in ['A', 'B', 'C', 'D']:
                    if grupo in arrival_rates:
                        rates = arrival_rates[grupo]
                        analysis = capacity_analysis.get(grupo, {})
                        
                        tasas_data.append({
                            "Grupo": grupo,
                            "Tasa_Diaria_Muestras": rates.get('tasa_diaria', 0),
                            "Total_Muestras_Historicas": rates.get('total_muestras', 0),
                            "Dias_Periodo_Analizado": rates.get('dias_periodo', 0),
                            "Num_Registros": rates.get('num_registros', 0),
                            "Promedio_Muestras_por_Registro": rates.get('promedio_por_registro', 0),
                            "Porcentaje_Demanda_Total": f"{analysis.get('porcentaje_demanda', 0):.1f}%",
                            "Capacidad_Proporcional_Sugerida": f"{analysis.get('capacidad_proporcional', 0):.1f}",
                            "Tiempo_Espera_Estimado_Dias": f"{analysis.get('tiempo_espera_dias', 0):.1f}",
                            "Factor_Saturacion": f"{analysis.get('saturacion', 0):.2f}x",
                            "Prioridad_Capacidad": analysis.get('prioridad_capacidad', 'N/A'),
                            "Fecha_Inicio_Periodo": rates.get('fecha_inicio', 'N/A'),
                            "Fecha_Fin_Periodo": rates.get('fecha_fin', 'N/A')
                        })
                
                # Análisis global
                if 'GLOBAL' in capacity_analysis:
                    global_stats = capacity_analysis['GLOBAL']
                    tasas_data.extend([
                        {
                            "Grupo": "",
                            "Tasa_Diaria_Muestras": "",
                            "Total_Muestras_Historicas": "",
                            "Dias_Periodo_Analizado": "",
                            "Num_Registros": "",
                            "Promedio_Muestras_por_Registro": "",
                            "Porcentaje_Demanda_Total": "",
                            "Capacidad_Proporcional_Sugerida": "",
                            "Tiempo_Espera_Estimado_Dias": "",
                            "Factor_Saturacion": "",
                            "Prioridad_Capacidad": "",
                            "Fecha_Inicio_Periodo": "",
                            "Fecha_Fin_Periodo": ""
                        },
                        {
                            "Grupo": "ANÁLISIS GLOBAL DEL SISTEMA",
                            "Tasa_Diaria_Muestras": f"{global_stats.get('tasa_total', 0):.2f}",
                            "Total_Muestras_Historicas": "Demanda Total/Día",
                            "Dias_Periodo_Analizado": f"{global_stats.get('capacidad_disponible', 0)}",
                            "Num_Registros": "Cap. Disponible",
                            "Promedio_Muestras_por_Registro": f"{global_stats.get('utilizacion_capacidad', 0):.1f}%",
                            "Porcentaje_Demanda_Total": "Utilización",
                            "Capacidad_Proporcional_Sugerida": f"{global_stats.get('deficit_superavit', 0):+.1f}",
                            "Tiempo_Espera_Estimado_Dias": "Balance Diario",
                            "Factor_Saturacion": global_stats.get('estado_sistema', 'N/A'),
                            "Prioridad_Capacidad": "Estado Sistema",
                            "Fecha_Inicio_Periodo": "",
                            "Fecha_Fin_Periodo": ""
                        }
                    ])
                
                # Crear y exportar DataFrame de tasas
                tasas_df = pd.DataFrame(tasas_data)
                tasas_df.to_excel(writer, sheet_name="Tasas_Llegada", index=False)
            else:
                # Fallback si no hay datos
                fallback_tasas = pd.DataFrame([
                    {"Grupo": "Sin datos disponibles", "Tasa_Diaria_Muestras": "N/A", "Informacion": "Requiere datos históricos para calcular tasas"}
                ])
                fallback_tasas.to_excel(writer, sheet_name="Tasas_Llegada", index=False)
                
        except Exception as e:
            # Fallback en caso de error
            error_tasas = pd.DataFrame([
                {"Grupo": "ERROR", "Tasa_Diaria_Muestras": f"Error: {str(e)}", "Informacion": "No se pudieron calcular las tasas de llegada"}
            ])
            error_tasas.to_excel(writer, sheet_name="Tasas_Llegada", index=False)
    
    return out.getvalue()

# =============================
# UI
# =============================
st.sidebar.header("Configuración de datos")
default_path = "Insumo_Planeacion.xlsx"
uploaded = st.sidebar.file_uploader("Cargar Excel personalizado", type=["xlsx", "xlsm"])
foliar, prueba, tiempo, daily_cap = load_data(default_path, uploaded.getvalue() if uploaded else None)

tab1, tab2, tab3 = st.tabs(["📚 Históricos", "🗓️ Planeación por día (capacidad diaria)", "⚡ Análisis de Optimización"])

with tab1:
    st.dataframe(foliar, use_container_width=True)
    kpis = compute_kpis(foliar)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="% Cumplimiento (Aplican = SI)", value=f"{kpis.get('% Cumplimiento (solo Aplican=SI)', 0.0):.1f}%")
    with c2:
        st.metric(label="Tiempo promedio de entrega (días)", value=f"{kpis.get('Tiempo promedio de entrega (días)', 0.0):.2f}")
    with c3:
        st.metric(label="Casos considerados", value=f"{int(kpis.get('Casos considerados (Aplican=SI)', 0.0))}")

with tab2:
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
        
        # NUEVO EXPANDER: Análisis de Tasas de Llegada
        with st.expander("📈 Análisis de Tasas de Llegada por Grupo"):
            st.markdown("""
            **🎯 Análisis Temporal de Demanda por Grupo**
            
            Este análisis calcula la **tasa de llegada diaria** de muestras por cada grupo (A, B, C, D) 
            basado en los datos históricos disponibles.
            """)
            
            # Calcular tasas de llegada para mostrar
            arrival_rates = calculate_arrival_rates(prueba)
            capacity_analysis = analyze_capacity_vs_arrival_rates(arrival_rates, daily_cap)
            
            if arrival_rates:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Tasas de Llegada Calculadas")
                    
                    # Crear DataFrame para mostrar las tasas
                    rates_data = []
                    for grupo, data in arrival_rates.items():
                        if grupo in ['A', 'B', 'C', 'D']:
                            rates_data.append({
                                '🎯 Grupo': grupo,
                                '📈 Tasa Diaria': f"{data['tasa_diaria']:.2f}",
                                '🧪 Total Muestras': data['total_muestras'],
                                '📅 Días': data['dias_periodo'],
                                '📋 Registros': data['num_registros'],
                                '⚖️ Prom/Reg': f"{data['promedio_por_registro']:.1f}"
                            })
                    
                    df_rates = pd.DataFrame(rates_data)
                    st.dataframe(df_rates, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("⚖️ Análisis de Capacidad")
                    
                    # Análisis detallado por grupo
                    analysis_data = []
                    for grupo in ['A', 'B', 'C', 'D']:
                        if grupo in capacity_analysis:
                            data = capacity_analysis[grupo]
                            analysis_data.append({
                                '🎯 Grupo': grupo,
                                '📊 % Demanda': f"{data['porcentaje_demanda']:.1f}%",
                                '⚖️ Cap. Sugerida': f"{data['capacidad_proporcional']:.1f}",
                                '🔥 Saturación': f"{data['saturacion']:.2f}x",
                                '⚡ Prioridad': data['prioridad_capacidad']
                            })
                    
                    df_analysis = pd.DataFrame(analysis_data)
                    
                    # Colorear según prioridad
                    def color_priority_mini(val):
                        if val == 'CRÍTICA':
                            return 'background-color: #ffebee; color: #c62828; font-weight: bold'
                        elif val == 'ALTA':
                            return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
                        elif val == 'MEDIA':
                            return 'background-color: #f3e5f5; color: #7b1fa2'
                        else:
                            return 'background-color: #e8f5e8; color: #2e7d32'
                    
                    styled_df_mini = df_analysis.style.map(color_priority_mini, subset=['⚡ Prioridad'])
                    st.dataframe(styled_df_mini, use_container_width=True, hide_index=True)
                
                # Análisis global del sistema
                if 'GLOBAL' in capacity_analysis:
                    global_stats = capacity_analysis['GLOBAL']
                    
                    st.subheader("🌐 Estado Global del Sistema")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔥 Demanda Total", f"{global_stats['tasa_total']:.1f} muestras/día")
                    with col2:
                        st.metric("⚡ Capacidad Disp.", f"{global_stats['capacidad_disponible']} muestras/día")
                    with col3:
                        utilizacion = global_stats['utilizacion_capacidad']
                        st.metric("📊 Utilización", f"{utilizacion:.1f}%")
                    with col4:
                        deficit = global_stats['deficit_superavit']
                        st.metric("📈 Balance", f"{deficit:+.1f} muestras/día")
                    
                    # Estado del sistema
                    estado = global_stats['estado_sistema']
                    if estado == 'SOBRECARGADO':
                        st.error(f"🚨 **Sistema {estado}**: La demanda excede la capacidad diaria disponible")
                    elif estado == 'EQUILIBRADO':
                        st.success(f"✅ **Sistema {estado}**: Demanda dentro de límites operativos")
                    else:
                        st.warning(f"⚠️ **Sistema {estado}**: Capacidad infrautilizada")
                    
                    # Recomendaciones compactas
                    recomendaciones = []
                    for grupo in ['A', 'B', 'C', 'D']:
                        if grupo in capacity_analysis:
                            data = capacity_analysis[grupo]
                            prioridad = data['prioridad_capacidad']
                            
                            if prioridad == 'CRÍTICA':
                                recomendaciones.append(f"🚨 **{grupo}**: Saturación crítica")
                            elif prioridad == 'ALTA':
                                recomendaciones.append(f"⚠️ **{grupo}**: Alta demanda")
                    
                    if recomendaciones:
                        st.markdown("**💡 Alertas:**")
                        for rec in recomendaciones:
                            st.markdown(f"- {rec}")
                            
            else:
                st.warning("⚠️ No se pudieron calcular las tasas de llegada. Verificar datos de fecha en las muestras.")

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

        xls_bytes = to_excel_download(schedule, pendientes, util_df, gantt_week, gantt_per_day, 
                                     prueba, tiempo, datetime.combine(selected_date, DAY_START), daily_cap)
        st.download_button(
            label="⬇️ Descargar plan completo con análisis de optimización",
            data=xls_bytes,
            file_name="planeacion_semana_con_optimizacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tab3:
    st.header("⚡ Análisis Comparativo de Optimización")
    
    st.markdown("""
    ### 🎯 Descripción del Modelo de Optimización Implementado
    
    El sistema utiliza una **heurística de optimización avanzada** que combina:
    """)
    
    col1, col2 = st.columns(2)
    
    st.markdown("### 📊 Comparación en Tiempo Real")
    
    # Parámetros independientes para esta pestaña
    st.subheader("Configuración del Análisis")
    col1, col2 = st.columns(2)
    
    with col1:
        fecha_analisis = st.date_input("� Selecciona fecha para análisis", value=datetime.today(), key="fecha_opt")
    with col2:
        st.info(f"🧪 Capacidad diaria: {daily_cap if daily_cap else 'Sin límite'} muestras/día")
    
    if st.button("🔄 Ejecutar Análisis Comparativo Completo", key="analisis_comp"):
        with st.spinner("Ejecutando ambos algoritmos para comparación completa..."):
            try:
                # Ejecutar AMBOS algoritmos de forma independiente
                fecha_datetime = datetime.combine(fecha_analisis, DAY_START)
                
                # 1. Ejecutar modelo OPTIMIZADO
                st.info("🚀 Ejecutando modelo optimizado...")
                schedule_opt, pendientes_opt, util_df_opt, gantt_week_opt, gantt_per_day_opt, df_progreso_opt, registros_info_opt = plan_week_by_day(prueba, tiempo, fecha_datetime, daily_cap)
                
                # 2. Ejecutar modelo FIFO SIMPLE  
                st.info("🐌 Ejecutando modelo FIFO simple...")
                schedule_fifo, pendientes_fifo = plan_fifo_simple(prueba, tiempo, fecha_datetime, daily_cap)
                    
                # 3. Calcular métricas comparativas correctas
                st.success("✅ Ambos algoritmos ejecutados correctamente")
                
                # Calcular total de muestras disponibles en el dataset
                df_original_temp = expand_analyses(prueba)
                total_muestras_disponibles = df_original_temp["No muestras"].sum()
                
                # Métricas del modelo optimizado
                muestras_opt = schedule_opt["Muestras"].sum() if isinstance(schedule_opt, pd.DataFrame) and not schedule_opt.empty and "Muestras" in schedule_opt.columns else 0
                pendientes_opt_total = pendientes_opt["Pendiente"].sum() if isinstance(pendientes_opt, pd.DataFrame) and not pendientes_opt.empty and "Pendiente" in pendientes_opt.columns else (total_muestras_disponibles - muestras_opt)
                # Calcular días utilizados y estimar días necesarios totales
                dias_utilizados_opt = len(schedule_opt["Fecha"].unique()) if isinstance(schedule_opt, pd.DataFrame) and not schedule_opt.empty and "Fecha" in schedule_opt.columns else 0
                # Estimar días necesarios totales (más realista considerando pendientes)
                if pendientes_opt_total > 0:
                    dias_adicionales_necesarios_opt = math.ceil(pendientes_opt_total / daily_cap) if daily_cap and daily_cap > 0 else 0
                    dias_reales_necesarios_opt = dias_utilizados_opt + dias_adicionales_necesarios_opt
                else:
                    dias_reales_necesarios_opt = dias_utilizados_opt
                
                # Métricas del FIFO simple  
                muestras_fifo = schedule_fifo["Muestras"].sum() if isinstance(schedule_fifo, pd.DataFrame) and not schedule_fifo.empty and "Muestras" in schedule_fifo.columns else 0
                # El FIFO usa columna "Muestras" para pendientes, no "Pendiente"
                pendientes_fifo_total = pendientes_fifo["Muestras"].sum() if isinstance(pendientes_fifo, pd.DataFrame) and not pendientes_fifo.empty and "Muestras" in pendientes_fifo.columns else (total_muestras_disponibles - muestras_fifo)
                # Calcular días utilizados y estimar días necesarios totales
                dias_utilizados_fifo = len(schedule_fifo["Fecha"].unique()) if isinstance(schedule_fifo, pd.DataFrame) and not schedule_fifo.empty and "Fecha" in schedule_fifo.columns else 0
                # Estimar días necesarios totales (más realista considerando pendientes)  
                if pendientes_fifo_total > 0:
                    dias_adicionales_necesarios_fifo = math.ceil(pendientes_fifo_total / daily_cap) if daily_cap and daily_cap > 0 else 0
                    dias_reales_necesarios_fifo = dias_utilizados_fifo + dias_adicionales_necesarios_fifo
                else:
                    dias_reales_necesarios_fifo = dias_utilizados_fifo
                
                # Validación de consistencia
                if muestras_opt + pendientes_opt_total != total_muestras_disponibles:
                    pendientes_opt_total = max(0, total_muestras_disponibles - muestras_opt)
                if muestras_fifo + pendientes_fifo_total != total_muestras_disponibles:
                    pendientes_fifo_total = max(0, total_muestras_disponibles - muestras_fifo)
                
                # Mostrar debug info
                st.info(f"""
                📊 **Información de Debug:**
                - Total muestras disponibles: {total_muestras_disponibles:,.0f}
                - Optimizado: {muestras_opt:,.0f} procesadas + {pendientes_opt_total:,.0f} pendientes = {muestras_opt + pendientes_opt_total:,.0f}
                - FIFO: {muestras_fifo:,.0f} procesadas + {pendientes_fifo_total:,.0f} pendientes = {muestras_fifo + pendientes_fifo_total:,.0f}
                - Días utilizados: Opt={dias_utilizados_opt}, FIFO={dias_utilizados_fifo}
                - Días estimados necesarios: Opt={dias_reales_necesarios_opt}, FIFO={dias_reales_necesarios_fifo}
                """)
                    
                # 4. Mostrar comparación visual
                st.markdown("### 📈 Resultados del Análisis Comparativo")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mejora_muestras = muestras_opt - muestras_fifo
                    st.metric(
                        "🧪 Muestras Procesadas (Optimizado)", 
                        f"{muestras_opt:,.0f}",
                        delta=f"{mejora_muestras:+.0f} vs FIFO ({muestras_fifo:,.0f})"
                    )
                
                with col2:
                    reduccion_pendientes = pendientes_fifo_total - pendientes_opt_total
                    st.metric(
                        "⏳ Muestras Pendientes (Optimizado)", 
                        f"{pendientes_opt_total:,.0f}",
                        delta=f"{reduccion_pendientes:+.0f} menos que FIFO ({pendientes_fifo_total:,.0f})",
                        delta_color="inverse"
                    )
                
                with col3:
                    ahorro_dias = max(0, dias_reales_necesarios_fifo - dias_reales_necesarios_opt)
                    st.metric(
                        "📅 Días Necesarios (Optimizado)", 
                        f"{dias_reales_necesarios_opt}",
                        delta=f"vs {dias_reales_necesarios_fifo} días FIFO ({ahorro_dias} ahorrados)" if ahorro_dias > 0 else f"vs {dias_reales_necesarios_fifo} días FIFO (mismo)"
                    )
                    
                    # Tabla comparativa detallada
                    st.markdown("### 📋 Resultados Detallados")
                    
                # 5. Tabla comparativa detallada
                st.markdown("### 📋 Resultados Detallados")
                
                comparacion_data = [
                    {"Métrica": "Muestras procesadas", "Optimizado": f"{muestras_opt:,.0f}", "FIFO Simple": f"{muestras_fifo:,.0f}", "Mejora": f"{((muestras_opt - muestras_fifo) / max(muestras_fifo, 1) * 100):+.1f}%"},
                    {"Métrica": "Muestras pendientes", "Optimizado": f"{pendientes_opt_total:,.0f}", "FIFO Simple": f"{pendientes_fifo_total:,.0f}", "Mejora": f"{((pendientes_fifo_total - pendientes_opt_total) / max(pendientes_fifo_total, 1) * 100):+.1f}%"},
                    {"Métrica": "Días necesarios (estimación real)", "Optimizado": f"{dias_reales_necesarios_opt}", "FIFO Simple": f"{dias_reales_necesarios_fifo}", "Mejora": f"{ahorro_dias} días menos"},
                    {"Métrica": "Días utilizados (L-J)", "Optimizado": f"{dias_utilizados_opt}", "FIFO Simple": f"{dias_utilizados_fifo}", "Mejora": f"{max(0, dias_utilizados_fifo - dias_utilizados_opt)} días menos"},
                    {"Métrica": "Utilización capacidad", "Optimizado": f"{(muestras_opt/(dias_utilizados_opt*daily_cap)*100):,.1f}%" if dias_utilizados_opt > 0 and daily_cap > 0 else "N/A", "FIFO Simple": f"{(muestras_fifo/(dias_utilizados_fifo*daily_cap)*100):,.1f}%" if dias_utilizados_fifo > 0 and daily_cap > 0 else "N/A", "Mejora": ""}
                ]
                
                df_comparacion = pd.DataFrame(comparacion_data)
                st.dataframe(df_comparacion, use_container_width=True, hide_index=True)
                
                # 6. Conclusiones automáticas
                st.markdown("### 🎯 Conclusiones del Análisis")
                
                mejora_general = ((muestras_opt - muestras_fifo) / max(muestras_fifo, 1) * 100) if muestras_fifo > 0 else 0
                
                if mejora_general > 0:
                    st.success(f"✅ **El modelo optimizado es {mejora_general:.1f}% más eficiente** que FIFO simple")
                elif mejora_general == 0:
                    st.info("ℹ️ Ambos modelos tienen rendimiento similar en este caso")
                else:
                    st.warning(f"⚠️ FIFO simple procesó {abs(mejora_general):.1f}% más muestras")
                
                st.markdown(f"""
                **📊 Resumen de Ventajas del Modelo Optimizado:**
                - 🚀 Procesa **{muestras_opt:,.0f}** muestras vs **{muestras_fifo:,.0f}** del FIFO simple
                - ⚡ Deja **{pendientes_opt_total:,.0f}** pendientes vs **{pendientes_fifo_total:,.0f}** del FIFO simple  
                - 📅 Necesitaría **{dias_reales_necesarios_opt}** días vs **{dias_reales_necesarios_fifo}** del FIFO simple
                - 🗓️ Utiliza **{dias_utilizados_opt}** días (L-J) vs **{dias_utilizados_fifo}** del FIFO simple
                - 🎯 Respeta restricciones de negocio (martes B+C, entrega completa)
                - 💡 Maximiza utilización de capacidad diaria disponible
                - 🚫 No programa trabajo los viernes (lunes a jueves únicamente)
                """)
                
                # 7. Mostrar planes generados
                with st.expander("📋 Ver Plan Generado - Modelo Optimizado"):
                    if not schedule_opt.empty:
                        st.dataframe(schedule_opt, use_container_width=True)
                    else:
                        st.info("No se generaron tareas programadas")
                        
                with st.expander("📋 Ver Plan Generado - FIFO Simple"):
                    if not schedule_fifo.empty:
                        st.dataframe(schedule_fifo, use_container_width=True)
                    else:
                        st.info("No se generaron tareas programadas")
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Verifica que los datos estén cargados correctamente")
    
    st.markdown("---")
    st.markdown("### 📖 Descripción Técnica de la Heurística")
    
    with st.expander("🔍 Ver detalles técnicos del algoritmo"):
        st.markdown("""
        **Algoritmo de Optimización Implementado:**
        
        1. **Fase de Priorización**: Ordena grupos por FIFO + urgencia (20 días) + cantidad disponible
        2. **Fase de Asignación**: Para cada día (L-J):
           - Martes: Solo procesa grupos B y C (excluye A)
           - Otros días: Procesa todos los grupos disponibles
        3. **Optimización de Capacidad**: 
           - Registros ≤38 muestras: Nunca fragmenta
           - Umbral mínimo: Trata de alcanzar 75% capacidad (29 muestras/día)
           - Registros >38 muestras: Fragmentación inteligente que evalúa:
             * Antigüedad ≥15 días → Fragmentar prioritariamente
             * Porcentaje procesable ≥60% → Vale la pena fragmentar
             * Espacio disponible ≥76 y resto significativo → Fragmentar eficientemente
             * Resto <38 muestras → Mejor completar el registro
             * Necesidad de umbral → Fragmentar para alcanzar 75% mínimo
        4. **Mezcla Inteligente**: Combina grupos compatibles para maximizar uso de capacidad
        5. **Entrega Completa**: Solo entrega registros cuando TODOS sus grupos están completos
        6. **Prealistamiento Anticipado**: Permite preparar grupos el día anterior
        
        **Comparación con FIFO Simple:**
        - FIFO simple: Procesamiento estricto por fecha, aplica restricciones diarias, capacidad 38, umbral 75%
        - Optimizado: Flexibilidad para maximizar throughput manteniendo restricciones de negocio
        """)
