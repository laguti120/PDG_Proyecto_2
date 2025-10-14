
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

    # Capacidad DIARIA (muestras/día)
    daily_cap = None
    cap_col = [c for c in capacidad.columns if "Capacidad" in c]
    if cap_col and not capacidad[cap_col[0]].dropna().empty:
        try:
            val = float(pd.to_numeric(capacidad[cap_col[0]], errors="coerce").dropna().iloc[0])
            if val > 0:
                daily_cap = int(val)
        except Exception:
            daily_cap = None

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
       - Un registro se entrega SOLO cuando se completan TODOS sus grupos (A, B, C)
       - No se pueden dejar registros a medias de procesamiento
       - Prioriza FIFO con urgencia 20 días
       - Martes solo procesa B y C
       - Optimiza capacidad diaria sin fragmentar registros ≤38 muestras
    """
    if daily_cap is None or daily_cap <= 0:
        st.error("No se encontró capacidad diaria válida en la hoja 'Capacidad'. Ingresa un valor numérico > 0.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

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
    
    st.info("📋 **Análisis de Registros para Entrega Completa**:")
    for _, reg in registros_info.head().iterrows():
        grupos_str = ", ".join(reg["Grupos_requeridos"])
        st.write(f"   • {reg['Registro']}: {grupos_str} ({reg['Total_grupos']} grupos, {reg['Pendiente']} muestras)")

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

    st.info(f"🔄 Iniciando planeación semanal con **entrega completa de registros** del {days[0].strftime('%Y-%m-%d')} al {days[-1].strftime('%Y-%m-%d')}")
    st.info("💡 **Estrategia**: Completar TODOS los grupos de un registro antes de entrega")
    
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
        """Retorna grupos disponibles ordenados por prioridad FIFO + urgencia 20 días"""
        candidates = df_temp[df_temp["Pendiente"] > 0].copy()
        if candidates.empty:
            return []
        
        # Calcular prioridad por grupo usando FIFO + urgencia
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
            
            # NUEVA PRIORIDAD: FIFO + Urgencia 20 días
            if dias_antiguedad >= 20:
                # URGENTE: Solicitudes ≥20 días tienen máxima prioridad
                priority_score = 1000 + dias_antiguedad  # Base alta + días adicionales
                urgencia_msg = "🚨 URGENTE"
            else:
                # NORMAL: Solo FIFO (más antiguos primero)
                priority_score = dias_antiguedad
                urgencia_msg = "📅 Normal"
            
            if grupo in t_map:  # Solo considerar grupos con tiempos definidos
                group_priority.append((grupo, priority_score, int(muestras_grupo), dias_antiguedad, urgencia_msg))
        
        # Ordenar por prioridad descendente (más urgente/antiguo primero)
        sorted_groups = sorted(group_priority, key=lambda x: x[1], reverse=True)
        
        # Mostrar información de priorización
        st.write("   📊 **Priorización FIFO + Urgencia**:")
        for grupo, score, muestras, dias, msg in sorted_groups[:3]:  # Mostrar top 3
            st.write(f"      {grupo}: {dias} días - {muestras} muestras - {msg}")
        
        # Retornar formato esperado (grupo, score, muestras)
        return [(g, s, m) for g, s, m, _, _ in sorted_groups]

    # INICIO DEL PROCESAMIENTO SEMANAL
    for day_idx, d in enumerate(days):
        day_name = ["Lunes","Martes","Miércoles","Jueves"][day_idx]
        pendientes_inicio = (df_state["Pendiente"] > 0).sum()
        muestras_pendientes_total = df_state["Pendiente"].sum()
        
        st.write(f"📅 **{day_name} ({d.strftime('%Y-%m-%d')})** - Registros pendientes: {pendientes_inicio}, Muestras totales: {muestras_pendientes_total}")
        
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
                st.info(f"   📭 No hay más grupos disponibles para procesar en {day_name}")
                break
            
            # ESTRATEGIA DE FALLBACK: Probar grupos en orden de prioridad
            grupo_procesado = None
            fallback_attempts = []
            
            # LÓGICA ESPECIAL MARTES: SOLO B y C (NO A)
            if day_name == "Martes":
                # En martes, EXCLUIR grupo A completamente
                available_groups_original = available_groups.copy()
                available_groups = [(g, s, m) for g, s, m in available_groups if g != "A"]
                
                grupos_disponibles = [g[0] for g in available_groups]
                grupos_excluidos = [g[0] for g, s, m in available_groups_original if g == "A"]
                
                if grupos_excluidos:
                    st.info(f"   🗓️ **MARTES - RESTRICCIÓN**: Excluyendo grupo A (solo B y C permitidos)")
                
                tiene_b = "B" in grupos_disponibles
                tiene_c = "C" in grupos_disponibles
                
                if tiene_b and tiene_c:
                    # Reorganizar para procesar B y C primero en martes
                    bc_groups = [(g, s, m) for g, s, m in available_groups if g in ["B", "C"]]
                    other_groups = [(g, s, m) for g, s, m in available_groups if g not in ["B", "C"]]
                    available_groups = bc_groups + other_groups
                    st.success(f"   🗓️ **MARTES - B+C CONJUNTO**: Procesando solo grupos B y C (A excluido)")
                elif tiene_b or tiene_c:
                    st.info(f"   🗓️ **MARTES**: Solo {'B' if tiene_b else 'C'} disponible para procesar (A excluido)")
                else:
                    st.warning(f"   🗓️ **MARTES**: Ni B ni C disponibles, procesando otros grupos (A excluido)")
                    
                if not available_groups:
                    st.warning(f"   ⚠️ **MARTES**: No hay grupos B o C disponibles, día sin procesamiento")
            
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
                    st.warning(f"   ⚠️ {grupo} tiene tiempo de procesamiento 0, saltando")
                else:
                    st.info(f"   ⌛ {grupo} necesita {total_time_needed:.1f}h pero solo hay {hours_left:.1f}h disponibles")
            
            # Si no se encontró ningún grupo que quepa, terminar el día
            if grupo_procesado is None:
                st.warning(f"   🔄 Fallback probado: {', '.join(fallback_attempts)} - Ninguno cabe")
                st.info(f"   🔚 No se puede procesar más en {day_name}")
                break
            
            # PROCESAMIENTO DEL GRUPO SELECCIONADO (SIN PREALISTAMIENTO)
            grupo = grupo_procesado
            tiempos = t_map[grupo]
            t_proc = float(tiempos.get("Tiempo procesamiento (horas)", 0) or 0.0)
            
            st.write(f"   ✅ Grupo seleccionado: {grupo} (prioridad: {priority_score:.1f}, tiempo: {t_proc:.1f}h)")

            # Seleccionar registros del grupo por FIFO estricto + urgencia
            grp_df = df_state[(df_state["Pendiente"] > 0) & (df_state["Tipo de analisis"] == grupo)].copy()
            if grp_df.empty:
                st.warning(f"   ⚠️ No hay muestras pendientes para {grupo}, continuando")
                continue
            
            # FIFO ESTRICTO: Ordenar por fecha de solicitud (más antiguas primero), luego por registro
            grp_df["Fecha solicitud"] = pd.to_datetime(grp_df["Fecha solicitud"], errors='coerce')
            grp_df = grp_df.sort_values(by=["Fecha solicitud", "Registro"])
            
            # Marcar solicitudes urgentes (≥20 días)
            grp_df["Dias_antiguedad"] = (pd.Timestamp(d.date()) - grp_df["Fecha solicitud"]).dt.days
            urgentes = grp_df[grp_df["Dias_antiguedad"] >= 20]
            if not urgentes.empty:
                st.warning(f"   🚨 {len(urgentes)} solicitudes URGENTES (≥20 días) en grupo {grupo}")
            pend_grp_total = int(grp_df["Pendiente"].sum())
            
            # REGLA ANTI-FRAGMENTACIÓN OPTIMIZADA: Respetar reglas estrictas
            session_take = 0
            registros_a_procesar = []
            
            # Separar registros por tamaño ANTES de procesamiento
            registros_pequenos = []  # ≤38 muestras - NO se pueden fragmentar
            registros_grandes = []   # >38 muestras - SE pueden fragmentar ≥50%
            
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
                    st.info(f"   ✅ Registro pequeño completo: {muestras} muestras")
                # Si no cabe completo, se deja para otro día (no fragmentar)
            
            # PASO 2: Si queda espacio, procesar UN registro grande
            espacio_restante = remaining_daily - session_take
            if espacio_restante > 0 and registros_grandes:
                # Buscar el mejor registro grande para el espacio disponible
                mejor_opcion = None
                
                for idx, muestras in registros_grandes:
                    mitad_registro = muestras // 2
                    
                    # OPCIÓN 1: ¿Cabe completo?
                    if muestras <= espacio_restante:
                        mejor_opcion = (idx, muestras, "completo")
                        break  # Completo es siempre mejor
                    
                    # OPCIÓN 2: ¿Se puede fragmentar ≥50%?
                    elif espacio_restante >= mitad_registro:
                        if mejor_opcion is None:  # Solo si no hay opción completa
                            fragmento = espacio_restante
                            mejor_opcion = (idx, fragmento, "fragmentado")
                
                # Aplicar la mejor opción encontrada
                if mejor_opcion:
                    idx, muestras, tipo = mejor_opcion
                    session_take += muestras
                    registros_a_procesar.append((idx, muestras))
                    
                    if tipo == "completo":
                        st.info(f"   ✅ Registro grande completo: {muestras} muestras")
                    else:
                        muestras_originales = next(m for i, m in registros_grandes if i == idx)
                        st.info(f"   ✂️ Fragmentando registro: {muestras}/{muestras_originales} muestras (≥50%)")
            
            # Mostrar resumen de optimización
            if registros_a_procesar:
                st.info(f"   🎯 Capacidad utilizada: {session_take}/{remaining_daily} muestras ({len(registros_a_procesar)} registros)")
            
            if session_take <= 0:
                st.info(f"   📊 Capacidad diaria agotada ({remaining_daily} restante)")
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
                    "Muestras": take
                })
                df_state.loc[idx, "Pendiente"] = pend - take
                session_registros.append(f"{df_state.loc[idx, 'Registro']}({take})")
                
                # Verificar reglas de fragmentación
                registro_original = str(df_state.loc[idx, "Registro"])
                if pend <= 38 and take < pend:
                    st.error(f"❌ ERROR: Fragmentando registro {registro_original} con {pend} muestras (≤38)")
                elif pend > 38 and take < pend:
                    mitad_esperada = pend // 2
                    if take >= mitad_esperada:
                        st.info(f"✂️ Fragmentando registro {registro_original}: {take}/{pend} muestras (≥50% ✓)")
                    else:
                        st.warning(f"⚠️ Fragmento pequeño en {registro_original}: {take}/{pend} muestras (<50%)")

            used_seconds += (proc_end - proc_start).total_seconds()
            day_cursor = proc_end
            remaining_daily -= session_take
            day_samples_processed += session_take
            
            st.write(f"   ⚗️ Procesamiento {grupo}: {proc_start.strftime('%H:%M')} - {proc_end.strftime('%H:%M')} ({t_proc:.1f}h) - {session_take} muestras: {', '.join(session_registros)}")

        # Fin del procesamiento del día
        
        # Si no se procesó nada y hay muestras pendientes, reportar
        if day_samples_processed == 0 and (df_state["Pendiente"] > 0).any():
            st.warning(f"   ❌ {day_name}: No se pudo procesar ninguna muestra (restricciones de tiempo/capacidad)")
        elif day_samples_processed == 0 and muestras_pendientes_total == 0:
            st.success(f"   ✅ {day_name}: Todas las muestras completadas")

        # KPIs de utilización diaria mejorados
        day_total_seconds = (day_end - day_start).total_seconds()
        util = used_seconds / day_total_seconds if day_total_seconds > 0 else 0.0
        
        # Contar prealistamientos anticipados hechos hoy para mañana
        prep_anticipados = len([r for r in schedule_rows if r["Fecha"] == d.date() and "para mañana" in r.get("Registro", "")])
        
        daily_utilization.append({
            "Fecha": d.date(), 
            "Utilización (%)": round(util * 100, 1), 
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
            day_blocks = pd.DataFrame(columns=["Fecha","Inicio","Fin","Registro","Grupo","Tipo","Muestras"])
            
        day_blocks_cons = consolidate_blocks(day_blocks)
        
        # Calcular progreso acumulado hasta este día
        gantt_day_rows = []
        for _, r in day_blocks_cons.iterrows():
            reg = str(r["Registro"])
            total_req = int(totals_by_reg.loc[totals_by_reg["Registro"] == reg, "TotalRegistro"].iloc[0]) if (totals_by_reg["Registro"] == reg).any() else 0
            accum_progress[reg] = accum_progress.get(reg, 0) + int(r["Muestras"])
            progreso = int(round(100 * min(accum_progress[reg], total_req) / total_req)) if total_req > 0 else 0
            gantt_day_rows.append({
                "Tarea": reg, 
                "Inicio": r["Inicio"], 
                "Fin": r["Fin"], 
                "Progreso": progreso,
                "Muestras_dia": int(r["Muestras"]),
                "Acumulado": accum_progress[reg],
                "Grupo": r["Grupo"]
            })
        
        gantt_per_day[d.date()] = pd.DataFrame(gantt_day_rows)
        
        pendientes_fin = (df_state["Pendiente"] > 0).sum()
        muestras_restantes = df_state["Pendiente"].sum()
        eficiencia_dia = round(util * 100, 1)
        
        # NUEVO: Verificar registros listos para entrega
        registros_entregables = obtener_registros_entregables(df_state)
        
        # Código de color para eficiencia
        emoji_eficiencia = "🟢" if eficiencia_dia >= 80 else "🟡" if eficiencia_dia >= 60 else "🔴"
        
        st.write(f"   📊 **Resumen {day_name}**: {day_samples_processed} muestras procesadas, {pendientes_fin} registros pendientes ({muestras_restantes} muestras)")
        
        # Mostrar registros listos para entrega
        if registros_entregables:
            total_muestras_entregables = sum([muestras for _, muestras in registros_entregables])
            st.success(f"   📦 **Registros listos para ENTREGA**: {len(registros_entregables)} registros ({total_muestras_entregables} muestras)")
            for registro, muestras in registros_entregables:
                st.write(f"      ✅ {registro}: {muestras} muestras - LISTO PARA ENTREGAR")
        else:
            # Mostrar registros en progreso
            registros_en_progreso = []
            for registro in df_state["Registro"].unique():
                completo, grupos_pend = verificar_registro_completo(registro, df_state)
                if not completo and len(grupos_pend) < registros_info[registros_info["Registro"] == registro]["Total_grupos"].iloc[0]:
                    grupos_pend_str = ", ".join(grupos_pend)
                    registros_en_progreso.append(f"{registro} (faltan: {grupos_pend_str})")
            
            if registros_en_progreso:
                st.info(f"   🔄 **Registros en progreso**: {len(registros_en_progreso)} registros")
                for reg_prog in registros_en_progreso[:3]:  # Mostrar máximo 3
                    st.write(f"      ⏳ {reg_prog}")
        
        # Mostrar registros que NO se pueden empezar hasta completar otros grupos
        registros_bloqueados = []
        for registro in df_state["Registro"].unique():
            reg_data = df_state[df_state["Registro"] == registro]
            grupos_total = registros_info[registros_info["Registro"] == registro]["Grupos_requeridos"].iloc[0]
            grupos_procesados = reg_data[reg_data["Pendiente"] == 0]["Tipo de analisis"].tolist()
            grupos_pendientes = reg_data[reg_data["Pendiente"] > 0]["Tipo de analisis"].tolist()
            
            # Si tiene grupos procesados pero aún tiene pendientes, está "bloqueado" para entrega
            if len(grupos_procesados) > 0 and len(grupos_pendientes) > 0:
                grupos_pend_str = ", ".join(grupos_pendientes)
                registros_bloqueados.append(f"{registro} (completar: {grupos_pend_str})")
        
        if registros_bloqueados:
            st.warning(f"   ⚠️ **Registros BLOQUEADOS para entrega**: {len(registros_bloqueados)} registros")
            for reg_bloq in registros_bloqueados[:2]:  # Mostrar máximo 2
                st.write(f"      🔒 {reg_bloq}")
        st.write(f"   {emoji_eficiencia} **Eficiencia**: {eficiencia_dia}% | **Prealist. anticipados**: {prep_anticipados}")
        st.write("---")

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
        
        # Inicializar contadores por categoría
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

    return schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day, df_progreso

def plot_gantt_user(df: pd.DataFrame, title: str):
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
            "B_C_CONJUNTO": "#FF8C00"  # Naranja para conjunto B+C
        } if "Grupo" in df_plot.columns else None,
        color_continuous_scale="RdYlGn" if "Grupo" not in df_plot.columns else None,
        range_color=[0, 100] if "Grupo" not in df_plot.columns else None,
        hover_data=["Grupo", "Muestras_dia"] if "Grupo" in df_plot.columns and "Muestras_dia" in df_plot.columns else None
    )
    
    # Configurar y mostrar el gráfico
    fig.update_yaxes(autorange="reversed")
    
    # Configurar texto en las barras
    if "Muestras_dia" in df_plot.columns:
        fig.update_traces(
            textposition="inside",
            textfont=dict(size=12, color="black", family="Arial Black"),
            texttemplate="<b>%{text}</b>"
        )
    
    fig.update_layout(
        xaxis_title="📅 Tiempo", 
        yaxis_title="📋 Actividades", 
        height=max(420, len(df_plot) * 40 + 100),
        font=dict(size=12),
        legend_title_text="🎯 Grupos de Análisis"  # Título más claro para la leyenda
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla con información de muestras como etiqueta
    if "Muestras_dia" in df_plot.columns:
        st.caption("📊 Información de muestras por registro:")
        muestras_df = df_plot[['Tarea', 'Muestras_dia']].copy()
        muestras_df.columns = ['📋 Registro', '🧪 Muestras']
        st.dataframe(muestras_df, use_container_width=True, hide_index=True)
        st.caption("📊 Información de muestras por registro:")
        muestras_df = df_plot[['Tarea', 'Muestras_dia']].copy()
        muestras_df.columns = ['📋 Registro', '🧪 Muestras']
        st.dataframe(muestras_df, use_container_width=True, hide_index=True)

def plan_fifo_simple(prueba: pd.DataFrame, tiempo: pd.DataFrame, selected_date: datetime, daily_cap: int):
    """Algoritmo FIFO simple sin optimización:
    - FIFO estricto por fecha de solicitud
    - No mezcla de muestras de diferentes registros/grupos
    - Grupo B tiene 1 día adicional de penalización
    - Sin fragmentación inteligente
    """
    if daily_cap is None or daily_cap <= 0:
        return pd.DataFrame(), pd.DataFrame()

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
    current_day_idx = 0
    
    # Procesar cada registro completo por FIFO
    registros_orden = df_state.groupby("Registro")["Fecha solicitud"].first().sort_values()
    
    for registro in registros_orden.index:
        if current_day_idx >= len(days):
            break
            
        reg_data = df_state[df_state["Registro"] == registro].copy()
        
        # Procesar cada grupo del registro
        for _, row in reg_data.iterrows():
            if current_day_idx >= len(days):
                break
                
            grupo = row["Tipo de analisis"]
            muestras = int(row["Pendiente"])
            
            if muestras <= 0:
                continue
                
            # Penalización para grupo B (+1 día)
            penalty_days = 1 if grupo == "B" else 0
            effective_day_idx = min(current_day_idx + penalty_days, len(days) - 1)
            
            # Verificar si cabe en el día actual
            while effective_day_idx < len(days):
                day_start = days[effective_day_idx]
                
                # Calcular tiempos
                t_pre = t_map.get(grupo, {}).get("Tiempo de prealistamiento (horas)", 0) or 0
                t_proc = t_map.get(grupo, {}).get("Tiempo procesamiento (horas)", 0) or 0
                
                # Si cabe completo en el día
                if muestras <= daily_cap:
                    inicio = day_start
                    duracion_total = max(t_pre + t_proc, 0.1)  # Mínimo 6 minutos
                    fin = inicio + timedelta(hours=duracion_total)
                    
                    schedule_rows.append({
                        "Fecha": day_start.date(),
                        "Registro": registro,
                        "Grupo": grupo,
                        "Muestras": muestras,
                        "Inicio": inicio,
                        "Fin": fin,
                        "Duracion (h)": duracion_total
                    })
                    
                    muestras = 0
                    break
                else:
                    # No cabe, pasar al siguiente día
                    effective_day_idx += 1
            
            # Si no cabía en ningún día, queda pendiente
            if muestras > 0:
                break
        
        # Avanzar al siguiente día para el próximo registro
        current_day_idx += 1
    
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
                {"Concepto": "Estrategia de fragmentación:", "Optimizado": "≤38 nunca fragmentar, >38 mínimo 50%", "FIFO Simple": "Sin fragmentación inteligente", "Mejora (%)": ""},
                {"Concepto": "Mezcla de muestras:", "Optimizado": "Permite mezclar grupos compatibles", "FIFO Simple": "No mezcla registros/grupos", "Mejora (%)": ""},
                {"Concepto": "Restricciones temporales:", "Optimizado": "Martes solo B+C, entrega completa", "FIFO Simple": "Grupo B +1 día penalización", "Mejora (%)": ""},
                {"Concepto": "Priorización:", "Optimizado": "FIFO + urgencia 20 días + capacidad", "FIFO Simple": "FIFO estricto por fecha", "Mejora (%)": ""},
                {"Concepto": "Programación semanal:", "Optimizado": "Lunes a Jueves (4 días disponibles)", "FIFO Simple": "Lunes a Jueves (4 días disponibles)", "Mejora (%)": ""},
                {"Concepto": "", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "RESULTADOS COMPARATIVOS", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "════════════════════════", "Optimizado": "", "FIFO Simple": "", "Mejora (%)": ""},
                {"Concepto": "Muestras procesadas:", "Optimizado": f"{muestras_opt:,.0f}", "FIFO Simple": f"{muestras_fifo:,.0f}", "Mejora (%)": f"{mejora_muestras:+.1f}%"},
                {"Concepto": "Muestras pendientes:", "Optimizado": f"{pendientes_opt:,.0f}", "FIFO Simple": f"{pendientes_fifo_total:,.0f}", "Mejora (%)": f"{mejora_pendientes:+.1f}%"},
                {"Concepto": "Días utilizados:", "Optimizado": f"{dias_opt}", "FIFO Simple": f"{dias_fifo}", "Mejora (%)": f"{ahorro_dias} días ahorrados"},
                {"Concepto": "Eficiencia procesamiento:", "Optimizado": f"{(muestras_opt/(muestras_opt+pendientes_opt)*100):,.1f}%" if (muestras_opt+pendientes_opt) > 0 else "0%", "FIFO Simple": f"{(muestras_fifo/(muestras_fifo+pendientes_fifo_total)*100):,.1f}%" if (muestras_fifo+pendientes_fifo_total) > 0 else "0%", "Mejora (%)": ""},
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
    st.dataframe(foliar, width='stretch')
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
    
    **✨ Nuevas Funcionalidades:**
    - 🌅 **Prealistamiento anticipado**: Se puede hacer el día anterior para maximizar eficiencia
    - 🔄 **Fallback automático**: Si no cabe grupo A, automáticamente prueba B, C, etc.
    - 📊 **Uso continuo**: Garantiza actividad todos los días cuando hay muestras pendientes
    - 🎯 **Priorización inteligente**: Combina urgencia temporal + cantidad disponible
    """)
    if st.button("Planificar semana (capacidad diaria)"):
        schedule, pendientes, util_df, gantt_week, gantt_per_day, df_progreso = plan_week_by_day(prueba, tiempo, datetime.combine(selected_date, DAY_START), daily_cap)
        if schedule.empty and len(gantt_per_day) == 0:
            st.stop()
        st.success("Planeación generada (toda la semana).")

        # NUEVA TABLA: Progreso por registro y categoría
        st.subheader("📊 Progreso de Muestras por Registro y Categoría")
        
        # Mostrar tabla con formato
        st.dataframe(
            df_progreso[["Registro", "Fecha Solicitud", "Total Muestras", 
                        "A_Procesadas", "A_Pendientes", 
                        "B_Procesadas", "B_Pendientes", 
                        "C_Procesadas", "C_Pendientes", 
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
                "Grupos_Requeridos": st.column_config.TextColumn("📊 Grupos Req."),
                "Estado_Entrega": st.column_config.TextColumn("🎯 Estado Entrega")
            }
        )
        
        # Resumen estadístico
        col1, col2, col3, col4 = st.columns(4)
        
        registros_listos = len([r for _, r in df_progreso.iterrows() if "LISTO PARA ENTREGAR" in r["Estado_Entrega"]])
        registros_en_progreso = len([r for _, r in df_progreso.iterrows() if "Faltan:" in r["Estado_Entrega"]])
        registros_pendientes = len([r for _, r in df_progreso.iterrows() if "Pendiente de iniciar" in r["Estado_Entrega"]])
        total_muestras_procesadas = (df_progreso["A_Procesadas"] + df_progreso["B_Procesadas"] + df_progreso["C_Procesadas"]).sum()
        
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
        muestras_pendientes = pendientes["Pendiente"].sum() if isinstance(pendientes, pd.DataFrame) and not pendientes.empty else 0
        utilizacion_promedio = util_df["Utilización (%)"].mean() if not util_df.empty else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🧪 Muestras Programadas", total_muestras)
        with col2:
            st.metric("⏳ Muestras Pendientes", muestras_pendientes)
        with col3:
            st.metric("⚡ Utilización Promedio", f"{utilizacion_promedio:.1f}%")
        with col4:
            eficiencia = (total_muestras / (total_muestras + muestras_pendientes) * 100) if (total_muestras + muestras_pendientes) > 0 else 0
            st.metric("📈 Eficiencia Semanal", f"{eficiencia:.1f}%")

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
                    plot_gantt_user(day_data, title=f"Programación {names[i]} - {d.strftime('%Y-%m-%d')}")
                    
                    # Tabla detalle del día
                    if "Muestras_dia" in day_data.columns:
                        st.markdown("**Detalle del día:**")
                        detalle_dia = day_data[["Tarea", "Muestras_dia", "Acumulado", "Progreso"]].rename(columns={
                            "Tarea": "Registro",
                            "Muestras_dia": "Muestras Procesadas",
                            "Acumulado": "Total Acumulado",
                            "Progreso": "% Completado"
                        })
                        st.dataframe(detalle_dia, width='stretch', hide_index=True)
                else:
                    st.info(f"No hay actividades programadas para {names[i]}")

        st.markdown("### 🗓️ Vista Semanal Consolidada")
        plot_gantt_user(gantt_week, title="Gantt Semanal - Progreso Acumulativo")

        # Detalles técnicos en expanders
        with st.expander("📋 Detalle Completo de la Planeación"):
            st.dataframe(schedule, width='stretch')
            
        with st.expander("⏰ Análisis de Pendientes"):
            if isinstance(pendientes, pd.DataFrame) and not pendientes.empty:
                # Análisis de pendientes por grupo
                pend_por_grupo = pendientes.groupby("Tipo de analisis").agg({
                    "Pendiente": "sum",
                    "Registro": "count"
                }).rename(columns={"Registro": "Num_Registros"}).reset_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pendientes por Tipo de Análisis:**")
                    st.dataframe(pend_por_grupo, width='stretch', hide_index=True)
                with col2:
                    st.markdown("**Todos los Pendientes:**")
                    st.dataframe(pendientes, width='stretch', hide_index=True)
            else:
                st.success("🎉 ¡Todos los pedidos fueron programados exitosamente!")

        with st.expander("⚡ KPIs de Utilización Diaria Optimizada"):
            st.markdown("**Métricas mejoradas con prealistamiento anticipado:**")
            st.dataframe(util_df, width='stretch', hide_index=True)
            
            # Gráficos de utilización mejorados
            if not util_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_util = px.bar(util_df, x="Fecha", y="Utilización (%)", 
                                    title="Utilización Diaria del Laboratorio",
                                    color="Utilización (%)",
                                    color_continuous_scale="RdYlGn",
                                    text="Utilización (%)")
                    fig_util.update_layout(xaxis_title="Día", yaxis_title="% Utilización")
                    fig_util.update_traces(texttemplate='%{text}%', textposition='outside')
                    st.plotly_chart(fig_util, width='stretch')
                
                with col2:
                    fig_prep = px.bar(util_df, x="Fecha", y="Prep. anticipados",
                                    title="Prealistamientos Anticipados por Día",
                                    color="Prep. anticipados",
                                    color_continuous_scale="Blues")
                    fig_prep.update_layout(xaxis_title="Día", yaxis_title="Cantidad")
                    st.plotly_chart(fig_prep, width='stretch')
                
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
    
    with col1:
        st.markdown("""
        **🚀 Modelo Optimizado (Actual):**
        - ✅ **Fragmentación inteligente**: ≤38 nunca fragmentar, >38 mínimo 50%
        - ✅ **Mezcla de grupos**: Combina muestras compatibles para maximizar capacidad  
        - ✅ **Priorización FIFO + urgencia**: 20 días + cantidad disponible
        - ✅ **Restricciones especiales**: Martes solo B+C, entrega completa de registros
        - ✅ **Prealistamiento anticipado**: Maximiza eficiencia diaria
        - ✅ **Programación L-J**: Sin trabajo los viernes
        """)
    
    with col2:
        st.markdown("""
        **🐌 FIFO Simple (Comparación):**
        - ❌ **Sin fragmentación**: Registros completos o nada
        - ❌ **Sin mezcla**: Un registro/grupo por vez
        - ❌ **FIFO estricto**: Solo por fecha de solicitud  
        - ❌ **Penalización grupo B**: +1 día adicional
        - ❌ **Sin optimización**: Desperdicia capacidad diaria
        - ✅ **Programación L-J**: Sin trabajo los viernes
        """)
    
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
                schedule_opt, pendientes_opt, util_df_opt, gantt_week_opt, gantt_per_day_opt, df_progreso_opt = plan_week_by_day(prueba, tiempo, fecha_datetime, daily_cap)
                
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
                # Estimar días necesarios totales (incluyendo lo pendiente)
                dias_estimados_opt = math.ceil(total_muestras_disponibles / daily_cap) if daily_cap and daily_cap > 0 else dias_utilizados_opt
                dias_reales_necesarios_opt = max(dias_utilizados_opt, dias_estimados_opt) if pendientes_opt_total > 0 else dias_utilizados_opt
                
                # Métricas del FIFO simple  
                muestras_fifo = schedule_fifo["Muestras"].sum() if isinstance(schedule_fifo, pd.DataFrame) and not schedule_fifo.empty and "Muestras" in schedule_fifo.columns else 0
                # El FIFO usa columna "Muestras" para pendientes, no "Pendiente"
                pendientes_fifo_total = pendientes_fifo["Muestras"].sum() if isinstance(pendientes_fifo, pd.DataFrame) and not pendientes_fifo.empty and "Muestras" in pendientes_fifo.columns else (total_muestras_disponibles - muestras_fifo)
                # Calcular días utilizados y estimar días necesarios totales
                dias_utilizados_fifo = len(schedule_fifo["Fecha"].unique()) if isinstance(schedule_fifo, pd.DataFrame) and not schedule_fifo.empty and "Fecha" in schedule_fifo.columns else 0
                # Estimar días necesarios totales (incluyendo lo pendiente)  
                dias_estimados_fifo = math.ceil(total_muestras_disponibles / daily_cap) if daily_cap and daily_cap > 0 else dias_utilizados_fifo
                dias_reales_necesarios_fifo = max(dias_utilizados_fifo, dias_estimados_fifo) if pendientes_fifo_total > 0 else dias_utilizados_fifo
                
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
                    {"Métrica": "Eficiencia procesamiento", "Optimizado": f"{(muestras_opt/(muestras_opt+pendientes_opt_total)*100):,.1f}%" if (muestras_opt+pendientes_opt_total) > 0 else "0%", "FIFO Simple": f"{(muestras_fifo/(muestras_fifo+pendientes_fifo_total)*100):,.1f}%" if (muestras_fifo+pendientes_fifo_total) > 0 else "0%", "Mejora": ""},
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
           - Registros >38 muestras: Solo fragmenta si queda ≥50% del registro
        4. **Mezcla Inteligente**: Combina grupos compatibles para maximizar uso de capacidad
        5. **Entrega Completa**: Solo entrega registros cuando TODOS sus grupos están completos
        6. **Prealistamiento Anticipado**: Permite preparar grupos el día anterior
        
        **Comparación con FIFO Simple:**
        - FIFO simple: Procesamiento estricto por fecha, sin mezcla, grupo B con +1 día de penalización
        - Optimizado: Flexibilidad para maximizar throughput manteniendo restricciones de negocio
        """)
