
import io
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
    return [datetime.combine((monday + timedelta(days=i)).date(), DAY_START) for i in range(5)]

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
    """Planifica L–V con **capacidad diaria optimizada**:
       - Prealistamiento se puede hacer el día anterior para maximizar eficiencia
       - Fallback automático: si no se puede A, se intenta B, C, etc.
       - Garantiza uso continuo de la máquina todos los días
       - Prioriza por fecha de solicitud más antigua y urgencia
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

    # Totales por Registro para % progreso acumulado
    totals_by_reg = df_original.groupby("Registro", as_index=False)["No muestras"].sum().rename(columns={"No muestras":"TotalRegistro"})

    # Días de la semana
    days = get_week_days(selected_date)

    schedule_rows = []
    daily_utilization = []
    pre_opened: Dict[Tuple[datetime.date,str], bool] = {}
    gantt_per_day = {}
    accum_progress: Dict[str,int] = {}
    
    # Estado global de muestras pendientes (se mantiene entre días)
    df_state = df_original.copy()

    st.info(f"🔄 Iniciando planeación semanal optimizada del {days[0].strftime('%Y-%m-%d')} al {days[-1].strftime('%Y-%m-%d')}")
    st.info("💡 **Estrategia**: Prealistamiento anticipado + fallback automático para maximizar uso de máquina")
    
    def get_available_groups_sorted(df_temp: pd.DataFrame, current_day: datetime) -> List[Tuple[str, float, int]]:
        """Retorna grupos disponibles ordenados por prioridad (grupo, score, muestras_disponibles)"""
        candidates = df_temp[df_temp["Pendiente"] > 0].copy()
        if candidates.empty:
            return []
        
        # Calcular prioridad por grupo
        group_priority = []
        for grupo in candidates["Tipo de analisis"].unique():
            group_data = candidates[candidates["Tipo de analisis"] == grupo]
            
            # Score = urgencia temporal + cantidad disponible
            try:
                dias_promedio = group_data["Fecha solicitud"].apply(
                    lambda x: (current_day.date() - pd.to_datetime(x).date()).days if pd.notna(x) else 0
                ).mean()
            except:
                dias_promedio = 0
                
            muestras_grupo = group_data["Pendiente"].sum()
            priority_score = dias_promedio * 3 + (muestras_grupo / 10)  # Más peso a urgencia temporal
            
            if grupo in t_map:  # Solo considerar grupos con tiempos definidos
                group_priority.append((grupo, priority_score, int(muestras_grupo)))
        
        # Ordenar por prioridad descendente
        return sorted(group_priority, key=lambda x: x[1], reverse=True)

    # OPTIMIZACIÓN: Prealistamiento anticipado para el día siguiente
    def try_prepare_next_day(current_day_idx: int, day_cursor: datetime, hours_available: float) -> Tuple[datetime, float]:
        """Intenta hacer prealistamiento para el día siguiente si hay tiempo disponible
        Returns: (new_cursor, prep_seconds)
        """
        try:
            # Verificar si hay día siguiente
            if current_day_idx >= len(days) - 1:
                return (day_cursor, 0.0)
            
            next_day = days[current_day_idx + 1]
            available_groups = get_available_groups_sorted(df_state, next_day)
            
            # Intentar prealistamiento para cada grupo disponible
            for grupo, _, _ in available_groups:
                if pre_opened.get((next_day.date(), grupo), False):
                    continue  # Ya prealistado para mañana
                    
                tiempos = t_map.get(grupo, {})
                t_pre = float(tiempos.get("Tiempo de prealistamiento (horas)", 0) or 0.0)
                
                if t_pre > 0 and hours_available >= t_pre:
                    # Hacer prealistamiento anticipado
                    pre_start = day_cursor
                    pre_end = pre_start + timedelta(hours=t_pre)
                    prep_seconds = (pre_end - pre_start).total_seconds()
                    
                    schedule_rows.append({
                        "Fecha": pre_start.date(),
                        "Inicio": pre_start,
                        "Fin": pre_end,
                        "Registro": f"Prealistamiento {grupo} (para mañana)",
                        "Grupo": grupo,
                        "Tipo": "Prealistamiento",
                        "Muestras": 0
                    })
                    
                    pre_opened[(next_day.date(), grupo)] = True
                    st.write(f"   🌅 Prealistamiento anticipado {grupo} para mañana: {pre_start.strftime('%H:%M')} - {pre_end.strftime('%H:%M')} ({t_pre:.1f}h)")
                    
                    return (pre_end, prep_seconds)
            
            # No se encontró ningún grupo para preparar
            return (day_cursor, 0.0)
            
        except Exception as e:
            st.error(f"Error en try_prepare_next_day: {e}")
            return (day_cursor, 0.0)
    
    for day_idx, d in enumerate(days):
        day_name = ["Lunes","Martes","Miércoles","Jueves","Viernes"][day_idx]
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
            
            for grupo, priority_score, muestras_disponibles in available_groups:
                tiempos = t_map.get(grupo, {})
                t_pre = float(tiempos.get("Tiempo de prealistamiento (horas)", 0) or 0.0)
                t_proc = float(tiempos.get("Tiempo procesamiento (horas)", 0) or 0.0)
                
                # Determinar si necesita prealistamiento (ya hecho hoy o ayer)
                pre_time = 0.0 if pre_opened.get((d.date(), grupo), False) else t_pre
                total_time_needed = pre_time + t_proc
                
                fallback_attempts.append(f"{grupo}({total_time_needed:.1f}h)")
                
                # VALIDACIÓN: ¿Cabe en el tiempo disponible?
                if hours_left >= total_time_needed and t_proc > 0:
                    grupo_procesado = grupo
                    break
                elif t_proc <= 0:
                    st.warning(f"   ⚠️ {grupo} tiene tiempo de procesamiento 0, saltando")
                else:
                    st.info(f"   ⌛ {grupo} necesita {total_time_needed:.1f}h pero solo hay {hours_left:.1f}h disponibles")
            
            # Si no se encontró ningún grupo que quepa, intentar prealistamiento para mañana
            if grupo_procesado is None:
                st.warning(f"   🔄 Fallback probado: {', '.join(fallback_attempts)} - Ninguno cabe")
                
                # Intentar prealistamiento anticipado para maximizar uso
                new_cursor, prep_seconds = try_prepare_next_day(day_idx, day_cursor, hours_left)
                if prep_seconds > 0:
                    day_cursor = new_cursor
                    used_seconds += prep_seconds
                    continue
                else:
                    st.info(f"   🔚 No se puede optimizar más tiempo en {day_name}")
                    break
            
            # PROCESAMIENTO DEL GRUPO SELECCIONADO
            grupo = grupo_procesado
            tiempos = t_map[grupo]
            t_pre = float(tiempos.get("Tiempo de prealistamiento (horas)", 0) or 0.0)
            t_proc = float(tiempos.get("Tiempo procesamiento (horas)", 0) or 0.0)
            pre_time = 0.0 if pre_opened.get((d.date(), grupo), False) else t_pre
            
            st.write(f"   ✅ Grupo seleccionado: {grupo} (prioridad: {priority_score:.1f}, tiempo: {pre_time + t_proc:.1f}h)")

            # Seleccionar registros del grupo por prioridad
            grp_df = df_state[(df_state["Pendiente"] > 0) & (df_state["Tipo de analisis"] == grupo)].copy()
            if grp_df.empty:
                st.warning(f"   ⚠️ No hay muestras pendientes para {grupo}, continuando")
                continue
                
            grp_df = grp_df.sort_values(by=["Fecha solicitud", "Registro"])
            pend_grp_total = int(grp_df["Pendiente"].sum())
            session_take = int(min(remaining_daily, pend_grp_total))
            
            if session_take <= 0:
                st.info(f"   📊 Capacidad diaria agotada ({remaining_daily} restante)")
                break

            session_attempted = True

            # Registrar prealistamiento si es necesario (y no fue hecho ayer)
            if pre_time > 0 and not pre_opened.get((d.date(), grupo), False):
                pre_start = day_cursor
                pre_end = pre_start + timedelta(hours=pre_time)
                schedule_rows.append({
                    "Fecha": pre_start.date(),
                    "Inicio": pre_start,
                    "Fin": pre_end,
                    "Registro": f"Prealistamiento {grupo}",
                    "Grupo": grupo,
                    "Tipo": "Prealistamiento",
                    "Muestras": 0
                })
                used_seconds += (pre_end - pre_start).total_seconds()
                day_cursor = pre_end
                pre_opened[(d.date(), grupo)] = True
                st.write(f"   🔧 Prealistamiento {grupo}: {pre_start.strftime('%H:%M')} - {pre_end.strftime('%H:%M')} ({pre_time:.1f}h)")

            # Sesión de procesamiento
            proc_start = day_cursor
            proc_end = proc_start + timedelta(hours=t_proc)

            # Asignar muestras a registros (más antiguos primero)
            to_assign = session_take
            session_registros = []
            
            for idx, row in grp_df.iterrows():
                if to_assign <= 0:
                    break
                pend = int(df_state.loc[idx, "Pendiente"])
                if pend <= 0:
                    continue
                    
                take = int(min(pend, to_assign))
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
                to_assign -= take
                session_registros.append(f"{df_state.loc[idx, 'Registro']}({take})")

            used_seconds += (proc_end - proc_start).total_seconds()
            day_cursor = proc_end
            remaining_daily -= session_take
            day_samples_processed += session_take
            
            st.write(f"   ⚗️ Procesamiento {grupo}: {proc_start.strftime('%H:%M')} - {proc_end.strftime('%H:%M')} ({t_proc:.1f}h) - {session_take} muestras: {', '.join(session_registros)}")

        # OPTIMIZACIÓN FINAL: Si queda tiempo y hay pendientes, intentar prealistamiento para mañana
        final_hours_left = (day_end - day_cursor).total_seconds() / 3600.0
        if final_hours_left > 0.5 and (df_state["Pendiente"] > 0).any():  # Al menos 30 min libres
            st.info(f"   🔍 Optimizando tiempo restante: {final_hours_left:.1f}h disponibles")
            new_cursor, prep_seconds = try_prepare_next_day(day_idx, day_cursor, final_hours_left)
            if prep_seconds > 0:
                used_seconds += prep_seconds
                day_cursor = new_cursor
        
        # Si no se procesó nada y hay muestras pendientes, reportar
        if not session_attempted and (df_state["Pendiente"] > 0).any():
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
                "Acumulado": accum_progress[reg]
            })
        
        gantt_per_day[d.date()] = pd.DataFrame(gantt_day_rows)
        
        pendientes_fin = (df_state["Pendiente"] > 0).sum()
        muestras_restantes = df_state["Pendiente"].sum()
        eficiencia_dia = round(util * 100, 1)
        
        # Código de color para eficiencia
        emoji_eficiencia = "🟢" if eficiencia_dia >= 80 else "🟡" if eficiencia_dia >= 60 else "🔴"
        
        st.write(f"   📊 **Resumen {day_name}**: {day_samples_processed} muestras procesadas, {pendientes_fin} registros pendientes ({muestras_restantes} muestras)")
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
    if len(gantt_per_day):
        gantt_week_df = pd.concat([g for g in gantt_per_day.values()], ignore_index=True)
    else:
        gantt_week_df = pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso"])

    return schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day

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
        color="Progreso",
        title=title,
        color_continuous_scale="RdYlGn",
        range_color=[0, 100]
    )
    
    # Añadir anotaciones con información de muestras
    if "Muestras_dia" in df_plot.columns:
        for _, row in df_plot.iterrows():
            fig.add_annotation(
                x=row["Inicio"] + (row["Fin"] - row["Inicio"]) / 2,
                y=row["Tarea"],
                text=f"{row['Muestras_dia']}",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.5)"
            )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Horario", 
        yaxis_title="Registro", 
        height=max(420, len(df_plot) * 40 + 100),
        coloraxis_colorbar=dict(title="% Progreso")
    )
    st.plotly_chart(fig, width='stretch')

def to_excel_download(schedule: pd.DataFrame, pendientes: pd.DataFrame, util_df: pd.DataFrame,
                      gantt_week: pd.DataFrame, gantt_per_day: Dict) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        schedule.to_excel(writer, sheet_name="Plan", index=False)
        pendientes.to_excel(writer, sheet_name="Pendiente", index=False)
        util_df.to_excel(writer, sheet_name="Utilizacion", index=False)
        gantt_week.to_excel(writer, sheet_name="Gantt_semana", index=False)
        for k, v in gantt_per_day.items():
            sheet = f"Gantt_{k.strftime('%a')}"
            v.to_excel(writer, sheet_name=sheet, index=False)
    return out.getvalue()

# =============================
# UI
# =============================
st.sidebar.header("Configuración de datos")
default_path = "Insumo_Planeacion.xlsx"
uploaded = st.sidebar.file_uploader("Cargar Excel personalizado", type=["xlsx", "xlsm"])
foliar, prueba, tiempo, daily_cap = load_data(default_path, uploaded.getvalue() if uploaded else None)

tab1, tab2 = st.tabs(["📚 Históricos", "🗓️ Planeación por día (capacidad diaria)"])

with tab1:
    st.subheader("Tabla de históricos — Foliar (KPIs solo Aplican = SI)")
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
    **🚀 Planeación Optimizada** - Lunes a Viernes, 8:00–18:00, capacidad diaria = {cap_txt}
    
    **✨ Nuevas Funcionalidades:**
    - 🌅 **Prealistamiento anticipado**: Se puede hacer el día anterior para maximizar eficiencia
    - 🔄 **Fallback automático**: Si no cabe grupo A, automáticamente prueba B, C, etc.
    - 📊 **Uso continuo**: Garantiza actividad todos los días cuando hay muestras pendientes
    - 🎯 **Priorización inteligente**: Combina urgencia temporal + cantidad disponible
    """)
    if st.button("Planificar semana (capacidad diaria)"):
        schedule, pendientes, util_df, gantt_week, gantt_per_day = plan_week_by_day(prueba, tiempo, datetime.combine(selected_date, DAY_START), daily_cap)
        if schedule.empty and len(gantt_per_day) == 0:
            st.stop()
        st.success("Planeación generada (toda la semana).")

        # Resumen ejecutivo
        st.markdown("### 📊 Resumen Ejecutivo de la Planeación")
        total_muestras = util_df["Muestras procesadas"].sum() if not util_df.empty else 0
        muestras_pendientes = pendientes["Pendiente"].sum() if not pendientes.empty else 0
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
        names = ["Lunes","Martes","Miércoles","Jueves","Viernes"]
        
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
            if not pendientes.empty:
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

        xls_bytes = to_excel_download(schedule, pendientes, util_df, gantt_week, gantt_per_day)
        st.download_button(
            label="⬇️ Descargar plan (semana) con Gantt por día",
            data=xls_bytes,
            file_name="planeacion_semana_cap_diaria.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
