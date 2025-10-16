
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
        xls =    # Personalizar el gráfico
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="📅 Tiempo", 
        yaxis_title="📋 Actividades", 
        height=420,
        showlegend=True,
        font=dict(size=12),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color="darkblue")
        )
    )
    
    # Mostrar información de muestras como tabla debajo del gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla con información de muestras
    if not df.empty and 'Muestras' in df.columns:
        muestras_info = df[['Tarea', 'Muestras']].copy()
        muestras_info.columns = ['📋 Registro', '🧪 Muestras']
        st.caption("📊 Detalle de muestras por registro:")
        st.dataframe(muestras_info, use_container_width=True, hide_index=True)esIO(uploaded))
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
    """Devuelve df(Tarea,Inicio,Fin,Progreso,Muestras,Categoria) + acumulador de progreso por Registro."""
    rows = []
    if blocks.empty:
        return pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso","Muestras","Categoria"]), accum
    blocks = blocks.sort_values("Inicio")
    for _, r in blocks.iterrows():
        reg = str(r["Registro"])
        total_req = int(totals_by_reg.loc[totals_by_reg["Registro"] == reg, "TotalRegistro"].iloc[0]) if (totals_by_reg["Registro"] == reg).any() else 0
        accum[reg] = accum.get(reg, 0) + int(r["Muestras"])
        progreso = int(round(100 * min(accum[reg], total_req) / total_req)) if total_req > 0 else 0
        rows.append({
            "Tarea": reg, 
            "Inicio": r["Inicio"], 
            "Fin": r["Fin"], 
            "Progreso": progreso,
            "Muestras": int(r["Muestras"]),
            "Categoria": str(r["Grupo"])  # Agregar categoria
        })
    return pd.DataFrame(rows), accum

def plan_week_by_day(prueba: pd.DataFrame, tiempo: pd.DataFrame, selected_date: datetime, daily_cap: int):
    """Planifica L–V con **capacidad diaria** (muestras/día) y tiempos por sesión:
       - Una **sesión** de un grupo dura `t_proc` horas **independiente** del # de muestras asignadas.
       - En una sesión se asignan hasta `remaining_daily` muestras (repartidas entre registros del grupo, más antiguos primero).
       - Prealistamiento se cobra una sola vez por (día, grupo).
    """
    if daily_cap is None or daily_cap <= 0:
        st.error("No se encontró capacidad diaria válida en la hoja 'Capacidad'. Ingresa un valor numérico > 0.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    # Map de tiempos
    tiempo = tiempo.copy()
    tiempo.columns = [c.strip() for c in tiempo.columns]
    tiempo["Grupo"] = tiempo["Grupo"].astype(str).str.strip()
    t_map = tiempo.set_index("Grupo")[["Tiempo de prealistamiento (horas)", "Tiempo procesamiento (horas)"]].to_dict("index")

    # Pedidos expandidos A,B
    df = expand_analyses(prueba)
    
    # Debug: verificar datos expandidos
    if df.empty:
        st.error("⚠️ No se encontraron datos para planificar")
        # Retornar DataFrames vacíos pero válidos en lugar de parar completamente
        empty_df = pd.DataFrame(columns=["Fecha", "Inicio", "Fin", "Registro", "Grupo", "Tipo", "Muestras"])
        return empty_df, empty_df, empty_df, {}, {}
    
    df = df.sort_values(by=["Fecha solicitud", "Registro"]).reset_index(drop=True)
    df["Pendiente"] = df["No muestras"].fillna(0).astype(int)
    df["Tipo de analisis"] = df["Tipo de analisis"].astype(str).str.strip()
    df["Registro"] = df["Registro"].astype(str)
    
    # Debug temporal: verificar estado de pendientes
    total_pendientes = df["Pendiente"].sum()
    if total_pendientes == 0:
        st.warning(f"⚠️ No hay muestras pendientes para procesar (total: {total_pendientes})")
        # Pero aún así retornar estructura válida para mostrar mensaje
        empty_df = pd.DataFrame(columns=["Fecha", "Inicio", "Fin", "Registro", "Grupo", "Tipo", "Muestras"])
        empty_gantt = pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso"])
        empty_util = pd.DataFrame(columns=["Fecha", "Utilización (%)", "Muestras procesadas"])
        return empty_df, empty_df, empty_util, empty_gantt, {}
    else:
        st.info(f"📊 Total de muestras pendientes: {total_pendientes} muestras en {len(df)} registros")

    # Totales por Registro para % progreso
    totals_by_reg = df.groupby("Registro", as_index=False)["No muestras"].sum().rename(columns={"No muestras":"TotalRegistro"})

    # Días
    days = get_week_days(selected_date)

    schedule_rows = []
    daily_utilization = []
    pre_opened: Dict[Tuple[datetime.date,str], bool] = {}
    gantt_per_day = {}
    accum_progress: Dict[str,int] = {}

    for d in days:
        day_start = d
        day_end = datetime.combine(d.date(), DAY_END)
        day_cursor = day_start
        used_seconds = 0.0
        remaining_daily = int(daily_cap)
        current_group = None
        
        # Debug por día
        dia_nombre = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"][d.weekday()]
        st.info(f"🗓️ Procesando {dia_nombre} ({d.date()}) - Capacidad: {remaining_daily} muestras")

        while day_cursor < day_end and (df["Pendiente"] > 0).any() and remaining_daily > 0:
            # Lógica simplificada: selección de grupo basada en día de la semana
            candidates = df[df["Pendiente"] > 0].sort_values(by=["Fecha solicitud", "Registro"])  # FIFO sempre
            if candidates.empty:
                break
            
            # Determinar día de la semana (0=Lunes, 1=Martes, ..., 4=Viernes)
            dia_semana = d.weekday()
            
            if dia_semana == 0:  # LUNES
                # Solo grupo A
                grupo_candidates = candidates[candidates["Tipo de analisis"] == "A"]
                if not grupo_candidates.empty:
                    grupo = "A"
                else:
                    break  # Si no hay A, salir
                    
            elif dia_semana == 1:  # MARTES  
                # Solo grupos B y C
                grupo_candidates = candidates[candidates["Tipo de analisis"].isin(["B", "C"])]
                if not grupo_candidates.empty:
                    grupo = "B_C_CONJUNTO"  # Procesar B y C juntos
                else:
                    break  # Si no hay B o C, salir
                    
            else:  # MIÉRCOLES, JUEVES, VIERNES
                # Cualquier grupo disponible - FIFO
                if current_group is not None and (candidates["Tipo de analisis"] == current_group).any():
                    grupo = current_group
                else:
                    # Tomar el primer registro disponible (FIFO)
                    grupo = str(candidates.iloc[0]["Tipo de analisis"])

            # Obtener tiempos de procesamiento (sin prealistamiento)
            if grupo == "B_C_CONJUNTO":
                # Usar tiempos de B para la sesión (B y C tienen los mismos tiempos)
                tiempos = t_map.get("B", None)
                if tiempos is None:
                    tiempos = t_map.get("C", None)
                if tiempos is None:
                    break
                grupo_display = "B+C"
                st.info(f"🗓️ MARTES: Procesamiento conjunto B+C")
            else:
                tiempos = t_map.get(grupo, None)
                
                # Si es categoría C, usar los mismos tiempos que B
                if tiempos is None and grupo.upper() == "C":
                    tiempos = t_map.get("B", None)
                
                if tiempos is None:
                    # Probar otro grupo
                    others = candidates[candidates["Tipo de analisis"] != grupo]
                    if others.empty:
                        break
                    grupo = str(others.iloc[0]["Tipo de analisis"])
                    tiempos = t_map.get(grupo, None)
                    if tiempos is None:
                        break
                grupo_display = grupo

            # Solo tiempo de procesamiento (sin prealistamiento)
            t_proc = float(tiempos["Tiempo procesamiento (horas)"] or 0.0)
            hours_left = (day_end - day_cursor).total_seconds() / 3600.0

            # ¿Cabe al menos una sesión de procesamiento?
            if hours_left < t_proc:
                break

            # --- Asignar registros completos (no fragmentar) ---
            # Buscar registros según el tipo de procesamiento
            if grupo == "B_C_CONJUNTO":
                # MARTES: Buscar registros de ambos grupos B y C para procesamiento conjunto
                grp_df = df[(df["Pendiente"] > 0) & (df["Tipo de analisis"].isin(["B", "C"]))].sort_values(by=["Fecha solicitud","Registro"])
            else:
                # Otros días: buscar registros del grupo específico
                grp_df = df[(df["Pendiente"] > 0) & (df["Tipo de analisis"] == grupo)].sort_values(by=["Fecha solicitud","Registro"])
            
            if grp_df.empty:
                current_group = None
                continue

            # Inicio directo de procesamiento (sin prealistamiento)
            proc_start = day_cursor
            if t_proc <= 0:
                proc_end = proc_start + timedelta(minutes=30)  # Tiempo mínimo de 30 min
            else:
                proc_end = proc_start + timedelta(hours=t_proc)

            # Algoritmo FIFO ESTRICTO: NO dividir registros ≤38 muestras
            total_assigned = 0
            
            # FIFO: Ordenar por fecha de solicitud y registro para procesamiento en orden
            # Para procesamiento conjunto B+C, usar el grp_df ya filtrado
            if grupo == "B_C_CONJUNTO":
                # Ya está filtrado arriba para B y C
                pass
            else:
                # Para otros días, refiltrar por el grupo específico
                grp_df = df[(df["Pendiente"] > 0) & (df["Tipo de analisis"] == grupo)].sort_values(by=["Fecha solicitud", "Registro"])
            
            # Procesar registros en orden FIFO, NUNCA dividir registros ≤38
            for idx, row in grp_df.iterrows():
                if remaining_daily <= 0:
                    break
                    
                pend = int(df.loc[idx, "Pendiente"])
                if pend <= 0:
                    continue
                
                # REGLA: Solo registros ≤38 muestras (no se pueden dividir) - FIFO
                if pend <= daily_cap:
                    # Si el registro cabe completo en el espacio restante
                    if pend <= remaining_daily:
                        # Para procesamiento conjunto, usar el grupo real del registro
                        grupo_real = str(df.loc[idx, "Tipo de analisis"]) if grupo == "B_C_CONJUNTO" else grupo
                        schedule_rows.append({
                            "Fecha": proc_start.date(),
                            "Inicio": proc_start,
                            "Fin": proc_end,
                            "Registro": str(df.loc[idx, "Registro"]),
                            "Grupo": grupo_real,
                            "Tipo": "Procesamiento",
                            "Muestras": pend  # SIEMPRE el registro completo
                        })
                        df.loc[idx, "Pendiente"] = 0
                        total_assigned += pend
                        remaining_daily -= pend
                        if grupo == "B_C_CONJUNTO":
                            st.info(f"✅ MARTES B+C: Procesado {df.loc[idx, 'Registro']} ({pend} muestras, grupo {grupo_real}) - Capacidad restante: {remaining_daily}")
                        else:
                            st.info(f"✅ NO DIVISIÓN: Procesado completo {df.loc[idx, 'Registro']} ({pend} muestras) - Capacidad restante: {remaining_daily}")
            
            # Si no se procesó nada con registros ≤38, intentar con registros >38 muestras
            if total_assigned == 0 and remaining_daily > 0:
                for idx, row in grp_df.iterrows():                        
                    pend = int(df.loc[idx, "Pendiente"])
                    if pend <= 0 or pend <= daily_cap:
                        continue
                    
                    # EXCEPCIÓN: Registros >38 muestras SÍ se pueden fraccionar (el más antiguo primero)
                    take = min(pend, remaining_daily)
                    if take > 0:
                        schedule_rows.append({
                            "Fecha": proc_start.date(),
                            "Inicio": proc_start,
                            "Fin": proc_end,
                            "Registro": str(df.loc[idx, "Registro"]),
                            "Grupo": grupo,
                            "Tipo": "Procesamiento",
                            "Muestras": take
                        })
                        df.loc[idx, "Pendiente"] = pend - take
                        total_assigned += take
                        remaining_daily -= take
                        st.info(f"� FIFO: Fraccionado {df.loc[idx, 'Registro']} ({take}/{pend} muestras) - Día {dia_semana + 1}")
                        break  # Solo procesar un registro grande por sesión

            used_seconds += (proc_end - proc_start).total_seconds()
            day_cursor = proc_end

            # Si no se asignó nada o no quedan más registros que quepan completos, cambiar de grupo
            if total_assigned == 0 or remaining_daily <= 0:
                current_group = None
            elif (df[(df["Pendiente"] > 0) & (df["Tipo de analisis"] == grupo)].empty):
                current_group = None
            else:
                current_group = grupo

        # LUNES: Prealistamiento obligatorio de B y C para el martes
        if d.weekday() == 0:  # Si es lunes
            for grupo_prep in ["B", "C"]:
                # Solo hacer prealistamiento si no se ha hecho ya y hay tiempo disponible
                if not pre_opened.get((d.date(), grupo_prep), False):
                    tiempos_prep = t_map.get(grupo_prep, None)
                    if tiempos_prep is not None:
                        t_pre_prep = float(tiempos_prep["Tiempo de prealistamiento (horas)"] or 0.0)
                        if t_pre_prep > 0:
                            hours_left_prep = (day_end - day_cursor).total_seconds() / 3600.0
                            if hours_left_prep >= t_pre_prep:
                                # Hacer prealistamiento para el martes
                                pre_start_prep = day_cursor
                                pre_end_prep = pre_start_prep + timedelta(hours=t_pre_prep)
                                schedule_rows.append({
                                    "Fecha": pre_start_prep.date(),
                                    "Inicio": pre_start_prep,
                                    "Fin": pre_end_prep,
                                    "Registro": f"Prealistamiento {grupo_prep} (para martes)",
                                    "Grupo": grupo_prep,
                                    "Tipo": "Prealistamiento",
                                    "Muestras": 0
                                })
                                used_seconds += (pre_end_prep - pre_start_prep).total_seconds()
                                day_cursor = pre_end_prep
                                pre_opened[(d.date(), grupo_prep)] = True
                                st.info(f"📋 Lunes: Prealistamiento de {grupo_prep} programado para martes ({t_pre_prep}h)")

        # KPIs de uso del día
        day_total_seconds = (day_end - day_start).total_seconds()
        util = used_seconds / day_total_seconds if day_total_seconds > 0 else 0.0
        daily_utilization.append({"Fecha": d.date(), "Utilización (%)": round(util * 100, 1), "Muestras procesadas": int(daily_cap - remaining_daily)})

        # Gantt del día (formato usuario)
        day_blocks = pd.DataFrame(schedule_rows)
        if not day_blocks.empty:
            day_blocks = day_blocks[(day_blocks["Fecha"] == d.date()) & (day_blocks["Tipo"] == "Procesamiento")]
        else:
            day_blocks = pd.DataFrame(columns=["Fecha","Inicio","Fin","Registro","Grupo","Tipo","Muestras"])
        day_blocks_cons = consolidate_blocks(day_blocks)
        # Construir progreso acumulado arrastrando entre días
        if 'totals_by_reg' not in locals():
            totals_by_reg = df.groupby("Registro", as_index=False)["No muestras"].sum().rename(columns={"No muestras":"TotalRegistro"})
        # crear/usar acumulador fuera del loop
        if 'accum_progress' not in locals():
            accum_progress = {}
        # Para mantener acumulado entre días, guardar fuera y reutilizar
        # (en esta función ya usamos 'accum_progress' del scope exterior del loop de días)
        # construir df del día
        gantt_day_rows = []
        # Recalcular con acumulado
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
                "Muestras": int(r["Muestras"]),
                "Categoria": str(r["Grupo"])  # Agregar categoria para coloración
            })
        gantt_per_day[d.date()] = pd.DataFrame(gantt_day_rows)

    # Resultado semanal
    if schedule_rows:
        schedule_week = pd.DataFrame(schedule_rows).sort_values("Inicio").reset_index(drop=True)
    else:
        # Crear DataFrame vacío con las columnas esperadas
        schedule_week = pd.DataFrame(columns=["Fecha", "Inicio", "Fin", "Registro", "Grupo", "Tipo", "Muestras"])

    # Pendientes
    pend_base = expand_analyses(prueba).copy()
    pend_base["Grupo"] = pend_base["Tipo de analisis"].astype(str).str.strip()
    pend_base["Registro"] = pend_base["Registro"].astype(str)
    if not schedule_week.empty:
        processed = schedule_week[schedule_week["Tipo"] == "Procesamiento"].copy()
        processed["Registro"] = processed["Registro"].astype(str)
        proc_sum = processed.groupby(["Registro", "Grupo"], as_index=False)["Muestras"].sum()
    else:
        proc_sum = pd.DataFrame(columns=["Registro","Grupo","Muestras"])
    pend_base = pend_base.merge(proc_sum, on=["Registro","Grupo"], how="left")
    pend_base["Muestras"] = pend_base["Muestras"].fillna(0).astype(int)
    pend_base["Procesadas"] = pend_base["Muestras"]
    pend_base.drop(columns=["Muestras"], inplace=True)
    pend_base["Pendiente"] = (pend_base["No muestras"].astype(int) - pend_base["Procesadas"]).clip(lower=0).astype(int)
    pendientes = pend_base[["ID analisis","Registro","Tipo de analisis","No muestras","Procesadas","Pendiente"]].rename(columns={"No muestras":"Solicitadas"})

    util_df = pd.DataFrame(daily_utilization)

    # Gantt semanal (concatenación de días)
    if len(gantt_per_day):
        gantt_week_df = pd.concat([g for g in gantt_per_day.values()], ignore_index=True)
        if not gantt_week_df.empty:
            st.success(f"✅ Gantt semanal generado: {len(gantt_week_df)} tareas a lo largo de {len(gantt_per_day)} días")
        else:
            st.warning("⚠️ Gantt semanal vacío: no hay tareas para mostrar")
    else:
        gantt_week_df = pd.DataFrame(columns=["Tarea","Inicio","Fin","Progreso","Muestras","Categoria"])
        st.warning("⚠️ No se generaron datos de Gantt para ningún día")

    return schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day

def plot_gantt_user(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Sin bloques para mostrar.")
        return
    
    # Agregar numeración y etiquetas detalladas a las tareas
    df_with_numbers = df.copy()
    df_with_numbers['Numero'] = range(1, len(df) + 1)
    df_with_numbers['Tarea_Numerada'] = df_with_numbers['Numero'].astype(str) + '. ' + df_with_numbers['Tarea'].astype(str)
    
    # Asegurar que las columnas de tiempo son datetime
    df_with_numbers['Inicio'] = pd.to_datetime(df_with_numbers['Inicio'])
    df_with_numbers['Fin'] = pd.to_datetime(df_with_numbers['Fin'])
    
    # Si no hay columna Categoria, usar Progreso como fallback
    color_column = "Categoria" if "Categoria" in df_with_numbers.columns else "Progreso"
    
    # Crear gráfico limpio con categorías como colores
    fig = px.timeline(df_with_numbers, 
                     x_start="Inicio", 
                     x_end="Fin", 
                     y="Tarea_Numerada", 
                     color=color_column,  # Usar categoria para coloración
                     title=title,
                     color_discrete_map={
                         "A": "#FF6B6B",  # Rojo para categoria A
                         "B": "#4ECDC4",  # Verde turquesa para categoria B  
                         "C": "#45B7D1",  # Azul para categoria C
                         "B_C_CONJUNTO": "#96CEB4"  # Verde claro para conjunto B+C
                     },
                     hover_data={
                         "Numero": True,
                         "Tarea": True, 
                         "Progreso": True,
                         "Muestras": True,
                         "Categoria": True if color_column == "Categoria" else False,
                         "Inicio": "|%H:%M",
                         "Fin": "|%H:%M"
                     })
    
    # Personalizar el gráfico
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="� Días", 
        yaxis_title="📋 Actividades", 
        height=420,
        showlegend=True,
        font=dict(size=12),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color="darkblue")
        ),
        xaxis=dict(
            type="date",
            tickformat="%a %d/%m",  # Formato: Lun 01/01, Mar 02/01, etc.
            dtick="D1",  # Un tick por día
            tickmode="linear"
        )
    )
    
    # Agregar anotaciones con estadísticas (con manejo de errores)
    try:
        total_tareas = len(df)
        if total_tareas > 0 and 'Inicio' in df.columns and 'Fin' in df.columns:
            inicio_min = pd.to_datetime(df['Inicio']).min()
            fin_max = pd.to_datetime(df['Fin']).max()
            duracion_total = (fin_max - inicio_min).total_seconds() / 3600
            
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"📈 Total: {total_tareas} tareas | ⏱️ Duración: {duracion_total:.1f}h",
                showarrow=False,
                font=dict(size=11, color="darkgreen"),
                bgcolor="lightgray",
                bordercolor="gray",
                borderwidth=1
            )
    except Exception as e:
        # Si hay error con las estadísticas, continuar sin anotaciones
        pass  # Cambiar a pass para evitar mostrar warnings
    
    st.plotly_chart(fig, use_container_width=True)

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
    st.caption(f"Se planifica de Lunes a Viernes, 8:00–18:00, con **capacidad diaria** = {cap_txt}. Mezcla pedidos del mismo grupo y aplica prealistamiento solo al abrir el grupo cada día.")
    if st.button("Planificar semana (capacidad diaria)"):
        with st.spinner("Generando planificación..."):
            schedule, pendientes, util_df, gantt_week, gantt_per_day = plan_week_by_day(prueba, tiempo, datetime.combine(selected_date, DAY_START), daily_cap)
        if schedule.empty and len(gantt_per_day) == 0:
            st.stop()
        st.success("Planeación generada (toda la semana).")

        st.markdown("### Gantt por día (formato solicitado)")
        days = sorted(list(gantt_per_day.keys()))
        names = ["Lunes","Martes","Miércoles","Jueves","Viernes"]
        for i, d in enumerate(days):
            st.markdown(f"**{names[i] if i < len(names) else d.strftime('%A')} — {d}**")
            plot_gantt_user(gantt_per_day[d], title=f"Gantt del día — {names[i] if i < len(names) else d.strftime('%A')}")

        st.markdown("### Gantt semanal (formato solicitado)")
        plot_gantt_user(gantt_week, title="Gantt de la semana (acumulado)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Detalle de la planeación (Plan)**")
            st.dataframe(schedule, width='stretch')
        with c2:
            st.markdown("**Pedidos pendientes**")
            st.dataframe(pendientes, width='stretch')

        st.markdown("### 📊 KPIs de uso de máquina (por día)")
        st.dataframe(util_df, use_container_width=True)

        # Agregar gráficas de utilización
        if not util_df.empty:
            # Gráfico de barras para utilización
            fig_util = px.bar(util_df, 
                             x='Fecha', 
                             y='Utilización (%)',
                             title="📈 Utilización Diaria del Laboratorio",
                             text='Utilización (%)',
                             hover_data=['Muestras procesadas'],
                             color='Utilización (%)',
                             color_continuous_scale='RdYlGn')
            
            fig_util.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_util.update_layout(
                xaxis_title="🗓️ Fecha",
                yaxis_title="📊 Utilización (%)",
                font=dict(size=12),
                height=400,
                title=dict(x=0.5, font=dict(size=16, color="darkblue"))
            )
            
            # Línea de referencia en 80% (utilización óptima)
            fig_util.add_hline(y=80, line_dash="dash", line_color="orange", 
                              annotation_text="Meta: 80%", annotation_position="bottom right")
            
            st.plotly_chart(fig_util, use_container_width=True)
            
            # Gráfico de muestras procesadas
            fig_samples = px.line(util_df,
                                 x='Fecha',
                                 y='Muestras procesadas', 
                                 title="🧪 Muestras Procesadas por Día",
                                 markers=True,
                                 text='Muestras procesadas')
            
            fig_samples.update_traces(textposition='top center', line=dict(width=3))
            fig_samples.update_layout(
                xaxis_title="🗓️ Fecha",
                yaxis_title="🧪 Número de Muestras",
                font=dict(size=12),
                height=400,
                title=dict(x=0.5, font=dict(size=16, color="darkblue"))
            )
            
            # Línea de capacidad máxima
            fig_samples.add_hline(y=38, line_dash="dash", line_color="red",
                                 annotation_text="Capacidad máx: 38", annotation_position="top right")
            
            st.plotly_chart(fig_samples, use_container_width=True)

        xls_bytes = to_excel_download(schedule, pendientes, util_df, gantt_week, gantt_per_day)
        st.download_button(
            label="⬇️ Descargar plan (semana) con Gantt por día",
            data=xls_bytes,
            file_name="planeacion_semana_cap_diaria.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
