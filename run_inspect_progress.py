import pandas as pd
from datetime import datetime
import app_planeacion as ap

prueba = pd.DataFrame([
    {"Registro":"R1","Tipo de analisis":"B","Fecha solicitud":"2025-10-01","No muestras":95},
    {"Registro":"R2","Tipo de analisis":"C","Fecha solicitud":"2025-10-05","No muestras":20},
    {"Registro":"R3","Tipo de analisis":"D","Fecha solicitud":"2025-10-10","No muestras":10},
    {"Registro":"R4","Tipo de analisis":"A","Fecha solicitud":"2025-10-12","No muestras":30},
])

tiempo = pd.DataFrame([
    {"Grupo":"A","Tiempo de prealistamiento (horas)":0.5,"Tiempo procesamiento (horas)":1},
    {"Grupo":"B","Tiempo de prealistamiento (horas)":0.5,"Tiempo procesamiento (horas)":1},
    {"Grupo":"C","Tiempo de prealistamiento (horas)":0.5,"Tiempo procesamiento (horas)":1},
    {"Grupo":"D","Tiempo de prealistamiento (horas)":0.5,"Tiempo procesamiento (horas)":1.5},
])

selected_date = datetime(2025,11,17)
res = ap.plan_week_by_day(prueba, tiempo, selected_date, daily_cap=38)
schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day, df_progreso, registros_info = res

print('Columns in df_progreso:', list(df_progreso.columns))
print('\nRow for R1:')
print(df_progreso[df_progreso['Registro']=='R1'].T)
print('\nDtypes:')
print(df_progreso.dtypes)

# Also recompute group sums and show expected values
from app_planeacion import expand_analyses, split_large_records
exp = expand_analyses(prueba)
orig = split_large_records(exp, 38)
print('\nComputed group sums from df_state-like orig:')
print(orig.groupby(['Registro','Tipo de analisis'])['No muestras'].sum())

# Combine pendientes grouping from returned pendientes
print('\nPendientes grouping:')
print(pendientes.groupby(['Registro','Tipo de analisis'])[['Solicitadas','Procesadas','Pendiente']].sum())
