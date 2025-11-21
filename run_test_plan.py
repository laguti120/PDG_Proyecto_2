import pandas as pd
from datetime import datetime
import traceback

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

try:
    res = ap.plan_week_by_day(prueba, tiempo, selected_date, daily_cap=38)
    schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day, df_progreso, registros_info = res

    print('\n--- schedule_week ---')
    print(schedule_week)
    print('\n--- util_df ---')
    print(util_df)
    print('\n--- df_progreso ---')
    print(df_progreso)
    print('\n--- pendientes ---')
    print(pendientes)

except Exception as e:
    print('Exception during execution:')
    traceback.print_exc()
