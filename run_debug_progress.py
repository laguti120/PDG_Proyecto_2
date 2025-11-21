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

schedule_week, pendientes, util_df, gantt_week_df, gantt_per_day, df_progreso, registros_info = ap.plan_week_by_day(prueba, tiempo, selected_date, daily_cap=38)

print('\n--- df_state (final) sample ---')
# df_state is internal; replicate logic to rebuild df_original and df_state from module functions
from app_planeacion import expand_analyses, split_large_records

df_expanded = expand_analyses(prueba)
print('df_expanded:')
print(df_expanded[['Registro','ID analisis','Tipo de analisis','No muestras']])

# After fragmentation
df_original = split_large_records(df_expanded, 38)
print('\n df_original (after split):')
print(df_original[['Registro','ID analisis','Tipo de analisis','No muestras']])

# Simulate df_state by copying and subtracting processed from pendientes output
print('\n pendings table from run:')
print(pendientes)

# Show grouped sums per Registro
for reg in df_original['Registro'].unique():
    reg_data = df_original[df_original['Registro']==reg].copy()
    print(f"\nReg {reg} reg_data before processing:")
    print(reg_data[['ID analisis','Tipo de analisis','No muestras']])

# Now compare with pendientes (which has Procesadas and Pendiente)
print('\n--- Combine pendientes info by Registro and Tipo ---')
pend = pendientes.copy()
pend_group = pend.groupby(['Registro','Tipo de analisis']).agg({'Solicitadas':'sum','Procesadas':'sum','Pendiente':'sum'}).reset_index()
print(pend_group)
