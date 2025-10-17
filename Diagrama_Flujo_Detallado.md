# DIAGRAMA DE FLUJO DETALLADO - SISTEMA DE PLANEACIÓN DE LABORATORIO

## INSTRUCCIONES PARA CREAR DIAGRAMA EN WORD

### HERRAMIENTAS NECESARIAS:
- Microsoft Word con SmartArt o Visio
- Formas de diagrama de flujo
- Conectores direccionales

---

## 🔄 DIAGRAMA PRINCIPAL DEL PROCESO

### **NIVEL 1: FLUJO GENERAL DEL SISTEMA**

```
[INICIO] 
    ↓
[Cargar Datos de Excel]
    ↓
[Expandir Análisis A, B, C, D]
    ↓
[Ordenar FIFO por Fecha + Registro]
    ↓
[Configurar Capacidad = 38 muestras/día]
    ↓
[BUCLE: Para cada día Lunes-Jueves] ←──┐
    ↓                                   │
[Aplicar Restricciones por Día]         │
    ↓                                   │
[PROCESO DE SELECCIÓN DIARIA]           │
    ↓                                   │
[Programar Sesiones de Trabajo]         │
    ↓                                   │
[Actualizar Estado de Pendientes]       │
    ↓                                   │
[¿Es último día?] ──NO──────────────────┘
    ↓ SÍ
[Verificar Registros Completos]
    ↓
[Generar Reporte Final]
    ↓
[FIN]
```

---

## 📅 DIAGRAMA DE RESTRICCIONES DIARIAS

### **Decisión por Día de la Semana:**

```
[Día de la Semana]
    ↓
┌─────────────┬─────────────┬──────────────┬──────────────┐
│    LUNES    │   MARTES    │  MIÉRCOLES   │   JUEVES     │
│             │             │              │              │
│ Solo A y D  │ Solo B y C  │ Todos los    │ Todos los    │
│ (D exclusivo)│(Combinables)│ grupos       │ grupos       │
│             │             │(D exclusivo) │(D exclusivo) │
└─────────────┴─────────────┴──────────────┴──────────────┘
    ↓              ↓              ↓              ↓
[Filtrar       [Filtrar      [Todos         [Todos
 grupos A,D]    grupos B,C]   disponibles]   disponibles]
    ↓              ↓              ↓              ↓
[Proceso de    [Proceso de   [Proceso de    [Proceso de
 Selección]     Selección]    Selección]     Selección]
```

---

## 🎯 DIAGRAMA DE PROCESO DE SELECCIÓN DIARIA

### **Algoritmo de 3 Pasos:**

```
[Registros Filtrados por Día]
    ↓
[PASO 1: Buscar Registros ≤38 muestras]
    ↓
[¿Hay registros completos ≤38?] ──NO──┐
    ↓ SÍ                               │
[Seleccionar por Prioridad FIFO]       │
    ↓                                  │
[Agregar a Sesión del Día]             │
    ↓                                  │
[¿Capacidad Completa (38)?] ──SÍ──┐    │
    ↓ NO                          │    │
[¿Queda Espacio Significativo?]   │    │
    ↓ SÍ                          │    │
[Volver a buscar ≤38] ────────────┘    │
    ↓ NO                               │
┌──────────────────────────────────────┘
│
└→ [PASO 2: Evaluar Registros >38]
    ↓
[¿Existen registros >38?] ──NO──┐
    ↓ SÍ                        │
[Aplicar Fragmentación          │
 Inteligente]                   │
    ↓                           │
[¿Se debe fragmentar?] ──NO──┐  │
    ↓ SÍ                     │  │
[Calcular Fragmento Óptimo]  │  │
    ↓                        │  │
[Agregar Fragmento]          │  │
    ↓                        │  │
┌────────────────────────────┘  │
│                               │
└→ [PASO 3: Verificar Umbral 60%] ←─┘
    ↓
[¿Alcanza 23 muestras mínimo?] ──SÍ──┐
    ↓ NO                              │
[Buscar Fragmentos Urgentes ≥10 días] │
    ↓                                 │
[¿Puede alcanzar umbral?] ──NO──┐     │
    ↓ SÍ                        │     │
[Agregar Fragmento Urgente]     │     │
    ↓                           │     │
┌───────────────────────────────┘     │
│                                     │
└→ [PROGRAMAR SESIONES] ←──────────────┘
```

---

## 🧠 DIAGRAMA DE FRAGMENTACIÓN INTELIGENTE

### **Evaluación Multi-Criterio:**

```
[Registro >38 muestras]
    ↓
[Calcular espacio disponible en día]
    ↓
[Grupos completos = espacio ÷ 38]
    ↓
[¿Grupos completos > 0?] ──NO──→ [NO FRAGMENTAR]
    ↓ SÍ
[Fragmento = min(grupos×38, total)]
    ↓
[CRITERIO 1: ¿Antigüedad ≥15 días?] ──SÍ──┐
    ↓ NO                                   │
[CRITERIO 2: ¿Porcentaje ≥60%?] ──SÍ──┐   │
    ↓ NO                               │   │
[CRITERIO 3: ¿Espacio ≥76 AND          │   │
 resto ≥38?] ──SÍ──┐                   │   │
    ↓ NO           │                   │   │
[CRITERIO 4:       │                   │   │
 ¿Resto <38?] ──SÍ─┤                   │   │
    ↓ NO           │                   │   │
[CRITERIO 5:       │                   │   │
 ¿Ayuda umbral?]──SÍ┤                   │   │
    ↓ NO           │                   │   │
[NO FRAGMENTAR]    │                   │   │
                   │                   │   │
                   └→ [FRAGMENTAR] ←────┴───┘
                        ↓
                   [Aplicar fragmento calculado]
                        ↓
                   [Actualizar pendientes]
```

---

## ⏰ DIAGRAMA DE PROGRAMACIÓN TEMPORAL

### **Asignación de Horarios:**

```
[Registros Seleccionados del Día]
    ↓
[¿Contiene Grupo D?] ──SÍ──┐
    ↓ NO                   │
[Agrupar tipos compatibles]│
    ↓                      │
[¿Es Martes con B+C?] ──SÍ─┤
    ↓ NO                   │
[Sesiones separadas        │
 por grupo]                │
    ↓                      │
[Calcular tiempos:         │
 Prealist(2h) + Proc(6h)] ←┘
    ↓
[Optimizar concurrencia:
 Prealist mientras Proc]
    ↓
[Total tiempo = Σ(tiempos grupos)]
    ↓
[¿Excede 10 horas/día?] ──SÍ──→ [Diferir registros
    ↓ NO                        a día siguiente]
[Confirmar programación]             ↓
    ↓                          [Actualizar
[Generar cronograma detallado]   planificación]
    ↓                               ↓
[Asignar horarios específicos] ←────┘
```

---

## 📊 DIAGRAMA DE VERIFICACIÓN DE ENTREGAS

### **Control de Registros Completos:**

```
[Estado Final de la Semana]
    ↓
[Para cada Registro único]
    ↓
[¿Tiene grupos A, B, C, D procesados?] ──NO──┐
    ↓ SÍ                                     │
[Marcar como COMPLETO - LISTO ENTREGA]       │
    ↓                                        │
[Calcular tiempo total de ciclo]             │
    ↓                                        │
[Agregar a lista de entregas] ←──────────────┘
    ↓                           ↓
[¿Quedan registros?] ──SÍ──→ [Marcar como EN PROCESO
    ↓ NO                     o PENDIENTE]
[Generar Reporte Final]           ↓
    ↓                        [Calcular % progreso]
[Mostrar estadísticas:            ↓
 - Registros entregados      [Programar para
 - En proceso                 próxima semana]
 - Utilización promedio]
    ↓
[Crear Gráfico Gantt]
    ↓
[FIN DEL PROCESO]
```

---

## 🎨 ELEMENTOS VISUALES PARA WORD

### **FORMAS RECOMENDADAS:**

**Inicio/Fin:** 
- Óvalo redondeado
- Color: Verde (inicio), Rojo (fin)

**Procesos:**
- Rectángulo 
- Color: Azul claro

**Decisiones:**
- Rombo
- Color: Amarillo

**Datos/Documentos:**
- Paralelogramo
- Color: Gris claro

**Conectores:**
- Flechas direccionales
- Líneas rectas
- Etiquetas SÍ/NO en decisiones

### **COLORES SUGERIDOS:**
- **Verde**: Inicio, procesos exitosos
- **Azul**: Procesos principales  
- **Amarillo**: Decisiones/evaluaciones
- **Naranja**: Procesos especiales (fragmentación)
- **Rojo**: Fin, alertas
- **Gris**: Datos, estados temporales

### **LAYOUT RECOMENDADO:**
- **Orientación**: Vertical principal con ramificaciones horizontales
- **Espaciado**: Uniforme entre elementos
- **Agrupación**: Usar cajas de grupo para subprocesos
- **Texto**: Fuente Arial 10-12pt, negrita para títulos

---

## 📝 INSTRUCCIONES DE CONSTRUCCIÓN

### **PASO A PASO EN WORD:**

1. **Insertar → SmartArt → Proceso → Proceso Básico**
2. **Personalizar formas según tipo de elemento**
3. **Agregar texto descriptivo en cada forma**
4. **Conectar con flechas direccionales**
5. **Aplicar colores según categoría**
6. **Agrupar subprocesos en cajas contenedoras**
7. **Añadir leyenda de colores**
8. **Revisar flujo lógico completo**

### **VALIDACIÓN DEL DIAGRAMA:**
- ✅ Todos los caminos llevan a una conclusión
- ✅ Las decisiones tienen salidas SÍ/NO claras
- ✅ Los bucles tienen condiciones de salida
- ✅ El flujo sigue la lógica del algoritmo
- ✅ Subprocesos están claramente definidos

---

*Guía para Diagrama de Flujo - Sistema de Planeación de Laboratorio*  
*Versión: 3.0 - Octubre 2025*