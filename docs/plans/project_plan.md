# Plan de Proyecto — SADOP-Muni

**Proyecto**: Sistema de Soporte de Decisiones para Obras Públicas (SADOP)
**Método**: Decisión Multicriterio Discreta (DMD) — Capítulo 12 bibliografía
**Stack Tecnológico**: Python 3.x + Streamlit + Pandas + NumPy + Plotly
**Duración estimada**: 4 semanas

---

## Fase 1 — Fundamentos y Estructura del Proyecto (Día 1–3)

### Objetivos
- Configurar el entorno de desarrollo y la estructura del proyecto
- Implementar el sistema de construcción de la matriz de decisión

### Tareas

| # | Tarea | Entregable | Estado |
|---|-------|-----------|--------|
| 1.1 | Inicializar proyecto con estructura de carpetas | `app.py`, `models/`, `requirements.txt` | Pending |
| 1.2 | Crear `requirements.txt` con dependencias | `streamlit`, `pandas`, `numpy`, `plotly` | Pending |
| 1.3 | Implementar módulo de normalización (`models/normalization.py`) | Funciones: proporcional máximo, proporcional suma, min-max, vectorial | Pending |
| 1.4 | Escribir tests manuales de normalización | Verificar con ejemplos del Cap. 12 | Pending |

### Criterios de éxito
- Todas las funciones de normalización producen resultados idénticos a los del Capítulo 12 de la bibliografía
- La estructura de carpetas sigue arquitectura limpia (UI separada de lógica)

---

## Fase 2 — Modelos Matemáticos (Día 4–8)

### Objetivos
- Implementar AHP, SAW y TOPSIS
- Validar cada modelo con ejemplos numéricos de la bibliografía

### Tareas

| # | Tarea | Entregable | Depende de | Estado |
|---|-------|-----------|------------|--------|
| 2.1 | Implementar AHP (`models/ahp.py`) — eigenvector + Ratio de Consistencia | Módulo AHP completo | 1.3 | Pending |
| 2.2 | Implementar SAW (`models/saw.py`) — ponderación lineal simple | Módulo SAW completo | 1.3 | Pending |
| 2.3 | Implementar TOPSIS (`models/topsis.py`) — soluciones ideales + distancias + ranking | Módulo TOPSIS completo | 1.3 | Pending |
| 2.4 | Validar AHP con ejemplos del Cap. 12 | Resultados coincidentes | 2.1 | Pending |
| 2.5 | Validar SAW y TOPSIS con ejemplos del Cap. 12 | Resultados coincidentes | 2.2, 2.3 | Pending |

### Criterios de éxito
- Los tres modelos (AHP, SAW, TOPSIS) producen resultados verificables contra la bibliografía
- El Ratio de Consistencia (RC) de AHP se calcula y reporta correctamente
- Se permite elegir métricas de distancia en TOPSIS (Euclidiana, Manhattan)

---

## Fase 3 — Interfaz Web con Streamlit (Día 9–13)

### Objetivos
- Construir la interfaz de usuario con flujo paso a paso
- Integrar todos los módulos matemáticos en la app

### Tareas

| # | Tarea | Entregable | Depende de | Estado |
|---|-------|-----------|------------|--------|
| 3.1 | Crear menú lateral y navegación en `app.py` | Estructura de navegación | 2.x | Pending |
| 3.2 | Implementar paso 1: Configuración del Problema (alternativas + criterios) | Formulario interactivo | 3.1 | Pending |
| 3.3 | Implementar paso 2: Matriz de Decisión (tabla interactiva) | Tabla editable con pandas | 3.2 | Pending |
| 3.4 | Implementar paso 3: Ponderación de Criterios (directa o AHP) | Selector de método + input | 3.3 | Pending |
| 3.5 | Implementar paso 4: Evaluación y Resultados (normalización + modelo + ranking) | Visualización con tablas y gráficos Plotly | 3.4, 2.x | Pending |
| 3.6 | Caso de estudio por defecto pre-cargado (Pavimentación, Hospital, Plaza, Escuela) | Datos ficticios inclusos | 3.5 | Pending |

### Criterios de éxito
- Flujo completo de 4 pasos funcional sin errores
- Caso de estudio por defecto carga correctamente
- Gráficos de barras interactivos muestran el ranking final
- Opción de exportar resultados a CSV

---

## Fase 4 — Verificación y Documentación (Día 14–16)

### Objetivos
- Validación completa del sistema
- Documentación final

### Tareas

| # | Tarea | Entregable | Depende de | Estado |
|---|-------|-----------|------------|--------|
| 4.1 | Verificación manual con ejemplos numéricos del Cap. 12 | Reporte de validación | 3.6 | Pending |
| 4.2 | Prueba de extremo a extremo (4 obras, flujo completo) | Screenshots + resultados | 4.1 | Pending |
| 4.3 | Documentación de la aplicación | README actualizado | 4.2 | Pending |
| 4.4 | Revisión final y ajustes | Código limpio y documentado | 4.3 | Pending |

### Criterios de éxito
- Todos los algoritmos producen resultados idénticos a los de la bibliografía
- La aplicación se ejecuta sin errores (`streamlit run app.py`)
- README con instrucciones de instalación y uso claras

---

## Dependencias y Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Problemas de compatibilidad con Streamlit | Medio | Probar con versión estable; tener fallback a script CLI |
| Errores de cálculo en AHP (eigenvector) | Alto | Validar con múltiples ejemplos de la bibliografía |
| Limitaciones de Streamlit con tablas grandes | Bajo | Usar st.data_editor para tablas pequeñas (4-10 filas) |

---

## Estructura del Proyecto

```
SADOP-Muni/
├── app.py                          # Punto de entrada — interfaz Streamlit
├── requirements.txt                # Dependencias Python
├── README.md                       # Documentación del proyecto
├── models/
│   ├── __init__.py                 # Exportaciones del paquete
│   ├── normalization.py            # Funciones de normalización
│   ├── ahp.py                      # Método AHP (Proceso de Análisis Jerárquico)
│   ├── saw.py                      # Método SAW (Simple Additive Weighting)
│   └── topsis.py                   # Método TOPSIS
├── data/
│   └── default_case.json           # Caso de estudio por defecto
└── docs/
    ├── plans/
    │   ├── implementation_plan.md
    │   └── project_plan.md         # Este archivo
    └── biblio/                     # Bibliografía en PDF
```

---

## Marco de Verificación

### Validación Algorítmica
- **Método**: Comparar resultados con ejemplos numéricos del Capítulo 12 de la bibliografía
- **Criterio**: Error < 0.001 en todos los cálculos
- **Herramienta**: Scripts de prueba ad-hoc con valores conocidos

### Validación de Interfaz
- **Método**: Ejecución manual completa de flujo de usuario
- **Criterio**: Todos los 4 pasos funcionan, resultados visibles y correctos
- **Herramienta**: Navegador web + `streamlit run app.py`
