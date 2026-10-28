# Plan de Proyecto — SADOP-Muni

**Proyecto**: Sistema de Soporte de Decisiones para Obras Públicas (SADOP)
**Método**: Decisión Multicriterio Discreta (DMD) — Capítulo 12 bibliografía
**Stack Tecnológico**: Python 3.x + Streamlit + Pandas + NumPy + Plotly
**Última actualización**: 28/10/2026

El plan original preveía un paquete `models/` con normalización, AHP, SAW y
TOPSIS. En la implementación la lógica que no depende del método (modelo,
normalización, preanálisis, pesos) pasó a `core/`, y `models/` quedó solo
para los métodos de evaluación y la sensibilidad. Las tablas reflejan los
archivos reales.

---

## Fase 1 — Fundamentos y Estructura del Proyecto

| # | Tarea | Entregable | Estado |
|---|-------|-----------|--------|
| 1.1 | Estructura de carpetas | `app.py`, `core/`, `models/`, `.gitignore` | Hecho |
| 1.2 | Dependencias | `requirements.txt` | Hecho |
| 1.3 | Normalización: máximo, suma, rango, vectorial + criterios a minimizar | `core/normalizacion.py` | Hecho |
| 1.4 | Preanálisis de dominación y satisfacción | `core/preanalisis.py` | Hecho |
| 1.5 | Tests de la base | `tests/test_base.py` (23 pruebas) | Hecho |

## Fase 2 — Modelos Matemáticos

| # | Tarea | Entregable | Estado |
|---|-------|-----------|--------|
| 2.1 | AHP: prioridades (aproximada y autovector) + IC, IA, RC | `core/pesos.py` | Hecho |
| 2.2 | Ponderación Lineal | `models/ponderacion_lineal.py` | Hecho |
| 2.3 | TOPSIS con métricas m1, m2, m∞ | `models/topsis.py` | Hecho |
| 2.4 | Validar AHP con el ejemplo de las computadoras | `tests/test_metodos.py` | Hecho |
| 2.5 | Validar Ponderación Lineal y TOPSIS con el ejemplo del inversionista | `tests/test_metodos.py` | Hecho |
| 2.6 | Análisis de sensibilidad (barrido de pesos y escenarios) | `models/sensibilidad.py` | Hecho |
| 2.7 | AHP como método: juicios entre alternativas por criterio | `models/ahp.py` | Hecho |
| 2.8 | TOPSIS con ideal según el sentido (versión de clase) | `models/topsis.py` | Hecho |
| 2.9 | Asistente de consistencia (sugerencias de juicios) | `core/pesos.py` | Hecho |
| 2.10 | Comparación de métodos (Spearman, consenso) | `models/comparacion.py` | Hecho |

## Fase 3 — Interfaz Web con Streamlit

| # | Tarea | Entregable | Estado |
|---|-------|-----------|--------|
| 3.1 | Navegación lateral | 7 pasos | Hecho |
| 3.2 | Paso 1: definición del problema | Formularios editables | Hecho |
| 3.3 | Paso 2: matriz de decisión | Tabla editable + normalizada | Hecho |
| 3.4 | Paso 3: preanálisis | Dominación, satisfacción, conjunto de elección | Hecho |
| 3.5 | Paso 4: pesos (directos o AHP) | Juicios de Saaty + consistencia | Hecho |
| 3.6 | Paso 5: evaluación | Ranking, gráfico, paso a paso, CSV | Hecho |
| 3.7 | Paso 6: comparación de métodos | Gráfico de posiciones, acuerdo | Hecho |
| 3.8 | Paso 7: sensibilidad | Intervalos estables, gráfico de barrido, escenarios | Hecho |
| 3.9 | Caso por defecto pre-cargado | `data/caso_ejemplo.json` (6 obras, 6 criterios, pesos) | Hecho |

## Fase 4 — Verificación y Documentación

| # | Tarea | Entregable | Estado |
|---|-------|-----------|--------|
| 4.1 | Verificación con los ejemplos del Cap. 12 | Tabla en el README + tests | Hecho |
| 4.2 | Prueba de extremo a extremo con el caso por defecto | Recorrido de los 7 pasos | Hecho (sin capturas guardadas) |
| 4.3 | Documentación | README | Hecho |
| 4.4 | Reemplazar los valores del caso por datos reales de pliego | `data/` | Pendiente |
| 4.5 | Informe / presentación para la defensa | — | Pendiente |

---

## Dependencias y Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Cambios de API de Streamlit | Medio | Se fija `streamlit>=1.50`; la lógica en `core/` y `models/` no depende de Streamlit |
| Diferencias de redondeo con el libro | Bajo | Tolerancias justificadas en cada test; discrepancias documentadas en el README |
| AHP con más de 10 criterios | Bajo | No hay IA tabulado: se informa el error en la interfaz |

---

## Marco de Verificación

- **Algoritmos**: `pytest` corre las pruebas de la base y las de métodos, que
  reproducen los ejemplos del capítulo con error menor a 0,001 (0,002 en
  TOPSIS, donde el libro parte de valores redondeados a 3 decimales).
- **Interfaz**: `python -m streamlit run app.py` y recorrido de los 7 pasos
  con el caso por defecto, en los dos modos de pesos y con los tres métodos.
