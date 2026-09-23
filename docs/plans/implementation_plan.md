# Sistema de Soporte de Decisiones para Obras Públicas (SADOP)

El objetivo es desarrollar un software interactivo que sirva como sistema de soporte para la selección de obras públicas de una municipalidad. Este software aplicará los modelos de decisión multicriterio discreta (DMD) extraídos del Capítulo 12 de la bibliografía proporcionada.

## User Review Required

> [!IMPORTANT]
> **Stack tecnológico**: Propongo utilizar **Python con el framework Streamlit**. Streamlit es excelente para construir aplicaciones web interactivas orientadas a datos y modelos matemáticos de forma muy rápida y con una interfaz limpia y moderna. ¿Estás de acuerdo con este enfoque, o prefieres otro tipo de aplicación (por ejemplo, aplicación de escritorio con PyQt, o web con React/Node)?

## Open Questions

- ¿Deseas incluir un **caso de estudio por defecto** pre-cargado en el software (por ejemplo: "Pavimentación", "Hospital", "Plaza", "Escuela") con sus respectivos datos ficticios para facilitar la presentación a los profesores?
- ¿Te gustaría que el software permita exportar/descargar los resultados finales en algún formato específico (como CSV o PDF)?

## Proposed Changes

La aplicación estará estructurada en una arquitectura limpia separando la interfaz de usuario de la lógica de los modelos matemáticos.

---

### Módulos de Modelos Matemáticos

#### [NEW] `models/normalization.py`
Implementación de las funciones para los distintos tipos de normalización requeridos para hacer comparables los criterios:
- Proporcional al máximo ($r_{ij} = a_{ij} / \max(a_{ij})$).
- Proporcional a la suma ($r_{ij} = a_{ij} / \sum a_{ij}$).
- Proporción por rango (Min-Max).
- Normalización vectorial (utilizada habitualmente por TOPSIS).

#### [NEW] `models/ahp.py`
Implementación del método AHP (Proceso de Análisis Jerárquico) para:
- Calcular los pesos (eigenvector) de los criterios a partir de la matriz de comparaciones pareadas de Saaty.
- Calcular y verificar el Ratio de Consistencia (RC) para asegurar que los juicios del decisor son lógicos.

#### [NEW] `models/saw.py`
Implementación del método de Ponderación Lineal (Simple Additive Weighting).

#### [NEW] `models/topsis.py`
Implementación del método TOPSIS calculando:
- Solución ideal positiva y negativa.
- Distancias (permitiendo elegir la métrica: Euclidiana, Ciudad/Manhattan, etc.).
- Índice de similaridad y ranking final.

---

### Interfaz Web (Streamlit)

#### [NEW] `app.py`
Punto de entrada de la interfaz gráfica. Contendrá un menú lateral con el siguiente flujo paso a paso:
1. **Configuración del Problema**: Definir las alternativas (obras públicas) y criterios (costo, impacto social, impacto ambiental, tiempo de ejecución, etc.), indicando si son a maximizar o minimizar.
2. **Matriz de Decisión**: Tabla interactiva para cargar las valoraciones de cada alternativa respecto a cada criterio.
3. **Ponderación de Criterios**: Elección entre asignación directa de pesos o cálculo mediante comparaciones pareadas (AHP).
4. **Evaluación y Resultados**: Selección del método de normalización y del modelo de evaluación (Ponderación Lineal o TOPSIS). Visualización del ranking final mediante tablas y gráficos de barras.

#### [NEW] `requirements.txt`
Archivo con las dependencias de Python necesarias:
- `streamlit`
- `pandas`
- `numpy`
- `plotly` (para gráficos interactivos)

## Verification Plan

### Automated Tests
- Por tratarse de un proyecto académico enfocado en la interfaz y aplicación, no se configurará una suite formal de tests automatizados a menos que lo solicites.

### Manual Verification
- Se validarán los cálculos de los algoritmos (AHP, Normalizaciones, TOPSIS y SAW) introduciendo los mismos ejemplos numéricos que aparecen en el Capítulo 12 del PDF de la bibliografía para corroborar que los resultados sean idénticos.
- Se ejecutará la aplicación Streamlit localmente, simulando de principio a fin la carga de 4 obras públicas y comprobando la correcta visualización de los resultados y el funcionamiento de la interfaz.
