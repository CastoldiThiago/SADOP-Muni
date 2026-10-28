# SADOP-Muni

Sistema de soporte a la decisión para **priorizar obras públicas municipales**,
basado en Decisión Multicriterio Discreta (Cap. 12 de la bibliografía de la
materia *Decisiones en Escenarios Complejos*).

## Estado

El flujo completo está implementado, de la definición del problema al análisis
de sensibilidad, y los métodos están validados contra los ejemplos numéricos
del Cap. 12.

| Etapa | Estado |
|---|---|
| Definición del problema (alternativas, criterios, sentido, escala, umbrales) | Hecho |
| Matriz de decisión + validación | Hecho |
| Conversión de criterios a minimizar (3 políticas) | Hecho |
| Normalización (4 procedimientos) | Hecho |
| Preanálisis de dominación y satisfacción | Hecho |
| Asignación de pesos: directa o comparaciones pareadas (AHP) con RC | Hecho |
| Asistente de consistencia: qué juicio revisar y con qué valor | Hecho |
| Ponderación Lineal | Hecho |
| AHP completo: juicios entre alternativas por criterio, o datos | Hecho |
| TOPSIS (m1, m2, m∞; ideal de clase o del Cap. 12) | Hecho |
| Comparación de métodos con acuerdo de Spearman | Hecho |
| Análisis de sensibilidad: pesos y elecciones del analista | Hecho |
| Persistencia del caso en JSON (incluye pesos y juicios AHP) | Hecho |

## Instalación y uso

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Se invoca como `python -m streamlit` y no como `streamlit` a secas porque,
cuando pip instala en el perfil del usuario, el ejecutable queda en un
directorio que suele no estar en el PATH. La forma con `-m` no depende del
PATH.

Para trabajar solo sobre la lógica alcanza con `numpy` y `pandas`.

```bash
pytest                        # o: python tests/test_base.py / test_metodos.py
```

## Estructura

```
SADOP-Muni/
├── app.py                    Interfaz Streamlit (7 pasos)
├── core/                     Lógica independiente del método
│   ├── modelo.py             Alternativa, Criterio, ProblemaDecision
│   ├── normalizacion.py      Procedimientos 1-3 + vectorial; criterios a minimizar
│   ├── preanalisis.py        Dominación (secc. 9) y satisfacción (secc. 10)
│   ├── pesos.py              Pesos directos y AHP: prioridades, consistencia, sugerencias
│   └── io_casos.py           Carga y guardado de casos en JSON
├── models/
│   ├── base.py               Contrato de los métodos de evaluación
│   ├── ponderacion_lineal.py Secc. 13
│   ├── ahp.py                Secc. 14-15
│   ├── topsis.py             Secc. 16
│   ├── comparacion.py        Mismo caso, distintos métodos
│   └── sensibilidad.py       Barrido de pesos y escenarios
├── data/caso_ejemplo.json    Caso de referencia: 6 obras, 6 criterios, pesos
├── tests/
│   ├── test_base.py          Base independiente del método
│   └── test_metodos.py       Pesos, métodos y sensibilidad vs. el capítulo
└── docs/                     Planes y bibliografía
```

## Flujo de la interfaz

1. **Definición del problema**: alternativas, criterios, sentido y umbrales.
2. **Matriz de decisión**: evaluaciones en unidades originales.
3. **Preanálisis**: descarta dominadas y las que no alcanzan los umbrales.
4. **Pesos**: asignación directa o juicios de a pares en escala de Saaty, con
   λmax, IC, IA y RC. Si RC > 0,10, el **asistente de consistencia** muestra
   los juicios que más bajan el RC y permite aplicarlos con un clic.
5. **Evaluación**: método y sus opciones; ranking, gráfico, paso a paso y
   descarga en CSV. Las opciones que el método no usa quedan deshabilitadas.
6. **Comparación de métodos**: Ponderación Lineal, AHP y TOPSIS (m1, m2, m∞)
   sobre los mismos datos y pesos, con un gráfico de posiciones y el
   acuerdo entre rankings (Spearman).
7. **Sensibilidad**: intervalo de pesos en el que la ganadora se mantiene,
   para cada criterio, y todas las combinaciones de método, normalización y
   política.

### AHP

Sigue las cinco etapas de la presentación de la cátedra. Los pesos de los
criterios salen del paso 4. Las prioridades de las alternativas se eligen
**criterio por criterio**:

- **De a pares**, como en los ejemplos de clase: conviene en los criterios
  cualitativos (urgencia social, impacto ambiental). Cada matriz muestra su
  RC y el asistente de consistencia.
- **Datos de la matriz** normalizados por la suma: conviene en los criterios
  ya medidos (costo, plazo), donde pedir juicios sería pedir lo que ya se
  sabe. Sin ningún criterio de a pares, AHP coincide con Ponderación Lineal
  con normalización por la suma, que es como lo define el capítulo.

### TOPSIS: dos formas de definir el ideal

- **Clase** (por defecto): se normaliza sin convertir y el ideal toma el
  máximo en los criterios a maximizar y el mínimo en los de minimizar. No
  depende de cómo se conviertan los criterios a minimizar.
- **Cap. 12**: primero se convierte todo a maximizar con la política elegida
  y el ideal es siempre el máximo. Es la que reproduce el ejemplo del libro.

Con normalización por rango y complemento ambas dan exactamente lo mismo,
y hay una prueba que lo comprueba.

## Validación contra el Cap. 12

`tests/test_metodos.py` reproduce los tres ejemplos del capítulo:

| Ejemplo | Resultado del libro | Obtenido |
|---|---|---|
| Ponderación Lineal, inversionista (Tablas 6-9) | 0.306 / 0.206 / 0.279 / 0.208 | 0.306 / 0.206 / 0.279 / 0.209 |
| AHP, pesos de criterios | 0.4888 / 0.1182 / 0.2453 / 0.1478 | igual |
| AHP, λmax, IC y RC | 4.1345, 0.0448 y 0.0498 | igual |
| AHP, síntesis global | 0.3163 / 0.3951 / 0.2889 | 0.3161 / 0.3951 / 0.2889 |
| TOPSIS m1 (Cap. 12), inversionista (Tabla 13) | 0.676 / 0.251 / 0.562 / 0.261 | 0.677 / 0.252 / 0.562 / 0.262 |
| Método AHP completo, computadoras | IBM gana con 0.3951 | igual |

Las diferencias en el tercer o cuarto decimal vienen de que el libro redondea
los valores intermedios. Para tener en cuenta en la defensa:

- **Índice aleatorio**: se usa la tabla de Saaty de la cátedra (n = 4 →
  0,90), que es con la que el capítulo resuelve el ejemplo, y no la de sus
  Tablas 11 y 12 (0,89).
- **Síntesis de HP**: el libro informa 0,3163, pero rehaciendo la cuenta con
  sus propios valores redondeados da 0,3161.

También se comprueba lo que el capítulo afirma sin demostrar: TOPSIS con
métrica m1 ordena igual que la Ponderación Lineal. Además se resuelve el
ejercicio que propone el libro: con m∞, el Proyecto 3 desplaza al 1.

## Decisiones de diseño

**Los criterios a minimizar se tratan explícitamente.** El capítulo exige que
los métodos operen con todos los criterios a maximizar. La transformación
elegida (complemento, inversión, inversión relativa) **cambia el ranking**, así
que es un parámetro visible del análisis y no un detalle escondido. Hay una
prueba que documenta que dos políticas dan resultados distintos.

**La normalización es un parámetro, no un default.** El capítulo advierte que
"el resultado final puede verse considerablemente afectado por el procedimiento
de normalización utilizado". Están los cuatro procedimientos disponibles para
comparar.

**El preanálisis va antes que los pesos.** Dominación y satisfacción no
dependen del método ni de las ponderaciones, y dejan asentado por qué se
descartó cada alternativa en lugar de hacerlo en silencio. Si los filtros
dejaran menos de dos alternativas, se devuelve el conjunto eficiente completo:
el sistema no debe "decidir" por descarte.

**El método se enchufa, no se cablea.** Para agregar uno: implementar la
interfaz `MetodoDMD`, decorarlo con `@registrar` e importarlo en
`models/__init__.py`. El paso 5 de la interfaz lo detecta solo. Si el método
no usa la normalización o la política para minimizar, lo declara
(`usa_procedimiento = False`, `usa_politica = False`) y la interfaz y los
escenarios dejan de variar esa opción.

**Los pesos no dependen del método.** Viven en `core/pesos.py` y se guardan en
el caso. Ponderación Lineal y TOPSIS reciben el mismo vector, así que comparar
métodos compara métodos y no pesos distintos.

**La sensibilidad se informa como intervalo, no como gráfico suelto.** Para
cada criterio se calcula hasta dónde puede moverse su peso sin que cambie la
ganadora. Los puntos de cambio se refinan por bisección y un empate no cuenta
como cambio de ganadora.

## Caso de ejemplo

Seis obras evaluadas con seis criterios. Con los umbrales cargados, el
preanálisis reporta:

- *Plaza y centro recreativo* está **dominada** por *Recambio a luminarias LED*
  (peor en los seis criterios).
- *Ampliación guardia hospital* no satisface el tope de costo (420 vs. 400).
- *Plaza* tampoco alcanza el mínimo de urgencia social (2 vs. 3).

Quedan cuatro alternativas en el conjunto de elección. Con los pesos directos
del caso y Ponderación Lineal (normalización por el máximo, complemento),
*Recambio a luminarias LED* queda primera con 0,908, seguida de la escuela
(0,701). Los cinco métodos (Ponderación Lineal, AHP y TOPSIS con sus tres
métricas) coinciden en el ranking completo, y la ganadora se mantiene en 26 de
los 27 escenarios. El criterio más sensible es *Urgencia social*: si su peso
sube de 23 % a 62 %, pasa a ganar la escuela.

Los valores son
estimaciones de anteproyecto y deben reemplazarse por datos de pliego antes de
usar el resultado como insumo real.
