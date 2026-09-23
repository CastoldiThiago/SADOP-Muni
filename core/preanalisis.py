"""Preanálisis de dominación y de satisfacción (Cap. 12, secciones 9 y 10).

Ambos filtros son **independientes del método de evaluación** y de los pesos:
reducen el conjunto de elección antes de aplicar Ponderación Lineal, AHP o
TOPSIS. Conviene correrlos primero porque:

* una alternativa dominada no puede ganar bajo ningún método compensatorio
  con pesos positivos, así que evaluarla es trabajo perdido;
* dejan documentado *por qué* quedaron afuera algunas alternativas, que es
  parte del resultado y no un descarte silencioso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modelo import ProblemaDecision, Sentido


@dataclass
class ResultadoDominacion:
    """Qué alternativas son eficientes y cuáles quedaron dominadas."""

    #: Índices de fila de las alternativas no dominadas (eficientes).
    eficientes: list[int]
    #: Índices de fila de las alternativas dominadas.
    dominadas: list[int]
    #: Para cada dominada, los índices de las alternativas que la dominan.
    dominada_por: dict[int, list[int]]

    def __bool__(self) -> bool:
        return bool(self.dominadas)


@dataclass
class ResultadoSatisfaccion:
    """Qué alternativas alcanzan los niveles de aspiración del decisor."""

    #: Índices de fila de las alternativas que cumplen todos los umbrales.
    satisfacen: list[int]
    #: Índices de fila de las alternativas que incumplen alguno.
    descartadas: list[int]
    #: Para cada descartada, los índices de los criterios que no alcanza.
    incumple: dict[int, list[int]]

    def __bool__(self) -> bool:
        return bool(self.descartadas)


def _orientada(A: np.ndarray, sentidos: list[Sentido]) -> np.ndarray:
    """Matriz con el signo dado vuelta en los criterios a minimizar.

    Para dominación no hace falta normalizar: alcanza con que *más sea mejor*
    en todas las columnas, y negar preserva el orden sin introducir ninguna
    de las distorsiones que sí tiene elegir una política de conversión.
    """
    A = np.asarray(A, dtype=float)
    if len(sentidos) != A.shape[1]:
        raise ValueError(
            f"La matriz tiene {A.shape[1]} criterios pero se recibieron "
            f"{len(sentidos)} sentidos."
        )
    signos = np.array(
        [1.0 if Sentido(s) is Sentido.MAXIMIZAR else -1.0 for s in sentidos]
    )
    return A * signos


def dominacion(A: np.ndarray, sentidos: list[Sentido]) -> ResultadoDominacion:
    """Preanálisis de dominación (sección 9).

    Una alternativa ``k`` está **dominada** por ``i`` si ``i`` la iguala o
    supera en todos los criterios y la supera estrictamente en al menos uno.
    Las no dominadas forman el conjunto eficiente (óptimo de Pareto), que es
    el único sobre el que tiene sentido seguir discutiendo pesos.
    """
    M = _orientada(A, sentidos)
    m = M.shape[0]

    dominada_por: dict[int, list[int]] = {}
    for k in range(m):
        dominantes = [
            i
            for i in range(m)
            if i != k
            and np.all(M[i] >= M[k])
            and np.any(M[i] > M[k])
        ]
        if dominantes:
            dominada_por[k] = dominantes

    dominadas = sorted(dominada_por)
    eficientes = [i for i in range(m) if i not in dominada_por]
    return ResultadoDominacion(eficientes, dominadas, dominada_por)


def satisfaccion(
    A: np.ndarray,
    sentidos: list[Sentido],
    umbrales: list[float | None],
) -> ResultadoSatisfaccion:
    """Preanálisis de satisfacción (sección 10).

    Descarta las alternativas que no alcanzan los niveles de aspiración del
    decisor. Un umbral ``None`` significa que ese criterio no impone
    restricción. Para un criterio a maximizar se exige ``a_ij >= umbral``;
    para uno a minimizar, ``a_ij <= umbral``.

    Es un filtro **no compensatorio**: un desempeño excelente en un criterio
    no rescata a una alternativa que no llega al mínimo en otro.
    """
    A = np.asarray(A, dtype=float)
    if len(umbrales) != A.shape[1]:
        raise ValueError(
            f"La matriz tiene {A.shape[1]} criterios pero se recibieron "
            f"{len(umbrales)} umbrales."
        )

    incumple: dict[int, list[int]] = {}
    for i in range(A.shape[0]):
        fallas = [
            j
            for j, umbral in enumerate(umbrales)
            if umbral is not None
            and (
                A[i, j] < umbral
                if Sentido(sentidos[j]) is Sentido.MAXIMIZAR
                else A[i, j] > umbral
            )
        ]
        if fallas:
            incumple[i] = fallas

    descartadas = sorted(incumple)
    satisfacen = [i for i in range(A.shape[0]) if i not in incumple]
    return ResultadoSatisfaccion(satisfacen, descartadas, incumple)


# ------------------------------------------------------- informe legible


def informe(problema: ProblemaDecision) -> list[str]:
    """Corre ambos preanálisis sobre un problema y describe los descartes.

    Devuelve líneas listas para mostrar en la interfaz o en el informe final.
    """
    A = problema.matriz()
    sentidos = problema.sentidos
    nombres = [a.nombre for a in problema.alternativas]
    crits = [c.nombre for c in problema.criterios]
    lineas: list[str] = []

    dom = dominacion(A, sentidos)
    if dom.dominadas:
        for k, dominantes in dom.dominada_por.items():
            por = ", ".join(nombres[i] for i in dominantes)
            lineas.append(f"'{nombres[k]}' está dominada por: {por}.")
    else:
        lineas.append("Ninguna alternativa está dominada: todas son eficientes.")

    umbrales = [c.umbral for c in problema.criterios]
    if any(u is not None for u in umbrales):
        sat = satisfaccion(A, sentidos, umbrales)
        for i, fallas in sat.incumple.items():
            detalle = ", ".join(
                f"{crits[j]} ({A[i, j]:g} vs. umbral {umbrales[j]:g})" for j in fallas
            )
            lineas.append(f"'{nombres[i]}' no satisface: {detalle}.")
        if not sat.descartadas:
            lineas.append("Todas las alternativas alcanzan los umbrales fijados.")
    else:
        lineas.append("No se fijaron umbrales de satisfacción.")

    return lineas


def sobrevivientes(problema: ProblemaDecision) -> list[int]:
    """Índices de las alternativas que pasan ambos filtros.

    Si los filtros dejaran menos de dos alternativas, se devuelve el conjunto
    eficiente completo: es preferible seguir evaluando a que la aplicación
    "decida" por descarte, que es justo lo que el trabajo quiere evitar.
    """
    A = problema.matriz()
    sentidos = problema.sentidos
    eficientes = set(dominacion(A, sentidos).eficientes)

    umbrales = [c.umbral for c in problema.criterios]
    if any(u is not None for u in umbrales):
        eficientes &= set(satisfaccion(A, sentidos, umbrales).satisfacen)

    if len(eficientes) < 2:
        return dominacion(A, sentidos).eficientes
    return sorted(eficientes)
