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
    minimos: list[float | None],
    maximos: list[float | None],
) -> ResultadoSatisfaccion:
    """Preanálisis de satisfacción (sección 10).

    Descarta las alternativas que quedan fuera del rango aceptable de algún
    criterio: por debajo de su piso (``minimos``) o por encima de su techo
    (``maximos``). ``None`` significa que ese lado no tiene límite.

    Es un filtro **no compensatorio**: un desempeño excelente en un criterio
    no rescata a una alternativa que queda fuera del rango en otro.
    """
    A = np.asarray(A, dtype=float)
    if not len(minimos) == len(maximos) == A.shape[1]:
        raise ValueError(
            f"La matriz tiene {A.shape[1]} criterios pero se recibieron "
            f"{len(minimos)} pisos y {len(maximos)} techos."
        )

    incumple: dict[int, list[int]] = {}
    for i in range(A.shape[0]):
        fallas = [
            j
            for j in range(A.shape[1])
            if (minimos[j] is not None and A[i, j] < minimos[j])
            or (maximos[j] is not None and A[i, j] > maximos[j])
        ]
        if fallas:
            incumple[i] = fallas

    descartadas = sorted(incumple)
    satisfacen = [i for i in range(A.shape[0]) if i not in incumple]
    return ResultadoSatisfaccion(satisfacen, descartadas, incumple)


def satisfaccion_del_problema(problema: ProblemaDecision) -> ResultadoSatisfaccion | None:
    """Satisfacción con los umbrales de los criterios, o ``None`` si no hay."""
    minimos = [c.umbral_min for c in problema.criterios]
    maximos = [c.umbral_max for c in problema.criterios]
    if all(u is None for u in minimos + maximos):
        return None
    return satisfaccion(problema.matriz(), minimos, maximos)


def motivo(valor: float, minimo: float | None, maximo: float | None) -> str:
    """Por qué un valor queda fuera del rango, en palabras."""
    if minimo is not None and valor < minimo:
        return f"{valor:g}, por debajo del mínimo {minimo:g}"
    if maximo is not None and valor > maximo:
        return f"{valor:g}, por encima del máximo {maximo:g}"
    return f"{valor:g}, dentro del rango"


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

    sat = satisfaccion_del_problema(problema)
    if sat is None:
        lineas.append("No se fijaron umbrales de satisfacción.")
    else:
        for i, fallas in sat.incumple.items():
            detalle = ", ".join(
                f"{crits[j]} ({motivo(A[i, j], *_rango(problema, j))})" for j in fallas
            )
            lineas.append(f"'{nombres[i]}' no satisface: {detalle}.")
        if not sat.descartadas:
            lineas.append("Todas las alternativas están dentro de los umbrales fijados.")

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

    sat = satisfaccion_del_problema(problema)
    if sat is not None:
        eficientes &= set(sat.satisfacen)

    if len(eficientes) < 2:
        return dominacion(A, sentidos).eficientes
    return sorted(eficientes)


def _rango(problema: ProblemaDecision, j: int) -> tuple[float | None, float | None]:
    c = problema.criterios[j]
    return c.umbral_min, c.umbral_max
