"""Normalización de las evaluaciones (Cap. 12, sección 8).

La normalización transforma las evaluaciones a valores comparables entre sí
(habitualmente entre 0 y 1) para poder hacer comparaciones intra e
inter-atributos. El capítulo advierte que *"el resultado final puede verse
considerablemente afectado por el procedimiento de normalización utilizado"*,
razón por la cual acá se implementan todos los procedimientos y se deja que
el usuario compare, en vez de fijar uno por defecto y esconder la decisión.

Procedimientos de la Tabla 5 del capítulo:

===================  ==========================  ==================================
Procedimiento        Fórmula                     Característica
===================  ==========================  ==================================
1. Por el máximo     r_ij = a_ij / max_i a_ij    Respeta la proporcionalidad.
2. Por el rango      (a_ij - min) / (max - min)  Respeta la cardinalidad, no la
                                                 proporcionalidad.
3. Por la suma       r_ij = a_ij / sum_i a_ij    Usado por AHP; valores más chicos.
===================  ==========================  ==================================

Se agrega la normalización **vectorial** (raíz de la suma de cuadrados), que
el capítulo introduce en la sección 16 como una de las métricas admitidas por
TOPSIS.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from .modelo import Sentido


class Procedimiento(str, Enum):
    """Procedimiento de normalización a aplicar sobre la matriz."""

    MAXIMO = "maximo"
    RANGO = "rango"
    SUMA = "suma"
    VECTORIAL = "vectorial"

    @property
    def etiqueta(self) -> str:
        return {
            Procedimiento.MAXIMO: "Proporcional al máximo",
            Procedimiento.RANGO: "Proporción por rango (min-max)",
            Procedimiento.SUMA: "Proporcional a la suma",
            Procedimiento.VECTORIAL: "Vectorial (norma euclídea)",
        }[self]


class PoliticaMinimizar(str, Enum):
    """Cómo convertir un criterio a minimizar en uno a maximizar.

    Los métodos compensatorios y TOPSIS operan con *todos* los criterios a
    maximizar ("transformados previamente todos los criterios a maximizar y
    todas las a_ij a valores mayores o iguales a cero", sección 16). La
    transformación elegida **cambia el ranking**, así que es una decisión
    explícita del analista y no un detalle de implementación.
    """

    #: r = max(a) - a. Conserva las diferencias absolutas (cardinalidad).
    COMPLEMENTO = "complemento"
    #: r = 1 / a. Conserva las razones (proporcionalidad). Requiere a_ij > 0.
    INVERSION = "inversion"
    #: r = min(a) / a. Como la inversión, pero deja el mejor valor en 1.
    INVERSION_RELATIVA = "inversion_relativa"

    @property
    def etiqueta(self) -> str:
        return {
            PoliticaMinimizar.COMPLEMENTO: "Complemento (máx - a)",
            PoliticaMinimizar.INVERSION: "Inversión (1 / a)",
            PoliticaMinimizar.INVERSION_RELATIVA: "Inversión relativa (mín / a)",
        }[self]


class ErrorDeNormalizacion(ValueError):
    """La matriz no admite el procedimiento de normalización pedido."""


# --------------------------------------------------------------- orientación


def orientar(
    A: np.ndarray,
    sentidos: list[Sentido],
    politica: PoliticaMinimizar = PoliticaMinimizar.COMPLEMENTO,
) -> np.ndarray:
    """Devuelve una matriz con todos los criterios a maximizar.

    Las columnas cuyo sentido ya es ``MAXIMIZAR`` se copian sin tocar.
    """
    A = _como_matriz(A, sentidos)
    R = A.astype(float).copy()

    for j, sentido in enumerate(sentidos):
        if Sentido(sentido) is Sentido.MAXIMIZAR:
            continue
        col = A[:, j]
        if politica is PoliticaMinimizar.COMPLEMENTO:
            R[:, j] = col.max() - col
        else:
            if np.any(col <= 0):
                raise ErrorDeNormalizacion(
                    f"La columna {j} tiene valores <= 0 y la política "
                    f"'{politica.value}' requiere valores estrictamente "
                    "positivos. Usá COMPLEMENTO para esa columna."
                )
            R[:, j] = (col.min() if politica is PoliticaMinimizar.INVERSION_RELATIVA else 1.0) / col
    return R


# ------------------------------------------------------------ procedimientos


def por_maximo(A: np.ndarray) -> np.ndarray:
    """Procedimiento 1: r_ij = a_ij / max_i a_ij. Respeta la proporcionalidad."""
    A = _como_matriz(A)
    maximos = A.max(axis=0)
    _verificar_no_nulo(maximos, "el máximo")
    return A / maximos


def por_rango(A: np.ndarray) -> np.ndarray:
    """Procedimiento 2: r_ij = (a_ij - min) / (max - min). Deja el peor en 0."""
    A = _como_matriz(A)
    minimos, maximos = A.min(axis=0), A.max(axis=0)
    rango = maximos - minimos
    if np.any(np.isclose(rango, 0.0)):
        cols = np.where(np.isclose(rango, 0.0))[0].tolist()
        raise ErrorDeNormalizacion(
            f"Las columnas {cols} tienen rango nulo (todas las alternativas "
            "valen lo mismo): no se pueden normalizar por rango."
        )
    return (A - minimos) / rango


def por_suma(A: np.ndarray) -> np.ndarray:
    """Procedimiento 3: r_ij = a_ij / sum_i a_ij. Es el que usa AHP."""
    A = _como_matriz(A)
    sumas = A.sum(axis=0)
    _verificar_no_nulo(sumas, "la suma")
    return A / sumas


def vectorial(A: np.ndarray) -> np.ndarray:
    """Normalización vectorial: r_ij = a_ij / sqrt(sum_i a_ij^2) (sección 16)."""
    A = _como_matriz(A)
    normas = np.sqrt((A**2).sum(axis=0))
    _verificar_no_nulo(normas, "la norma")
    return A / normas


_PROCEDIMIENTOS = {
    Procedimiento.MAXIMO: por_maximo,
    Procedimiento.RANGO: por_rango,
    Procedimiento.SUMA: por_suma,
    Procedimiento.VECTORIAL: vectorial,
}


def escalar(A: np.ndarray, procedimiento: Procedimiento) -> np.ndarray:
    """Aplica el procedimiento sin orientar los criterios.

    Cada columna conserva su sentido original (en las de minimizar, menos
    sigue siendo mejor). Es lo que necesita TOPSIS cuando el ideal se toma
    según el sentido de cada criterio.
    """
    return _PROCEDIMIENTOS[Procedimiento(procedimiento)](A)


def normalizar(
    A: np.ndarray,
    sentidos: list[Sentido],
    procedimiento: Procedimiento = Procedimiento.MAXIMO,
    politica: PoliticaMinimizar = PoliticaMinimizar.COMPLEMENTO,
) -> np.ndarray:
    """Orienta todos los criterios a maximizar y después normaliza.

    Es el punto de entrada que deberían usar los métodos de evaluación: deja
    una matriz en la que *más es mejor* en todas las columnas.
    """
    return _PROCEDIMIENTOS[Procedimiento(procedimiento)](orientar(A, sentidos, politica))


# ---------------------------------------------------------------- auxiliares


def _como_matriz(A: np.ndarray, sentidos: list[Sentido] | None = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ErrorDeNormalizacion(f"Se esperaba una matriz 2D, llegó {A.ndim}D.")
    if A.size == 0:
        raise ErrorDeNormalizacion("La matriz de decisión está vacía.")
    if np.any(np.isnan(A)):
        raise ErrorDeNormalizacion(
            "La matriz tiene evaluaciones sin cargar (nan). Completala antes "
            "de normalizar."
        )
    if sentidos is not None and len(sentidos) != A.shape[1]:
        raise ErrorDeNormalizacion(
            f"La matriz tiene {A.shape[1]} criterios pero se recibieron "
            f"{len(sentidos)} sentidos."
        )
    return A


def _verificar_no_nulo(valores: np.ndarray, que: str) -> None:
    if np.any(np.isclose(valores, 0.0)):
        cols = np.where(np.isclose(valores, 0.0))[0].tolist()
        raise ErrorDeNormalizacion(
            f"En las columnas {cols}, {que} es cero: no se puede dividir."
        )
