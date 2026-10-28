"""TOPSIS (Cap. 12, sección 16).

Ordena las alternativas por su índice de similaridad a la solución ideal:

    v_ij = w_j r_ij
    A+   = mejor valor de cada columna de V  (alternativa ideal, virtual)
    A-   = peor valor de cada columna de V   (alternativa anti-ideal, virtual)
    S_i+ = distancia de Minkowski de la fila i a A+
    S_i- = distancia de Minkowski de la fila i a A-
    C_i* = S_i- / (S_i+ + S_i-)

Hay dos formas de tratar los criterios a minimizar (ver :class:`Ideal`). La
métrica también es un parámetro: el capítulo admite p = 1 (ciudad), p = 2
(euclídea) y p = infinito (Chebyshev), y señala que con p = 1 TOPSIS ordena
igual que la Ponderación Lineal.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from core.modelo import Sentido
from core.normalizacion import Procedimiento

from .base import (
    ContextoEvaluacion,
    ErrorDeEvaluacion,
    Resultado,
    ordenar_por_puntaje,
    registrar,
)


class Metrica(str, Enum):
    """Métrica de Minkowski para medir las distancias a los ideales."""

    CIUDAD = "m1"
    EUCLIDEA = "m2"
    CHEBYSHEV = "minf"

    @property
    def etiqueta(self) -> str:
        return {
            Metrica.CIUDAD: "m1 · ciudad (p = 1)",
            Metrica.EUCLIDEA: "m2 · euclídea (p = 2)",
            Metrica.CHEBYSHEV: "m∞ · Chebyshev (p = ∞)",
        }[self]

    @property
    def normalizacion_asociada(self) -> Procedimiento:
        """Normalización que el capítulo empareja con cada métrica."""
        return {
            Metrica.CIUDAD: Procedimiento.SUMA,
            Metrica.EUCLIDEA: Procedimiento.VECTORIAL,
            Metrica.CHEBYSHEV: Procedimiento.MAXIMO,
        }[self]

    def distancia(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distancia de cada fila de ``X`` al punto ``y``."""
        d = np.abs(X - y)
        if self is Metrica.CIUDAD:
            return d.sum(axis=1)
        if self is Metrica.EUCLIDEA:
            return np.sqrt((d**2).sum(axis=1))
        return d.max(axis=1)


class Ideal(str, Enum):
    """Cómo se definen el ideal y el anti-ideal ante criterios a minimizar."""

    #: Presentación de la cátedra (Hwang y Yoon): se normaliza sin convertir
    #: y el ideal es el máximo en los criterios a maximizar y el mínimo en los
    #: de minimizar. No depende de la política para criterios a minimizar.
    SEGUN_SENTIDO = "sentido"
    #: Capítulo 12: primero se convierten todos los criterios a maximizar
    #: (con la política elegida) y el ideal es siempre el máximo.
    ORIENTADO = "orientado"

    @property
    def etiqueta(self) -> str:
        return (
            "Según el sentido de cada criterio (clase)"
            if self is Ideal.SEGUN_SENTIDO
            else "Convirtiendo todo a maximizar (Cap. 12)"
        )


@registrar
class TOPSIS:
    id = "topsis"
    nombre = "TOPSIS"

    def __init__(
        self,
        metrica: Metrica = Metrica.EUCLIDEA,
        ideal: Ideal = Ideal.SEGUN_SENTIDO,
    ) -> None:
        self.metrica = Metrica(metrica)
        self.ideal = Ideal(ideal)

    @property
    def usa_politica(self) -> bool:
        return self.ideal is Ideal.ORIENTADO

    def evaluar(self, ctx: ContextoEvaluacion) -> Resultado:
        if self.ideal is Ideal.ORIENTADO:
            R = ctx.matriz_normalizada()
            V = R * ctx.pesos
            ideal, anti = V.max(axis=0), V.min(axis=0)
        else:
            R = ctx.matriz_escalada()
            V = R * ctx.pesos
            maximizar = np.array([s is Sentido.MAXIMIZAR for s in ctx.problema.sentidos])
            ideal = np.where(maximizar, V.max(axis=0), V.min(axis=0))
            anti = np.where(maximizar, V.min(axis=0), V.max(axis=0))

        s_mas = self.metrica.distancia(V, ideal)
        s_menos = self.metrica.distancia(V, anti)
        total = s_mas + s_menos
        if np.any(np.isclose(total, 0.0)):
            raise ErrorDeEvaluacion(
                "Todas las alternativas evaluadas son idénticas: el ideal y el "
                "anti-ideal coinciden y TOPSIS no puede ordenarlas."
            )
        C = s_menos / total

        return Resultado(
            metodo=f"{self.nombre} ({self.metrica.etiqueta}"
            + (", Cap. 12)" if self.ideal is Ideal.ORIENTADO else ")"),
            puntajes=C,
            orden=ordenar_por_puntaje(ctx.filas, C),
            nombres=ctx.nombres,
            detalle={
                "Matriz normalizada (r_ij)": ctx.rotular(R),
                "Matriz ponderada (v_ij = w_j · r_ij)": ctx.rotular(V),
                "Soluciones ideales": pd.DataFrame(
                    [ideal, anti],
                    index=["A+ (ideal)", "A- (anti-ideal)"],
                    columns=[c.nombre for c in ctx.problema.criterios],
                ),
                "Distancias e índice de similaridad": ctx.rotular(
                    np.column_stack([s_mas, s_menos, C]),
                    ["S+", "S-", "C*"],
                ),
            },
        )
