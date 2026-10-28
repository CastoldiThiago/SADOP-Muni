"""AHP — Proceso Analítico Jerárquico (Cap. 12, secciones 14 y 15).

Etapas, tal como las presenta la cátedra:

1. Árbol jerárquico: objetivo, criterios y alternativas.
2. Comparaciones de a pares entre criterios y entre alternativas respecto de
   cada criterio.
3. Prioridades de cada matriz (aproximada o por autovector).
4. Consistencia de cada matriz (RC <= 0,10).
5. Evaluación global por ponderación lineal: U(A_i) = sum_j w_j p_ij.

Los pesos de los criterios (etapas 2 a 4 sobre criterios) llegan en el
contexto, igual que para los demás métodos. Las prioridades de las
alternativas se obtienen, criterio por criterio, de una de dos formas:

* **Comparaciones pareadas**, si el problema tiene juicios entre alternativas
  para ese criterio. Es lo natural en criterios cualitativos (urgencia
  social, impacto ambiental).
* **Datos de la matriz** normalizados por la suma (procedimiento 3, el que el
  capítulo asocia a AHP), para los criterios con una medida objetiva (costo,
  plazo). Evita pedir juicios sobre algo que ya está medido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.normalizacion import orientar, por_suma
from core.pesos import (
    Consistencia,
    ErrorDePesos,
    Estimacion,
    consistencia,
    matriz_pareada,
    prioridades,
)

from .base import (
    ContextoEvaluacion,
    ErrorDeEvaluacion,
    Resultado,
    ordenar_por_puntaje,
    registrar,
)


@registrar
class AHP:
    id = "ahp"
    nombre = "AHP"
    #: Las prioridades por datos siempre se normalizan por la suma.
    usa_procedimiento = False

    def __init__(self, estimacion: Estimacion = Estimacion.APROXIMADA) -> None:
        self.estimacion = Estimacion(estimacion)

    def prioridades_alternativas(
        self, ctx: ContextoEvaluacion
    ) -> tuple[np.ndarray, dict[int, Consistencia]]:
        """Matriz de prioridades P (alternativas x criterios) y la consistencia
        de cada criterio evaluado por comparaciones pareadas."""
        p = ctx.problema
        ids = [p.alternativas[i].id for i in ctx.filas]
        pareados = [
            j for j, c in enumerate(p.criterios) if c.id in p.comparaciones_alternativas
        ]
        por_datos = [j for j in range(len(p.criterios)) if j not in pareados]

        P = np.empty((len(ids), len(p.criterios)))
        if por_datos:
            A = ctx.matriz_cruda()[:, por_datos]
            sentidos = [p.sentidos[j] for j in por_datos]
            P[:, por_datos] = por_suma(orientar(A, sentidos, ctx.politica))

        cons: dict[int, Consistencia] = {}
        for j in pareados:
            juicios = p.comparaciones_alternativas[p.criterios[j].id]
            try:
                M = matriz_pareada(ids, juicios)
                P[:, j] = prioridades(M, self.estimacion)
                cons[j] = consistencia(M, P[:, j])
            except ErrorDePesos as exc:
                raise ErrorDeEvaluacion(
                    f"Comparaciones entre alternativas para "
                    f"'{p.criterios[j].nombre}': {exc}"
                ) from exc
        return P, cons

    def evaluar(self, ctx: ContextoEvaluacion) -> Resultado:
        P, cons = self.prioridades_alternativas(ctx)
        U = P @ ctx.pesos
        criterios = ctx.problema.criterios

        detalle: dict[str, object] = {
            "Matriz de prioridades (p_ij)": ctx.rotular(P),
            "Contribución de cada criterio (w_j · p_ij)": ctx.rotular(P * ctx.pesos),
        }
        if cons:
            detalle["Consistencia de los juicios entre alternativas"] = pd.DataFrame(
                {
                    "λ máx": [c.lambda_max for c in cons.values()],
                    "IC": [c.ic for c in cons.values()],
                    "IA": [c.ia for c in cons.values()],
                    "RC": [np.nan if c.rc is None else c.rc for c in cons.values()],
                },
                index=[criterios[j].nombre for j in cons],
            )
        return Resultado(
            metodo=self.nombre,
            puntajes=U,
            orden=ordenar_por_puntaje(ctx.filas, U),
            nombres=ctx.nombres,
            detalle=detalle,
        )
