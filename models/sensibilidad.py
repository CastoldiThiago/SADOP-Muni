"""Análisis de sensibilidad del ranking.

Responde las dos preguntas que más pesan al defender un resultado de DMD:

1. **¿Qué tan firme es la ganadora frente a los pesos?** Se barre el peso de
   un criterio de 0 a 1, repartiendo el resto entre los demás en la misma
   proporción que tenían, y se ubican los puntos donde cambia la ganadora.
2. **¿Qué tan firme es frente a las decisiones del analista?** Se repite la
   evaluación con todas las combinaciones de método, normalización y política
   para criterios a minimizar. El capítulo advierte que el resultado "puede
   verse considerablemente afectado" por estas elecciones; este análisis lo
   muestra con el caso concreto en lugar de afirmarlo.

Funciona con cualquier método que cumpla :class:`MetodoDMD`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from core.normalizacion import ErrorDeNormalizacion, PoliticaMinimizar, Procedimiento

from .base import (
    ContextoEvaluacion,
    ErrorDeEvaluacion,
    MetodoDMD,
    usa_politica,
    usa_procedimiento,
)

# ---------------------------------------------------------- barrido de peso


def pesos_con(w: np.ndarray, j: int, t: float) -> np.ndarray:
    """Pesos con ``w_j = t`` y los demás reescalados para que sumen ``1 - t``.

    Los demás criterios conservan sus proporciones relativas. Si todo el peso
    estaba en ``j``, el resto se reparte en partes iguales.
    """
    w = np.asarray(w, dtype=float) / np.sum(w)
    nuevos = np.empty_like(w)
    resto = 1.0 - w[j]
    otros = [k for k in range(w.size) if k != j]
    if np.isclose(resto, 0.0):
        nuevos[otros] = (1.0 - t) / len(otros)
    else:
        nuevos[otros] = w[otros] * (1.0 - t) / resto
    nuevos[j] = t
    return nuevos


@dataclass
class Barrido:
    """Puntajes de las alternativas a medida que varía el peso de un criterio."""

    criterio: str
    peso_actual: float
    #: Una fila por valor del peso (índice), una columna por alternativa.
    puntajes: pd.DataFrame
    #: ``(peso, ganadora antes, ganadora después)`` en cada cambio de ganadora.
    cambios: list[tuple[float, str, str]]

    @property
    def ganadora_actual(self) -> str:
        return _ganadora_en(self, self.peso_actual)

    def intervalo_estable(self) -> tuple[float, float]:
        """Rango de pesos del criterio en el que la ganadora actual no cambia."""
        desde, hasta = 0.0, 1.0
        for peso, _, _ in self.cambios:
            if peso <= self.peso_actual:
                desde = peso
            elif peso < hasta:
                hasta = peso
        return desde, hasta


def _ganadora_en(barrido: Barrido, t: float) -> str:
    ganadora = str(barrido.puntajes.iloc[0].idxmax())
    for peso, _, despues in barrido.cambios:
        if peso <= t:
            ganadora = despues
    return ganadora


def barrido_peso(
    metodo: MetodoDMD,
    ctx: ContextoEvaluacion,
    j: int,
    puntos: int = 101,
    tolerancia: float = 1e-4,
) -> Barrido:
    """Barre el peso del criterio ``j`` de 0 a 1.

    Los cambios de ganadora se ubican primero en la grilla y después se
    refinan por bisección hasta ``tolerancia``, así que no dependen de la
    resolución con la que se grafica.
    """

    def evaluar(t: float) -> np.ndarray:
        return metodo.evaluar(replace(ctx, pesos=pesos_con(ctx.pesos, j, t))).puntajes

    def sigue_primera(puntajes: np.ndarray, g: int) -> bool:
        # Un empate no destrona a la ganadora vigente: si no, un empate en un
        # extremo (por ejemplo, dos obras igual de buenas en el único criterio
        # que pesa) aparecería como un cambio que en realidad no ocurre.
        return bool(np.isclose(puntajes[g], puntajes.max()))

    grilla = np.linspace(0.0, 1.0, puntos)
    filas = np.array([evaluar(t) for t in grilla])
    nombres = ctx.nombres
    puntajes = pd.DataFrame(filas, index=pd.Index(grilla, name="peso"), columns=nombres)

    cambios: list[tuple[float, str, str]] = []
    primera = int(np.argmax(filas[0]))
    for k in range(1, puntos):
        if sigue_primera(filas[k], primera):
            continue
        nueva = int(np.argmax(filas[k]))
        lo, hi = grilla[k - 1], grilla[k]
        while hi - lo > tolerancia:
            medio = (lo + hi) / 2
            if sigue_primera(evaluar(medio), primera):
                lo = medio
            else:
                hi = medio
        cambios.append((hi, nombres[primera], nombres[nueva]))
        primera = nueva

    return Barrido(
        criterio=ctx.problema.criterios[j].nombre,
        peso_actual=float(ctx.pesos[j]),
        puntajes=puntajes,
        cambios=cambios,
    )


# ---------------------------------------------------------------- escenarios


def escenarios(ctx: ContextoEvaluacion, metodos: dict[str, MetodoDMD]) -> pd.DataFrame:
    """Evalúa todas las combinaciones de método, normalización y política.

    Devuelve una fila por escenario con la ganadora y la posición de cada
    alternativa. Las combinaciones que el caso no admite (por ejemplo, invertir
    una columna con ceros) quedan registradas con su motivo en vez de cortar
    el análisis. Las opciones que un método no usa no se varían, para no
    contar varias veces el mismo escenario.
    """
    filas = []
    for etiqueta, metodo in metodos.items():
        procs = list(Procedimiento) if usa_procedimiento(metodo) else [ctx.procedimiento]
        pols = list(PoliticaMinimizar) if usa_politica(metodo) else [ctx.politica]
        for proc in procs:
            for pol in pols:
                fila: dict[str, object] = {
                    "Método": etiqueta,
                    "Normalización": proc.etiqueta if usa_procedimiento(metodo) else "no aplica",
                    "Criterios a minimizar": pol.etiqueta if usa_politica(metodo) else "no aplica",
                }
                try:
                    res = metodo.evaluar(replace(ctx, procedimiento=proc, politica=pol))
                except (ErrorDeNormalizacion, ErrorDeEvaluacion) as exc:
                    fila["Ganadora"] = None
                    fila["Motivo"] = str(exc)
                else:
                    fila["Ganadora"] = ctx.problema.alternativas[res.ganadora].nombre
                    for pos, nombre, _ in res.ranking():
                        fila[nombre] = pos
                filas.append(fila)
    return pd.DataFrame(filas)


def resumen_escenarios(tabla: pd.DataFrame, nombres: list[str]) -> pd.DataFrame:
    """Por alternativa: cuántas veces gana y qué posiciones ocupa."""
    validos = tabla.dropna(subset=["Ganadora"])
    posiciones = validos[nombres]
    resumen = pd.DataFrame(
        {
            "Veces primera": [int((validos["Ganadora"] == n).sum()) for n in nombres],
            "Posición media": posiciones.mean(),
            "Mejor posición": posiciones.min().astype(int),
            "Peor posición": posiciones.max().astype(int),
        },
        index=nombres,
    )
    return resumen.sort_values(["Veces primera", "Posición media"], ascending=[False, True])
