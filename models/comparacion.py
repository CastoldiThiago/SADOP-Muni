"""Comparación de métodos sobre el mismo problema y los mismos pesos.

Si métodos con supuestos distintos (compensación lineal, distancia al ideal,
comparaciones pareadas) coinciden en la ganadora, el resultado es más
defendible que si se apoya en uno solo. Si no coinciden, conviene saberlo y
explicar por qué antes de presentar el ranking.

El acuerdo entre dos rankings se mide con el coeficiente de Spearman: 1 si
ordenan igual, -1 si ordenan al revés.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.normalizacion import ErrorDeNormalizacion

from .base import ContextoEvaluacion, ErrorDeEvaluacion, MetodoDMD


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de rangos entre dos vectores de posiciones sin empates."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = a.size
    if n < 2:
        return 1.0
    return float(1 - 6 * np.sum((a - b) ** 2) / (n * (n**2 - 1)))


@dataclass
class Comparacion:
    #: Posición de cada alternativa (filas) según cada método (columnas).
    posiciones: pd.DataFrame
    #: Puntaje de cada alternativa según cada método. No son comparables
    #: entre métodos: cada uno tiene su propia escala.
    puntajes: pd.DataFrame
    #: Spearman entre cada par de métodos.
    acuerdo: pd.DataFrame
    #: Métodos que no pudieron evaluar el caso, con el motivo.
    errores: dict[str, str]

    @property
    def ganadoras(self) -> pd.Series:
        return self.posiciones.idxmin()

    def consenso(self) -> pd.DataFrame:
        """Alternativas ordenadas por posición media entre métodos."""
        return pd.DataFrame(
            {
                "Posición media": self.posiciones.mean(axis=1),
                "Veces primera": (self.posiciones == 1).sum(axis=1),
                "Mejor": self.posiciones.min(axis=1),
                "Peor": self.posiciones.max(axis=1),
            }
        ).sort_values(["Posición media", "Veces primera"], ascending=[True, False])


def comparar(ctx: ContextoEvaluacion, metodos: dict[str, MetodoDMD]) -> Comparacion:
    """Evalúa el mismo contexto con cada método."""
    posiciones: dict[str, dict[str, int]] = {}
    puntajes: dict[str, dict[str, float]] = {}
    errores: dict[str, str] = {}
    for etiqueta, metodo in metodos.items():
        try:
            res = metodo.evaluar(ctx)
        except (ErrorDeNormalizacion, ErrorDeEvaluacion) as exc:
            errores[etiqueta] = str(exc)
            continue
        posiciones[etiqueta] = {nombre: pos for pos, nombre, _ in res.ranking()}
        puntajes[etiqueta] = dict(zip(res.nombres, res.puntajes, strict=True))

    pos = pd.DataFrame(posiciones).reindex(ctx.nombres)
    cols = list(pos.columns)
    acuerdo = pd.DataFrame(
        [[spearman(pos[a], pos[b]) for b in cols] for a in cols], index=cols, columns=cols
    )
    return Comparacion(pos, pd.DataFrame(puntajes).reindex(ctx.nombres), acuerdo, errores)
