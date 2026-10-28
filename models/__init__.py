"""Métodos de evaluación DMD.

Cada método cumple el contrato de ``models/base.py`` y se da de alta en el
registro al importarse, por eso se importan acá aunque no se usen
directamente: así la interfaz los encuentra con ``metodos_disponibles()``.
"""

from .base import (
    ContextoEvaluacion,
    ErrorDeEvaluacion,
    MetodoDMD,
    Resultado,
    metodos_disponibles,
    obtener,
    ordenar_por_puntaje,
    registrar,
    usa_politica,
    usa_procedimiento,
)
# El orden de importación es el orden del selector: el del capítulo.
from .ponderacion_lineal import PonderacionLineal  # isort: skip
from .ahp import AHP  # isort: skip
from .topsis import TOPSIS, Ideal, Metrica  # isort: skip

__all__ = [
    "AHP",
    "TOPSIS",
    "ContextoEvaluacion",
    "ErrorDeEvaluacion",
    "Ideal",
    "MetodoDMD",
    "Metrica",
    "PonderacionLineal",
    "Resultado",
    "metodos_disponibles",
    "obtener",
    "ordenar_por_puntaje",
    "registrar",
    "usa_politica",
    "usa_procedimiento",
]
