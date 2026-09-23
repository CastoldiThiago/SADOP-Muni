"""Métodos de evaluación DMD.

Todavía vacío a propósito: la elección del método (Ponderación Lineal, AHP,
TOPSIS, u otro) es una decisión pendiente del trabajo. Acá solo vive el
contrato que esos métodos tendrán que cumplir.

Cuando se implemente el primero, importarlo en este archivo para que quede
registrado al importar el paquete.
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
)

__all__ = [
    "ContextoEvaluacion",
    "ErrorDeEvaluacion",
    "MetodoDMD",
    "Resultado",
    "metodos_disponibles",
    "obtener",
    "ordenar_por_puntaje",
    "registrar",
]
