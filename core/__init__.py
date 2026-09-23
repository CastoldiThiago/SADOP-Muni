"""Núcleo del SSD: todo lo que es independiente del método de evaluación."""

from .io_casos import CASO_EJEMPLO, a_json, cargar, guardar
from .modelo import (
    Alternativa,
    Criterio,
    ErrorDeProblema,
    Escala,
    ProblemaDecision,
    Sentido,
)
from .normalizacion import (
    ErrorDeNormalizacion,
    PoliticaMinimizar,
    Procedimiento,
    normalizar,
    orientar,
)
from .preanalisis import dominacion, informe, satisfaccion, sobrevivientes

__all__ = [
    "CASO_EJEMPLO",
    "Alternativa",
    "Criterio",
    "ErrorDeNormalizacion",
    "ErrorDeProblema",
    "Escala",
    "PoliticaMinimizar",
    "Procedimiento",
    "ProblemaDecision",
    "Sentido",
    "a_json",
    "cargar",
    "dominacion",
    "guardar",
    "informe",
    "normalizar",
    "orientar",
    "satisfaccion",
    "sobrevivientes",
]
