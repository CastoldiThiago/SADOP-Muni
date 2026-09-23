"""Modelo de dominio de un problema de Decisión Multicriterio Discreta (DMD).

Define el problema con independencia del método de evaluación que se use
después (Ponderación Lineal, AHP, TOPSIS, u otro). La matriz de decisión y
el vector de pesos son, según el Cap. 12, "toda la información necesaria,
en principio, para resolver un problema de DMD".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class Sentido(str, Enum):
    """Dirección de preferencia de un criterio."""

    MAXIMIZAR = "max"
    MINIMIZAR = "min"

    @property
    def etiqueta(self) -> str:
        return "Maximizar" if self is Sentido.MAXIMIZAR else "Minimizar"


class Escala(str, Enum):
    """Escala de medida del criterio (Cap. 12, sección 5).

    La escala condiciona qué operaciones son legítimas sobre las
    evaluaciones: sobre una escala ordinal no tiene sentido promediar.
    """

    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVALO = "intervalo"
    RAZON = "razon"


@dataclass
class Criterio:
    """Un criterio de evaluación (columna de la matriz de decisión)."""

    id: str
    nombre: str
    sentido: Sentido = Sentido.MAXIMIZAR
    unidad: str = ""
    escala: Escala = Escala.RAZON
    descripcion: str = ""
    # Nivel de aspiración para el preanálisis de satisfacción (sección 10).
    # None = el criterio no impone umbral.
    umbral: float | None = None

    def __post_init__(self) -> None:
        self.sentido = Sentido(self.sentido)
        self.escala = Escala(self.escala)

    @property
    def rotulo(self) -> str:
        return f"{self.nombre} [{self.unidad}]" if self.unidad else self.nombre


@dataclass
class Alternativa:
    """Una alternativa de decisión (fila de la matriz): una obra pública."""

    id: str
    nombre: str
    descripcion: str = ""


class ErrorDeProblema(ValueError):
    """El problema de decisión está mal formado."""


@dataclass
class ProblemaDecision:
    """Un problema de DMD completo: alternativas, criterios y evaluaciones.

    Las evaluaciones se guardan indexadas por ``(id_alternativa, id_criterio)``
    en lugar de como matriz densa para que agregar o quitar un criterio no
    invalide silenciosamente los datos ya cargados.
    """

    titulo: str = "Problema de decisión"
    descripcion: str = ""
    decisor: str = ""
    alternativas: list[Alternativa] = field(default_factory=list)
    criterios: list[Criterio] = field(default_factory=list)
    evaluaciones: dict[tuple[str, str], float] = field(default_factory=dict)

    # ---------------------------------------------------------------- acceso

    @property
    def ids_alternativas(self) -> list[str]:
        return [a.id for a in self.alternativas]

    @property
    def ids_criterios(self) -> list[str]:
        return [c.id for c in self.criterios]

    @property
    def sentidos(self) -> list[Sentido]:
        return [c.sentido for c in self.criterios]

    def alternativa(self, id_alt: str) -> Alternativa:
        for a in self.alternativas:
            if a.id == id_alt:
                return a
        raise KeyError(f"No existe la alternativa '{id_alt}'")

    def criterio(self, id_crit: str) -> Criterio:
        for c in self.criterios:
            if c.id == id_crit:
                return c
        raise KeyError(f"No existe el criterio '{id_crit}'")

    # ------------------------------------------------------------ mutadores

    def agregar_alternativa(self, alt: Alternativa) -> None:
        if alt.id in self.ids_alternativas:
            raise ErrorDeProblema(f"Alternativa duplicada: '{alt.id}'")
        self.alternativas.append(alt)

    def agregar_criterio(self, crit: Criterio) -> None:
        if crit.id in self.ids_criterios:
            raise ErrorDeProblema(f"Criterio duplicado: '{crit.id}'")
        self.criterios.append(crit)

    def quitar_alternativa(self, id_alt: str) -> None:
        self.alternativas = [a for a in self.alternativas if a.id != id_alt]
        self.evaluaciones = {
            k: v for k, v in self.evaluaciones.items() if k[0] != id_alt
        }

    def quitar_criterio(self, id_crit: str) -> None:
        self.criterios = [c for c in self.criterios if c.id != id_crit]
        self.evaluaciones = {
            k: v for k, v in self.evaluaciones.items() if k[1] != id_crit
        }

    def evaluar(self, id_alt: str, id_crit: str, valor: float) -> None:
        """Carga la evaluación a_ij de una alternativa en un criterio."""
        if id_alt not in self.ids_alternativas:
            raise KeyError(f"No existe la alternativa '{id_alt}'")
        if id_crit not in self.ids_criterios:
            raise KeyError(f"No existe el criterio '{id_crit}'")
        self.evaluaciones[(id_alt, id_crit)] = float(valor)

    # ------------------------------------------------------------- matrices

    def faltantes(self) -> list[tuple[str, str]]:
        """Celdas sin cargar de la matriz de decisión."""
        return [
            (a, c)
            for a in self.ids_alternativas
            for c in self.ids_criterios
            if (a, c) not in self.evaluaciones
        ]

    def matriz(self) -> np.ndarray:
        """Matriz de decisión A (m alternativas x n criterios), sin normalizar.

        Las celdas no cargadas quedan como ``nan``; usar :meth:`validar`
        antes de pasarla a un método de evaluación.
        """
        return np.array(
            [
                [self.evaluaciones.get((a, c), np.nan) for c in self.ids_criterios]
                for a in self.ids_alternativas
            ],
            dtype=float,
        )

    def tabla(self) -> pd.DataFrame:
        """Matriz de decisión como DataFrame rotulado, para mostrar o exportar."""
        return pd.DataFrame(
            self.matriz(),
            index=[a.nombre for a in self.alternativas],
            columns=[c.rotulo for c in self.criterios],
        )

    # ----------------------------------------------------------- validación

    def validar(self) -> list[str]:
        """Devuelve la lista de problemas encontrados (vacía si está todo bien).

        Se devuelven todos los problemas juntos en vez de lanzar en el primero
        para poder mostrarlos de una sola vez en la interfaz.
        """
        errores: list[str] = []

        if len(self.alternativas) < 2:
            errores.append("Se requieren al menos 2 alternativas para decidir.")
        if len(self.criterios) < 2:
            errores.append(
                "Se requieren al menos 2 criterios (si no, no es multicriterio)."
            )

        for dup in _duplicados(self.ids_alternativas):
            errores.append(f"Id de alternativa duplicado: '{dup}'.")
        for dup in _duplicados(self.ids_criterios):
            errores.append(f"Id de criterio duplicado: '{dup}'.")

        faltantes = self.faltantes()
        if faltantes:
            muestra = ", ".join(f"{a}/{c}" for a, c in faltantes[:5])
            resto = f" (y {len(faltantes) - 5} más)" if len(faltantes) > 5 else ""
            errores.append(f"Faltan evaluaciones: {muestra}{resto}.")

        # Un criterio en el que todas las alternativas valen lo mismo no aporta
        # información y además rompe la normalización por rango (max = min).
        if not faltantes and self.alternativas:
            A = self.matriz()
            for j, c in enumerate(self.criterios):
                col = A[:, j]
                if np.allclose(col, col[0]):
                    errores.append(
                        f"El criterio '{c.nombre}' tiene el mismo valor en todas "
                        "las alternativas: no discrimina y conviene quitarlo."
                    )
        return errores

    def es_valido(self) -> bool:
        return not self.validar()


def _duplicados(items: Iterable[str]) -> list[str]:
    vistos: set[str] = set()
    dups: list[str] = []
    for it in items:
        if it in vistos and it not in dups:
            dups.append(it)
        vistos.add(it)
    return dups
