"""Contrato común de los métodos de evaluación DMD.

Este módulo **no implementa ningún método**: define la interfaz que tendrán
que cumplir Ponderación Lineal, AHP, TOPSIS o el que se elija más adelante,
para que la interfaz y el preanálisis no dependan de esa decisión.

Para agregar un método después alcanza con:

.. code-block:: python

    from models.base import ContextoEvaluacion, Resultado, registrar

    @registrar
    class PonderacionLineal:
        id = "saw"
        nombre = "Ponderación Lineal"
        requiere_normalizacion = True

        def evaluar(self, ctx: ContextoEvaluacion) -> Resultado:
            ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from core.modelo import ProblemaDecision
from core.normalizacion import PoliticaMinimizar, Procedimiento, normalizar


class ErrorDeEvaluacion(ValueError):
    """El método no puede evaluar el problema tal como está planteado."""


@dataclass
class ContextoEvaluacion:
    """Todo lo que un método necesita para producir un ranking.

    Separar el contexto del problema permite evaluar un **subconjunto** de
    alternativas (las que sobrevivieron al preanálisis) sin tener que mutar
    el problema original.
    """

    problema: ProblemaDecision
    #: Pesos de los criterios, en el orden de ``problema.criterios``.
    pesos: np.ndarray
    #: Índices de las alternativas a evaluar. ``None`` = todas.
    indices: list[int] | None = None
    procedimiento: Procedimiento = Procedimiento.MAXIMO
    politica: PoliticaMinimizar = PoliticaMinimizar.COMPLEMENTO

    def __post_init__(self) -> None:
        self.pesos = np.asarray(self.pesos, dtype=float)
        n = len(self.problema.criterios)
        if self.pesos.shape != (n,):
            raise ErrorDeEvaluacion(
                f"Se esperaban {n} pesos (uno por criterio), llegaron "
                f"{self.pesos.size}."
            )
        if np.any(self.pesos < 0):
            raise ErrorDeEvaluacion("Los pesos no pueden ser negativos.")
        total = self.pesos.sum()
        if np.isclose(total, 0.0):
            raise ErrorDeEvaluacion("Los pesos suman cero.")
        # TOPSIS y Ponderación Lineal asumen pesos normalizados; normalizarlos
        # acá evita que cada método tenga que acordarse de hacerlo.
        self.pesos = self.pesos / total

    @property
    def filas(self) -> list[int]:
        return (
            list(range(len(self.problema.alternativas)))
            if self.indices is None
            else list(self.indices)
        )

    @property
    def nombres(self) -> list[str]:
        return [self.problema.alternativas[i].nombre for i in self.filas]

    def matriz_cruda(self) -> np.ndarray:
        """Matriz de decisión de las alternativas evaluadas, sin transformar."""
        return self.problema.matriz()[self.filas, :]

    def matriz_normalizada(self) -> np.ndarray:
        """Matriz orientada a maximizar y normalizada según el contexto.

        La normalización se hace **sobre el subconjunto evaluado**: los máximos
        y mínimos de referencia son los de las alternativas que siguen en
        juego, no los de las ya descartadas.
        """
        return normalizar(
            self.matriz_cruda(),
            self.problema.sentidos,
            self.procedimiento,
            self.politica,
        )


@dataclass
class Resultado:
    """Salida de un método: puntajes y orden de preferencia."""

    metodo: str
    #: Puntaje global de cada alternativa evaluada, en el orden de ``filas``.
    puntajes: np.ndarray
    #: Índices de fila (del problema original) ordenados de mejor a peor.
    orden: list[int]
    #: Nombres de las alternativas, en el orden de ``puntajes``.
    nombres: list[str] = field(default_factory=list)
    #: Matrices intermedias para mostrar el paso a paso en la defensa.
    detalle: dict[str, object] = field(default_factory=dict)

    @property
    def ganadora(self) -> int:
        return self.orden[0]

    def ranking(self) -> list[tuple[int, str, float]]:
        """``(posición, nombre, puntaje)`` de mejor a peor."""
        pos = {fila: k for k, fila in enumerate(self.orden)}
        pares = sorted(
            zip(self.nombres, self.puntajes, strict=False),
            key=lambda p: p[1],
            reverse=True,
        )
        return [(k + 1, nombre, float(p)) for k, (nombre, p) in enumerate(pares)]


@runtime_checkable
class MetodoDMD(Protocol):
    """Interfaz que debe cumplir todo método de evaluación."""

    id: str
    nombre: str

    def evaluar(self, ctx: ContextoEvaluacion) -> Resultado: ...


_REGISTRO: dict[str, type] = {}


def registrar(cls: type) -> type:
    """Decorador que da de alta un método en el registro."""
    if not hasattr(cls, "id") or not hasattr(cls, "nombre"):
        raise TypeError(f"{cls.__name__} debe definir los atributos 'id' y 'nombre'.")
    if cls.id in _REGISTRO:
        raise ValueError(f"Ya hay un método registrado con id '{cls.id}'.")
    _REGISTRO[cls.id] = cls
    return cls


def metodos_disponibles() -> dict[str, type]:
    """Métodos registrados, por id. Vacío hasta que se implemente el primero."""
    return dict(_REGISTRO)


def obtener(id_metodo: str) -> MetodoDMD:
    """Instancia el método registrado con ese id."""
    if id_metodo not in _REGISTRO:
        disponibles = ", ".join(sorted(_REGISTRO)) or "ninguno todavía"
        raise KeyError(
            f"No hay un método registrado con id '{id_metodo}'. "
            f"Disponibles: {disponibles}."
        )
    return _REGISTRO[id_metodo]()


def ordenar_por_puntaje(filas: list[int], puntajes: np.ndarray) -> list[int]:
    """Índices de fila ordenados de mayor a menor puntaje.

    Auxiliar para que todos los métodos resuelvan los empates igual: a igual
    puntaje, se conserva el orden de carga de las alternativas.
    """
    return [filas[k] for k in np.argsort(-np.asarray(puntajes), kind="stable")]
