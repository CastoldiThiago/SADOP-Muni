"""Asignación de pesos a los criterios (Cap. 12, secciones 11, 14 y 15).

Los pesos son independientes del método de evaluación: Ponderación Lineal y
TOPSIS reciben el mismo vector, venga de una asignación directa o de las
comparaciones pareadas de AHP. Por eso viven en ``core`` y no en ``models``.

AHP se implementa acá en su parte general (prioridades de cualquier matriz de
comparaciones pareadas y su consistencia) para poder aplicarlo tanto a los
criterios como, si hiciera falta, a las alternativas respecto de un criterio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .modelo import ModoPesos, ProblemaDecision

#: Índice aleatorio (IA) por orden de la matriz: tabla de Saaty que usa la
#: cátedra (presentación de AHP), hasta n = 8; 9 y 10 completan con los
#: valores de Saaty. Las Tablas 11 y 12 del capítulo traen otra serie (0,52;
#: 0,89; ...), pero el propio ejemplo del capítulo calcula RC con 0,90.
#: Para n <= 2 la matriz es siempre consistente.
INDICE_ALEATORIO = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

#: Umbral de la relación de consistencia por encima del cual conviene revisar
#: los juicios del decisor.
RC_MAXIMO = 0.10

#: Escala Fundamental de Saaty (Tabla 10). Los pares 2, 4, 6, 8 son valores
#: intermedios de compromiso.
ESCALA_SAATY = {
    1: "Igualmente importante",
    2: "Intermedio",
    3: "Moderadamente más importante",
    4: "Intermedio",
    5: "Notablemente más importante",
    6: "Intermedio",
    7: "Importancia muy fuerte o demostrada",
    8: "Intermedio",
    9: "Importancia absoluta",
}


class ErrorDePesos(ValueError):
    """Los pesos o las comparaciones pareadas están mal formados."""


class Estimacion(str, Enum):
    """Cómo estimar el vector de prioridades de una matriz pareada."""

    #: Promedio de las filas de la matriz normalizada por columnas. Es el
    #: "procedimiento de dos pasos" de la Etapa 3, con el que el capítulo
    #: resuelve el ejemplo.
    APROXIMADA = "aproximada"
    #: Autovector dominante exacto (Anexo 1).
    AUTOVECTOR = "autovector"

    @property
    def etiqueta(self) -> str:
        return (
            "Aproximada (promedio de filas)"
            if self is Estimacion.APROXIMADA
            else "Autovector dominante"
        )


# --------------------------------------------------------- asignación directa


def directos(valores: list[float] | np.ndarray) -> np.ndarray:
    """Normaliza pesos asignados directamente para que sumen 1."""
    w = np.asarray(valores, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ErrorDePesos("Se esperaba un vector de pesos no vacío.")
    if np.any(np.isnan(w)):
        raise ErrorDePesos("Hay pesos sin cargar.")
    if np.any(w < 0):
        raise ErrorDePesos("Los pesos no pueden ser negativos.")
    total = w.sum()
    if np.isclose(total, 0.0):
        raise ErrorDePesos("Los pesos suman cero.")
    return w / total


# ------------------------------------------------------------------------ AHP


def matriz_pareada(ids: list[str], juicios: dict[tuple[str, str], float]) -> np.ndarray:
    """Arma la matriz recíproca de comparaciones a partir de los juicios.

    Para cada par alcanza con un juicio ``(i, j)``; el ``(j, i)`` es su
    recíproco. Un par sin juicio se toma como igualmente importante (1), que
    es el valor neutro de la escala.
    """
    n = len(ids)
    M = np.ones((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            i, j = ids[a], ids[b]
            if (i, j) in juicios:
                valor = float(juicios[(i, j)])
            elif (j, i) in juicios:
                valor = 1.0 / float(juicios[(j, i)])
            else:
                valor = 1.0
            if valor <= 0:
                raise ErrorDePesos(
                    f"El juicio entre '{i}' y '{j}' debe ser positivo."
                )
            M[a, b], M[b, a] = valor, 1.0 / valor
    return M


def prioridades(
    M: np.ndarray, estimacion: Estimacion = Estimacion.APROXIMADA
) -> np.ndarray:
    """Vector de prioridades (pesos relativos) de una matriz pareada."""
    M = _verificar_reciproca(M)
    if Estimacion(estimacion) is Estimacion.APROXIMADA:
        return (M / M.sum(axis=0)).mean(axis=1)

    valores, vectores = np.linalg.eig(M)
    # Por Perron-Frobenius el autovalor de mayor parte real es real y su
    # autovector puede tomarse con todas las componentes positivas.
    k = int(np.argmax(valores.real))
    w = np.abs(vectores[:, k].real)
    return w / w.sum()


@dataclass
class Consistencia:
    """Resultado de la Etapa 4 de AHP para una matriz pareada."""

    n: int
    lambda_max: float
    #: Índice de consistencia, (lambda_max - n) / (n - 1).
    ic: float
    #: Índice aleatorio para matrices de orden n.
    ia: float
    #: Relación de consistencia IC / IA. ``None`` si n <= 2: no hay forma de
    #: ser inconsistente con un solo juicio.
    rc: float | None

    @property
    def aceptable(self) -> bool:
        return self.rc is None or self.rc <= RC_MAXIMO


def consistencia(M: np.ndarray, w: np.ndarray | None = None) -> Consistencia:
    """Calcula lambda_max, IC y RC como en la Etapa 4 del capítulo.

    ``lambda_max`` se estima promediando ``(Aw)_i / w_i``, igual que en el
    ejemplo. Si no se pasa ``w`` se usa la estimación aproximada.
    """
    M = _verificar_reciproca(M)
    n = M.shape[0]
    if n not in INDICE_ALEATORIO:
        raise ErrorDePesos(
            f"No hay índice aleatorio tabulado para {n} elementos (máximo 10). "
            "Saaty recomienda no comparar más de 9 elementos a la vez."
        )
    w = prioridades(M) if w is None else np.asarray(w, dtype=float)
    lambda_max = float(np.mean((M @ w) / w))
    if n <= 2:
        return Consistencia(n, lambda_max, 0.0, 0.0, None)
    ic = (lambda_max - n) / (n - 1)
    ia = INDICE_ALEATORIO[n]
    return Consistencia(n, lambda_max, ic, ia, ic / ia)


#: Valores admitidos por la escala de Saaty, como razones: 1/9 ... 1 ... 9.
VALORES_SAATY = np.array([1 / k for k in range(9, 1, -1)] + list(range(1, 10)), dtype=float)


@dataclass
class Sugerencia:
    """Un juicio a revisar para mejorar la consistencia de una matriz."""

    #: Posición del juicio en la matriz (fila i, columna j, con i < j).
    i: int
    j: int
    #: Juicio actual a_ij.
    actual: float
    #: Valor de la escala de Saaty que, cambiando solo este juicio, deja la
    #: matriz más consistente.
    sugerido: float
    #: RC que quedaría con ese cambio.
    rc_nuevo: float


def sugerencias(
    M: np.ndarray,
    estimacion: Estimacion = Estimacion.APROXIMADA,
    cantidad: int = 3,
) -> list[Sugerencia]:
    """Juicios cuyo cambio más mejora la consistencia.

    Para cada par se prueban los 17 valores de la escala de Saaty en lugar del
    juicio actual, dejando el resto igual, y se toma el que da el menor RC. Se
    devuelven los pares que más bajan el RC.

    No se usa la razón w_i / w_j de las prioridades actuales porque un juicio
    muy errado distorsiona esas mismas prioridades y la razón termina
    sugiriendo un valor intermedio que no lo corrige.

    Es una ayuda para revisar, no un reemplazo del juicio del decisor.
    """
    M = _verificar_reciproca(M)
    n = M.shape[0]
    if n <= 2:
        return []

    def rc_con(i: int, j: int, valor: float) -> float:
        M2 = M.copy()
        M2[i, j], M2[j, i] = valor, 1.0 / valor
        return consistencia(M2, prioridades(M2, estimacion)).rc

    candidatas: list[Sugerencia] = []
    for i in range(n):
        for j in range(i + 1, n):
            rcs = [rc_con(i, j, v) for v in VALORES_SAATY]
            k = int(np.argmin(rcs))
            if np.isclose(VALORES_SAATY[k], M[i, j]):
                continue
            candidatas.append(Sugerencia(i, j, float(M[i, j]), float(VALORES_SAATY[k]), rcs[k]))
    candidatas.sort(key=lambda s: s.rc_nuevo)
    return candidatas[:cantidad]


def _verificar_reciproca(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.size == 0:
        raise ErrorDePesos("La matriz de comparaciones debe ser cuadrada.")
    if np.any(M <= 0):
        raise ErrorDePesos("Todos los juicios deben ser positivos.")
    if not np.allclose(np.diag(M), 1.0):
        raise ErrorDePesos("La diagonal debe ser 1: cada elemento vale igual que sí mismo.")
    if not np.allclose(M * M.T, 1.0):
        raise ErrorDePesos("La matriz no es recíproca: a_ji debe ser 1 / a_ij.")
    return M


# ------------------------------------------------------------- del problema


@dataclass
class ResultadoPesos:
    """Pesos normalizados de un problema y, si vienen de AHP, su consistencia."""

    pesos: np.ndarray
    modo: ModoPesos
    #: Matriz de comparaciones usada (solo AHP).
    matriz: np.ndarray | None = None
    consistencia: Consistencia | None = None


def del_problema(
    problema: ProblemaDecision, estimacion: Estimacion = Estimacion.APROXIMADA
) -> ResultadoPesos:
    """Calcula los pesos según el modo elegido en el problema."""
    ids = problema.ids_criterios
    if not ids:
        raise ErrorDePesos("No hay criterios definidos.")

    if problema.modo_pesos is ModoPesos.DIRECTO:
        faltan = [c.nombre for c in problema.criterios if c.id not in problema.pesos_directos]
        if faltan:
            raise ErrorDePesos(f"Falta asignar peso a: {', '.join(faltan)}.")
        return ResultadoPesos(
            directos([problema.pesos_directos[i] for i in ids]), ModoPesos.DIRECTO
        )

    M = matriz_pareada(ids, problema.comparaciones)
    w = prioridades(M, estimacion)
    return ResultadoPesos(w, ModoPesos.AHP, M, consistencia(M, w))
