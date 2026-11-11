"""Validación contra los trabajos de clase (planillas de la cátedra).

Cada caso de ``data/validacion`` reproduce un ejercicio resuelto en las
planillas de la cátedra; los valores esperados son los que calcularon los
grupos. Las planillas no redondean los intermedios, así que la tolerancia es
estricta (1e-5).

Los ejercicios de clase evalúan **todas** las alternativas, aunque alguna esté
dominada: por eso acá no se aplica el preanálisis. En la interfaz hay que
destildar "Evaluar solo el conjunto de elección" para obtener los mismos
números.

Corre con pytest (``pytest``) o directamente (``python tests/test_clase.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cargar  # noqa: E402
from core.io_casos import DIR_DATOS  # noqa: E402
from core.normalizacion import Procedimiento  # noqa: E402
from core.pesos import del_problema  # noqa: E402
from core.preanalisis import dominacion  # noqa: E402
from models import ContextoEvaluacion, Ideal, Metrica, obtener  # noqa: E402

VALIDACION = DIR_DATOS / "validacion"
TOL = 1e-5


def _topsis_de_clase(archivo: str, procedimiento: Procedimiento) -> np.ndarray:
    p = cargar(VALIDACION / archivo)
    ctx = ContextoEvaluacion(p, del_problema(p).pesos, procedimiento=procedimiento)
    topsis = obtener("topsis", metrica=Metrica.EUCLIDEA, ideal=Ideal.SEGUN_SENTIDO)
    return topsis.evaluar(ctx)


# ------------------------------------------------------------------ TOPSIS


def test_topsis_inversiones_como_en_clase():
    r = _topsis_de_clase("topsis_inversiones.json", Procedimiento.VECTORIAL)
    assert np.allclose(r.puntajes, [0.285583, 0.654391, 0.379250, 0.566097, 0.620750], atol=TOL)
    d = r.detalle["Distancias e índice de similaridad"]
    assert np.allclose(d["S+"], [0.178033, 0.0741922, 0.172883, 0.0952379, 0.105623], atol=TOL)
    assert np.allclose(d["S-"], [0.0711674, 0.140479, 0.105623, 0.124253, 0.172883], atol=TOL)
    assert [r.nombres[i] for i in r.orden] == ["B", "E", "D", "C", "A"]


def test_topsis_inversiones_a_esta_dominada():
    # Por eso en la interfaz hay que evaluar todas las alternativas para
    # reproducir la planilla: si no, A se descarta y cambia la normalización.
    p = cargar(VALIDACION / "topsis_inversiones.json")
    dom = dominacion(p.matriz(), p.sentidos)
    assert dom.dominadas == [0]
    assert dom.dominada_por[0] == [1]


def test_topsis_sucursales_como_en_clase():
    r = _topsis_de_clase("topsis_sucursales.json", Procedimiento.SUMA)
    esperado = [0.409072, 0.393065, 0.564176, 0.521183, 0.476924,
                0.449862, 0.505145, 0.547440, 0.418151, 0.430768]
    assert np.allclose(r.puntajes, esperado, atol=TOL)
    assert r.orden == [2, 7, 3, 6, 4, 5, 9, 8, 0, 1]


# --------------------------------------------------------------------- AHP


def test_ahp_julian_como_en_clase():
    p = cargar(VALIDACION / "ahp_julian.json")
    pesos = del_problema(p)
    assert np.allclose(pesos.pesos, [0.567873, 0.333937, 0.098190], atol=TOL)
    c = pesos.consistencia
    assert np.isclose(c.lambda_max, 3.02466, atol=TOL)
    assert np.isclose(c.ic, 0.0123288, atol=TOL)
    assert np.isclose(c.rc, 0.0212565, atol=TOL)
    r = obtener("ahp").evaluar(ContextoEvaluacion(p, pesos.pesos))
    assert np.allclose(r.puntajes, [0.574517, 0.425483], atol=TOL)


if __name__ == "__main__":
    fallos = 0
    pruebas = [
        (nombre, fn)
        for nombre, fn in sorted(globals().items())
        if nombre.startswith("test_") and callable(fn)
    ]
    for nombre, fn in pruebas:
        try:
            fn()
            print(f"  ok    {nombre}")
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            print(f"  FALLA {nombre}: {exc}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas pasaron.")
    sys.exit(1 if fallos else 0)
