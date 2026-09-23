"""Pruebas de la base independiente del método.

Corre con pytest (``pytest``) o directamente (``python tests/test_base.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cargar  # noqa: E402
from core.modelo import Alternativa, Criterio, ProblemaDecision, Sentido  # noqa: E402
from core.normalizacion import (  # noqa: E402
    PoliticaMinimizar,
    Procedimiento,
    normalizar,
    orientar,
    por_maximo,
    por_rango,
    por_suma,
    vectorial,
)
from core.preanalisis import dominacion, satisfaccion, sobrevivientes  # noqa: E402

# Matriz chica de referencia: 3 alternativas x 2 criterios.
A = np.array([[100.0, 4.0], [200.0, 2.0], [400.0, 8.0]])
MAXIMIZAR = [Sentido.MAXIMIZAR, Sentido.MAXIMIZAR]


# --------------------------------------------------------- normalización


def test_por_maximo_deja_el_mejor_en_uno():
    R = por_maximo(A)
    assert np.allclose(R[:, 0], [0.25, 0.5, 1.0])
    assert np.isclose(R.max(axis=0).min(), 1.0)


def test_por_suma_da_columnas_que_suman_uno():
    R = por_suma(A)
    assert np.allclose(R.sum(axis=0), [1.0, 1.0])


def test_por_rango_deja_peor_en_cero_y_mejor_en_uno():
    R = por_rango(A)
    assert np.allclose(R.min(axis=0), [0.0, 0.0])
    assert np.allclose(R.max(axis=0), [1.0, 1.0])


def test_vectorial_da_columnas_de_norma_uno():
    R = vectorial(A)
    assert np.allclose(np.sqrt((R**2).sum(axis=0)), [1.0, 1.0])


def test_por_maximo_respeta_la_proporcionalidad():
    # 400 es 4 veces 100, y debe seguir siéndolo tras normalizar (Tabla 5).
    R = por_maximo(A)
    assert np.isclose(R[2, 0] / R[0, 0], A[2, 0] / A[0, 0])


def test_por_rango_no_respeta_la_proporcionalidad():
    # Contraste explícito con el caso anterior: el rango no conserva razones.
    # 400 es el doble de 200, pero normalizado por rango pasa a ser el triple.
    R = por_rango(A)
    assert np.isclose(A[2, 0] / A[1, 0], 2.0)
    assert np.isclose(R[2, 0] / R[1, 0], 3.0)


# ------------------------------------------------- criterios a minimizar


def test_orientar_no_toca_los_criterios_a_maximizar():
    assert np.allclose(orientar(A, MAXIMIZAR), A)


def test_orientar_por_complemento_invierte_el_orden():
    sentidos = [Sentido.MINIMIZAR, Sentido.MAXIMIZAR]
    R = orientar(A, sentidos, PoliticaMinimizar.COMPLEMENTO)
    # El costo más bajo (100) pasa a ser el valor más alto.
    assert R[0, 0] > R[1, 0] > R[2, 0]
    assert np.allclose(R[:, 0], [300.0, 200.0, 0.0])
    assert np.allclose(R[:, 1], A[:, 1])


def test_politicas_de_minimizacion_pueden_dar_rankings_distintos():
    # Justifica que la política sea una decisión explícita del analista.
    sentidos = [Sentido.MINIMIZAR, Sentido.MAXIMIZAR]
    comp = normalizar(A, sentidos, Procedimiento.MAXIMO, PoliticaMinimizar.COMPLEMENTO)
    inv = normalizar(A, sentidos, Procedimiento.MAXIMO, PoliticaMinimizar.INVERSION)
    assert not np.allclose(comp[:, 0], inv[:, 0])


def test_inversion_falla_con_ceros():
    sentidos = [Sentido.MINIMIZAR]
    try:
        orientar(np.array([[0.0], [5.0]]), sentidos, PoliticaMinimizar.INVERSION)
    except ValueError:
        return
    raise AssertionError("Se esperaba un error al invertir una columna con cero.")


def test_normalizar_deja_mas_es_mejor_en_todas_las_columnas():
    sentidos = [Sentido.MINIMIZAR, Sentido.MAXIMIZAR]
    R = normalizar(A, sentidos)
    mejor_costo = int(np.argmax(R[:, 0]))
    assert mejor_costo == int(np.argmin(A[:, 0]))


# ------------------------------------------------------------ preanálisis


def test_dominacion_detecta_la_alternativa_dominada():
    # b es peor que a en todo; a y c no se dominan entre sí.
    M = np.array([[10.0, 10.0], [5.0, 5.0], [12.0, 3.0]])
    res = dominacion(M, MAXIMIZAR)
    assert res.dominadas == [1]
    assert res.dominada_por[1] == [0]
    assert res.eficientes == [0, 2]


def test_dominacion_considera_el_sentido_de_cada_criterio():
    # Con costo a minimizar, la fila de costo bajo deja de estar dominada.
    M = np.array([[10.0, 10.0], [5.0, 5.0]])
    assert dominacion(M, [Sentido.MINIMIZAR, Sentido.MAXIMIZAR]).dominadas == []


def test_alternativas_identicas_no_se_dominan():
    M = np.array([[7.0, 7.0], [7.0, 7.0]])
    assert dominacion(M, MAXIMIZAR).dominadas == []


def test_satisfaccion_aplica_umbrales_segun_el_sentido():
    M = np.array([[100.0, 5.0], [500.0, 5.0], [100.0, 1.0]])
    sentidos = [Sentido.MINIMIZAR, Sentido.MAXIMIZAR]
    res = satisfaccion(M, sentidos, [400.0, 3.0])
    assert res.satisfacen == [0]
    assert res.incumple[1] == [0]  # supera el tope de costo
    assert res.incumple[2] == [1]  # no llega al mínimo del segundo criterio


def test_satisfaccion_ignora_criterios_sin_umbral():
    M = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert satisfaccion(M, MAXIMIZAR, [None, None]).descartadas == []


# ----------------------------------------------------------- modelo y caso


def test_validar_detecta_evaluaciones_faltantes():
    p = ProblemaDecision()
    p.agregar_alternativa(Alternativa("a1", "Obra 1"))
    p.agregar_alternativa(Alternativa("a2", "Obra 2"))
    p.agregar_criterio(Criterio("c1", "Costo", Sentido.MINIMIZAR))
    p.agregar_criterio(Criterio("c2", "Alcance"))
    p.evaluar("a1", "c1", 10)
    errores = p.validar()
    assert any("Faltan evaluaciones" in e for e in errores)


def test_validar_detecta_criterio_que_no_discrimina():
    p = ProblemaDecision()
    for i in (1, 2):
        p.agregar_alternativa(Alternativa(f"a{i}", f"Obra {i}"))
    p.agregar_criterio(Criterio("c1", "Costo", Sentido.MINIMIZAR))
    p.agregar_criterio(Criterio("c2", "Constante"))
    for i in (1, 2):
        p.evaluar(f"a{i}", "c1", i * 10)
        p.evaluar(f"a{i}", "c2", 5)
    assert any("no discrimina" in e for e in p.validar())


def test_quitar_criterio_limpia_sus_evaluaciones():
    p = ProblemaDecision()
    p.agregar_alternativa(Alternativa("a1", "Obra 1"))
    p.agregar_criterio(Criterio("c1", "Costo"))
    p.evaluar("a1", "c1", 10)
    p.quitar_criterio("c1")
    assert p.evaluaciones == {}


def test_caso_ejemplo_es_valido():
    p = cargar()
    assert p.validar() == []
    assert len(p.alternativas) == 6
    assert len(p.criterios) == 6


def test_caso_ejemplo_tiene_una_alternativa_dominada():
    # La plaza está dominada por el recambio de luminarias: sirve para mostrar
    # el preanálisis funcionando sobre el caso por defecto.
    p = cargar()
    res = dominacion(p.matriz(), p.sentidos)
    dominadas = [p.alternativas[i].id for i in res.dominadas]
    assert dominadas == ["plaza"]


def test_caso_ejemplo_filtra_por_umbrales_y_dominacion():
    p = cargar()
    ids = [p.alternativas[i].id for i in sobrevivientes(p)]
    # 'plaza' cae por dominación y por urgencia; 'hospital' por tope de costo.
    assert ids == ["pavimentacion", "escuela", "cloacas", "luminarias"]


def test_ida_y_vuelta_json_conserva_el_problema(tmp_path=None):
    import tempfile

    from core import guardar

    p = cargar()
    destino = Path(tmp_path or tempfile.mkdtemp()) / "caso.json"
    guardar(p, destino)
    q = cargar(destino)
    assert q.titulo == p.titulo
    assert q.ids_criterios == p.ids_criterios
    assert np.allclose(q.matriz(), p.matriz())
    assert [c.umbral for c in q.criterios] == [c.umbral for c in p.criterios]


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
