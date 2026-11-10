"""Pruebas de la interfaz: caso en blanco y carga de un caso desde cero.

Usan el ``AppTest`` de Streamlit, que ejecuta ``app.py`` sin navegador.
Corre con pytest (``pytest``) o directamente (``python tests/test_app.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from streamlit.testing.v1 import AppTest  # noqa: E402

from core.modelo import Escala, ModoPesos, ProblemaDecision, Sentido  # noqa: E402

APP = str(RAIZ / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _caso_nuevo(at: AppTest) -> AppTest:
    """Abre el diálogo de caso nuevo y confirma."""
    at.button(key="abrir_nuevo_caso").click().run()
    at.button(key="nuevo_caso").click().run()
    assert not at.exception
    return at


def test_cancelar_el_caso_nuevo_conserva_el_actual():
    at = _app()
    at.button(key="abrir_nuevo_caso").click().run()
    # El borrado pide confirmación: abrir el diálogo no toca el caso.
    assert len(at.session_state.problema.alternativas) == 6
    at.button(key="cancelar_nuevo_caso").click().run()
    assert not at.exception
    assert len(at.session_state.problema.alternativas) == 6


def test_nuevo_caso_deja_el_problema_vacio():
    at = _caso_nuevo(_app())
    p = at.session_state.problema
    assert p.alternativas == [] and p.criterios == []
    assert p.evaluaciones == {}
    assert p.pesos_directos == {} and p.comparaciones == {}
    assert p.comparaciones_alternativas == {}
    assert any("Caso en blanco" in i.value for i in at.info)


def test_nuevo_caso_vuelve_al_paso_1():
    at = _app()
    at.radio(key="paso").set_value(at.radio(key="paso").options[4]).run()
    _caso_nuevo(at)
    assert at.radio(key="paso").value.startswith("1")


def test_nuevo_caso_olvida_los_controles_del_caso_anterior():
    at = _app()
    # Un juicio AHP de un criterio que en el caso nuevo ya no existe.
    at.session_state["ahp_costo_plazo"] = -3
    at.run()
    _caso_nuevo(at)
    assert "ahp_costo_plazo" not in at.session_state


def test_todos_los_pasos_funcionan_con_el_caso_vacio():
    at = _caso_nuevo(_app())
    for paso in at.radio(key="paso").options:
        at.radio(key="paso").set_value(paso).run()
        assert not at.exception, paso


def test_restaurar_el_ejemplo_despues_de_un_caso_nuevo():
    at = _caso_nuevo(_app())
    at.button[[b.label for b in at.button].index("Restaurar caso de ejemplo")].click().run()
    assert not at.exception
    assert len(at.session_state.problema.alternativas) == 6


def test_modo_de_pesos_vuelve_a_directo_con_un_clic():
    at = _app()
    at.radio(key="paso").set_value(at.radio(key="paso").options[3]).run()

    def modo():
        return next(r for r in at.radio if r.label == "Cómo asignar los pesos")

    modo().set_value(ModoPesos.AHP).run()
    assert at.session_state.problema.modo_pesos is ModoPesos.AHP
    modo().set_value(ModoPesos.DIRECTO).run()
    assert not at.exception
    assert at.session_state.problema.modo_pesos is ModoPesos.DIRECTO


def test_calculos_con_cache_coinciden_con_los_directos():
    import app
    from core import a_json, cargar, sobrevivientes
    from core.pesos import del_problema
    from models import ContextoEvaluacion, obtener
    from models.sensibilidad import barrido_peso, escenarios

    p = cargar()
    cfg = (
        ("metodo", "ponderacion_lineal"), ("metrica", "m2"), ("ideal", "sentido"),
        ("procedimiento", "maximo"), ("politica", "complemento"),
        ("estimacion", "aproximada"), ("solo_eleccion", True),
    )
    ctx = ContextoEvaluacion(p, del_problema(p).pesos, indices=sobrevivientes(p))
    directos = [barrido_peso(obtener("ponderacion_lineal"), ctx, j) for j in range(len(p.criterios))]
    cacheados = app.calcular_barridos(a_json(p), cfg)
    assert [b.cambios for b in cacheados] == [b.cambios for b in directos]
    tabla = app.calcular_escenarios(a_json(p), cfg)
    assert tabla["Ganadora"].tolist() == escenarios(ctx, app._metodos(dict(cfg)))["Ganadora"].tolist()


def test_sensibilidad_cambia_de_criterio_sin_errores():
    at = _app()
    at.radio(key="paso").set_value(at.radio(key="paso").options[6]).run()
    assert not at.exception
    detalle = next(s for s in at.selectbox if s.label == "Ver el detalle de un criterio")
    detalle.set_value(3).run()
    assert not at.exception
    # El cambio de ganadora de la urgencia social aparece en el detalle.
    assert any("62.2%" in m.value for m in at.markdown)


def test_aplicar_arma_un_caso_escribiendo_solo_los_nombres():
    import app

    p = ProblemaDecision()
    alternativas = pd.DataFrame(
        {
            "id": pd.Series([None, "esc", None], dtype="string"),
            "nombre": pd.Series(["Red cloacal", "Escuela", None], dtype="string"),
            "descripcion": pd.Series([None, "Refacción", None], dtype="string"),
        }
    )
    criterios = pd.DataFrame(
        {
            "id": pd.Series([None, None], dtype="string"),
            "nombre": pd.Series(["Costo", "Urgencia social"], dtype="string"),
            # Una fila nueva puede llegar sin sentido ni escala elegidos.
            "sentido": pd.Series(["min", None], dtype="string"),
            "unidad": pd.Series([None, "1-5"], dtype="string"),
            "escala": pd.Series([None, "ordinal"], dtype="string"),
            "umbral_min": pd.Series([None, 3], dtype="float"),
            "umbral_max": pd.Series([400, None], dtype="float"),
            "descripcion": pd.Series([None, None], dtype="string"),
        }
    )
    app._aplicar(p, alternativas, criterios)

    # La fila vacía se ignora y las que no tienen id lo reciben del nombre.
    assert p.ids_alternativas == ["red_cloacal", "esc"]
    assert p.alternativas[1].descripcion == "Refacción"
    assert p.ids_criterios == ["costo", "urgencia_social"]
    costo, urgencia = p.criterios
    assert costo.sentido is Sentido.MINIMIZAR and costo.escala is Escala.RAZON
    assert urgencia.sentido is Sentido.MAXIMIZAR and urgencia.escala is Escala.ORDINAL
    assert (costo.umbral_min, costo.umbral_max) == (None, 400)
    assert (urgencia.umbral_min, urgencia.umbral_max) == (3, None)

    # Con la matriz y los pesos cargados, el caso queda listo para evaluar.
    for a, valores in zip(p.ids_alternativas, [(310, 5), (95, 4)], strict=True):
        for c, v in zip(p.ids_criterios, valores, strict=True):
            p.evaluar(a, c, v)
    assert p.validar() == []


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
            print(f"  FALLA {nombre}: {exc!r}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas pasaron.")
    sys.exit(1 if fallos else 0)
