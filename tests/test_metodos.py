"""Validación de pesos, métodos y sensibilidad contra el Cap. 12.

Los ejemplos numéricos del capítulo se resuelven con valores intermedios
redondeados a tres o cuatro decimales, así que la comparación se hace con la
tolerancia que corresponde a esos redondeos y no exige igualdad exacta.

Corre con pytest (``pytest``) o directamente (``python tests/test_metodos.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cargar, sobrevivientes  # noqa: E402
from core.modelo import (  # noqa: E402
    Alternativa,
    Criterio,
    ModoPesos,
    ProblemaDecision,
)
from core.normalizacion import PoliticaMinimizar, Procedimiento  # noqa: E402
from core.pesos import (  # noqa: E402
    ErrorDePesos,
    Estimacion,
    consistencia,
    del_problema,
    directos,
    matriz_pareada,
    prioridades,
)
from core.pesos import sugerencias  # noqa: E402
from models import ContextoEvaluacion, Ideal, Metrica, obtener  # noqa: E402
from models.comparacion import comparar, spearman  # noqa: E402
from models.sensibilidad import (  # noqa: E402
    barrido_peso,
    escenarios,
    pesos_con,
    resumen_escenarios,
)


def proyectos_de_inversion() -> ContextoEvaluacion:
    """Ejemplo del inversionista (Tablas 6 a 9), con las elecciones del libro.

    El libro convierte los criterios a minimizar por inversión (1 / a) y
    normaliza por la suma (procedimiento 3).
    """
    p = ProblemaDecision()
    for i in range(1, 5):
        p.agregar_alternativa(Alternativa(f"p{i}", f"Proyecto {i}"))
    for id_crit, sentido in [("IA", "max"), ("RE", "max"), ("RI", "min"), ("TR", "min")]:
        p.agregar_criterio(Criterio(id_crit, id_crit, sentido))
    tabla_6 = [[8, 14, 1, 7], [3, 16, 5, 2], [5, 13, 1, 4], [3, 20, 3, 5]]
    for i, fila in enumerate(tabla_6):
        for j, valor in enumerate(fila):
            p.evaluar(f"p{i + 1}", p.ids_criterios[j], valor)
    return ContextoEvaluacion(
        p,
        pesos=[5, 8, 7, 3],  # Tabla 7, escala 1 a 10
        procedimiento=Procedimiento.SUMA,
        politica=PoliticaMinimizar.INVERSION,
    )


# Ejemplo de las computadoras (sección 14): matriz de criterios P, VP, G, ST.
CRITERIOS_PC = np.array(
    [[1, 3, 2, 5], [1 / 3, 1, 1 / 2, 1 / 2], [1 / 2, 2, 1, 2], [1 / 5, 2, 1 / 2, 1]]
)


# ------------------------------------------------------------------- pesos


def test_pesos_directos_se_normalizan():
    # Tabla 9: los pesos 5, 8, 7, 3 normalizados.
    assert np.allclose(directos([5, 8, 7, 3]), [0.217, 0.348, 0.304, 0.130], atol=5e-4)


def test_pesos_directos_rechazan_negativos():
    try:
        directos([1, -1])
    except ErrorDePesos:
        return
    raise AssertionError("Se esperaba un error con un peso negativo.")


def test_ahp_pesos_de_criterios_como_en_el_libro():
    w = prioridades(CRITERIOS_PC, Estimacion.APROXIMADA)
    assert np.allclose(w, [0.4888, 0.1182, 0.2453, 0.1478], atol=1e-4)


def test_ahp_consistencia_como_en_el_libro():
    c = consistencia(CRITERIOS_PC)
    assert np.isclose(c.lambda_max, 4.1345, atol=1e-4)
    assert np.isclose(c.ic, 0.0448, atol=1e-4)
    # Con IA = 0,90 (tabla de la cátedra) se obtiene el 0,0498 del libro.
    assert np.isclose(c.ia, 0.90)
    assert np.isclose(c.rc, 0.0498, atol=1e-4)
    assert c.aceptable


def test_ahp_sintesis_global_como_en_el_libro():
    # Etapa 5: prioridades de HP, IBM y Compaq por criterio, y ponderación
    # lineal con los pesos de los criterios. IBM gana con 0,3951.
    por_criterio = [
        [[1, 2, 3], [1 / 2, 1, 2], [1 / 3, 1 / 2, 1]],  # Precio
        [[1, 1 / 2, 1 / 4], [2, 1, 1 / 3], [4, 3, 1]],  # Velocidad
        [[1, 1 / 6, 1 / 3], [6, 1, 3], [3, 1 / 3, 1]],  # Garantía
        [[1, 1 / 4, 1 / 7], [4, 1, 1], [7, 1, 1]],  # Servicio técnico
    ]
    P = np.column_stack([prioridades(np.array(M)) for M in por_criterio])
    assert np.allclose(P[:, 0], [0.5390, 0.2973, 0.1638], atol=1e-4)
    assert np.allclose(P[:, 3], [0.0870, 0.4148, 0.4981], atol=1e-4)
    U = P @ prioridades(CRITERIOS_PC)
    # Para HP el libro informa 0,3163, pero rehaciendo la cuenta con sus
    # propios valores redondeados da 0,3161: la diferencia es del libro.
    assert np.allclose(U, [0.3163, 0.3951, 0.2889], atol=5e-4)
    assert int(np.argmax(U)) == 1


def test_ahp_autovector_coincide_con_matriz_consistente():
    # En una matriz perfectamente consistente (a_ij = w_i / w_j) ambas
    # estimaciones recuperan w exacto y lambda_max = n.
    w = np.array([0.5, 0.3, 0.2])
    M = w[:, None] / w[None, :]
    for est in Estimacion:
        assert np.allclose(prioridades(M, est), w)
    c = consistencia(M)
    assert np.isclose(c.lambda_max, 3.0)
    assert np.isclose(c.rc, 0.0)


def test_ahp_detecta_juicios_inconsistentes():
    # A > B, B > C, pero C > A con fuerza: circular.
    M = matriz_pareada(["a", "b", "c"], {("a", "b"): 5, ("b", "c"): 5, ("c", "a"): 5})
    assert not consistencia(M).aceptable


def test_matriz_pareada_completa_reciprocos_y_neutros():
    M = matriz_pareada(["a", "b", "c"], {("b", "a"): 4})
    assert np.isclose(M[0, 1], 0.25) and np.isclose(M[1, 0], 4.0)
    assert np.isclose(M[0, 2], 1.0)  # par sin juicio: igual importancia


def test_pesos_del_problema_segun_modo():
    p = cargar()
    p.modo_pesos = ModoPesos.DIRECTO
    assert np.isclose(del_problema(p).pesos.sum(), 1.0)
    p.modo_pesos = ModoPesos.AHP
    res = del_problema(p)
    assert np.isclose(res.pesos.sum(), 1.0)
    assert res.consistencia is not None and res.consistencia.aceptable


def test_pesos_directos_faltantes_se_reportan():
    p = cargar()
    del p.pesos_directos["plazo"]
    try:
        del_problema(p)
    except ErrorDePesos as exc:
        assert "Plazo" in str(exc)
        return
    raise AssertionError("Se esperaba un error por peso faltante.")


def test_quitar_criterio_limpia_pesos_y_comparaciones():
    p = cargar()
    p.quitar_criterio("costo")
    assert "costo" not in p.pesos_directos
    assert all("costo" not in par for par in p.comparaciones)


# ---------------------------------------------------------------- métodos


def test_ponderacion_lineal_como_en_el_libro():
    res = obtener("ponderacion_lineal").evaluar(proyectos_de_inversion())
    assert np.allclose(res.puntajes, [0.306, 0.206, 0.279, 0.208], atol=1e-3)
    assert res.orden == [0, 2, 3, 1]


def topsis_cap12(metrica: Metrica):
    """TOPSIS como lo resuelve el capítulo: todo convertido a maximizar."""
    return obtener("topsis", metrica=metrica, ideal=Ideal.ORIENTADO)


def test_topsis_metrica_ciudad_como_en_el_libro():
    res = topsis_cap12(Metrica.CIUDAD).evaluar(proyectos_de_inversion())
    # Tolerancia mayor: el libro parte de v_ij redondeados a 3 decimales.
    assert np.allclose(res.puntajes, [0.676, 0.251, 0.562, 0.261], atol=2e-3)
    assert res.orden == [0, 2, 3, 1]


def test_topsis_ciudad_ordena_igual_que_ponderacion_lineal():
    # "Utilizando la métrica m1, los métodos TOPSIS y Ponderación lineal
    # proporcionarán la misma ordenación." Se comprueba en ambos casos.
    for ctx in (proyectos_de_inversion(), _contexto_caso()):
        pl = obtener("ponderacion_lineal").evaluar(ctx)
        tp = topsis_cap12(Metrica.CIUDAD).evaluar(ctx)
        assert pl.orden == tp.orden


def test_topsis_la_metrica_puede_cambiar_la_ganadora():
    # Ejercicio propuesto por el libro: resolver con m2 y m∞ y comparar.
    # Con m∞ el Proyecto 3 desplaza al 1.
    ctx = proyectos_de_inversion()
    assert topsis_cap12(Metrica.EUCLIDEA).evaluar(ctx).ganadora == 0
    assert topsis_cap12(Metrica.CHEBYSHEV).evaluar(ctx).ganadora == 2


def test_topsis_indice_entre_cero_y_uno():
    for m in Metrica:
        C = obtener("topsis", metrica=m).evaluar(_contexto_caso()).puntajes
        assert np.all((C >= 0) & (C <= 1))


def test_topsis_de_clase_toma_el_ideal_segun_el_sentido():
    # Con un criterio a minimizar, el ideal de esa columna es el mínimo.
    res = obtener("topsis").evaluar(proyectos_de_inversion())
    ideal = res.detalle["Soluciones ideales"].loc["A+ (ideal)"].to_numpy()
    V = res.detalle["Matriz ponderada (v_ij = w_j · r_ij)"].to_numpy()
    assert np.allclose(ideal, [V[:, 0].max(), V[:, 1].max(), V[:, 2].min(), V[:, 3].min()])


def test_topsis_de_clase_no_depende_de_la_politica():
    ctx = proyectos_de_inversion()
    topsis = obtener("topsis")
    a = topsis.evaluar(ctx).puntajes
    ctx.politica = PoliticaMinimizar.COMPLEMENTO
    assert np.allclose(topsis.evaluar(ctx).puntajes, a)


def test_topsis_ambas_versiones_coinciden_con_rango_y_complemento():
    # Normalizar por rango el complemento es reflejar la columna (1 - r): las
    # distancias al ideal son las mismas que tomando el mínimo como ideal.
    ctx = proyectos_de_inversion()
    ctx.procedimiento = Procedimiento.RANGO
    ctx.politica = PoliticaMinimizar.COMPLEMENTO
    for m in Metrica:
        clase = obtener("topsis", metrica=m).evaluar(ctx).puntajes
        assert np.allclose(clase, topsis_cap12(m).evaluar(ctx).puntajes)


def test_ahp_completo_como_en_el_libro():
    # El ejemplo de las computadoras de punta a punta con el método AHP:
    # pesos de criterios y comparaciones entre alternativas por criterio.
    p = ProblemaDecision()
    for nombre in ("HP", "IBM", "Compaq"):
        p.agregar_alternativa(Alternativa(nombre.lower(), nombre))
    for c in ("P", "VP", "G", "ST"):
        p.agregar_criterio(Criterio(c, c))
    triangulos = {  # a_12, a_13, a_23 de cada matriz del ejemplo
        "P": (2, 3, 2),
        "VP": (1 / 2, 1 / 4, 1 / 3),
        "G": (1 / 6, 1 / 3, 3),
        "ST": (1 / 4, 1 / 7, 1),
    }
    for c, (a12, a13, a23) in triangulos.items():
        p.comparaciones_alternativas[c] = {
            ("hp", "ibm"): a12,
            ("hp", "compaq"): a13,
            ("ibm", "compaq"): a23,
        }
    res = obtener("ahp").evaluar(ContextoEvaluacion(p, prioridades(CRITERIOS_PC)))
    assert np.allclose(res.puntajes, [0.3163, 0.3951, 0.2889], atol=5e-4)
    assert res.ganadora == 1
    cons = res.detalle["Consistencia de los juicios entre alternativas"]
    assert (cons["RC"] <= 0.10).all()


def test_ahp_usa_los_datos_en_criterios_sin_juicios():
    # Sin juicios entre alternativas, AHP es Ponderación Lineal con
    # normalización por la suma: lo que dice el capítulo.
    ctx = proyectos_de_inversion()  # ya usa normalización por la suma
    ahp = obtener("ahp").evaluar(ctx)
    pl = obtener("ponderacion_lineal").evaluar(ctx)
    assert np.allclose(ahp.puntajes, pl.puntajes)


def test_ahp_mezcla_juicios_y_datos():
    p = cargar()
    idx = sobrevivientes(p)
    ids = [p.alternativas[i].id for i in idx]
    # La escuela es 3 veces más urgente que cada una de las demás.
    p.comparaciones_alternativas["urgencia"] = {
        ("escuela", otra): 3.0 for otra in ids if otra != "escuela"
    }
    ctx = ContextoEvaluacion(p, del_problema(p).pesos, indices=idx)
    P, cons = obtener("ahp").prioridades_alternativas(ctx)
    j = p.ids_criterios.index("urgencia")
    assert int(np.argmax(P[:, j])) == ids.index("escuela")
    assert list(cons) == [j]
    assert np.allclose(P.sum(axis=0), 1.0)


def test_metodos_evaluan_solo_el_subconjunto_pedido():
    ctx = _contexto_caso()
    res = obtener("ponderacion_lineal").evaluar(ctx)
    assert sorted(res.orden) == sorted(ctx.indices)
    assert len(res.puntajes) == len(ctx.indices)


def test_sugerencias_encuentran_el_juicio_incoherente():
    # Matriz consistente (pesos 0,4 / 0,3 / 0,2 / 0,1) con un juicio dado
    # vuelta: d sobre a con fuerza 5. La primera sugerencia debe ser ese par.
    w = np.array([0.4, 0.3, 0.2, 0.1])
    M = w[:, None] / w[None, :]
    M[0, 3], M[3, 0] = 1 / 5, 5
    assert consistencia(M).rc > 0.10
    primera = sugerencias(M)[0]
    assert (primera.i, primera.j) == (0, 3)
    assert np.isclose(primera.sugerido, 4.0)  # w_a / w_d
    assert np.isclose(primera.rc_nuevo, 0.0)


def test_sugerencias_vacias_con_dos_elementos():
    assert sugerencias(np.array([[1.0, 3.0], [1 / 3, 1.0]])) == []


# --------------------------------------------------------------- comparación


def test_spearman_extremos():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_comparar_metodos_sobre_el_caso():
    ctx = _contexto_caso()
    metodos = {
        "PL": obtener("ponderacion_lineal"),
        "AHP": obtener("ahp"),
        "TOPSIS": obtener("topsis"),
    }
    comp = comparar(ctx, metodos)
    assert list(comp.posiciones.columns) == ["PL", "AHP", "TOPSIS"]
    assert comp.posiciones.shape == (len(ctx.filas), 3)
    assert np.allclose(np.diag(comp.acuerdo), 1.0)
    assert comp.consenso()["Posición media"].is_monotonic_increasing


# ------------------------------------------------------------ sensibilidad


def test_pesos_con_conserva_las_proporciones_de_los_demas():
    w = pesos_con(np.array([0.5, 0.3, 0.2]), 0, 0.8)
    assert np.isclose(w[0], 0.8) and np.isclose(w.sum(), 1.0)
    assert np.isclose(w[1] / w[2], 0.3 / 0.2)


def test_barrido_ubica_el_cambio_de_ganadora():
    # Dos alternativas y dos criterios: a gana en c1 y b en c2. Con
    # Ponderación Lineal y normalización por el máximo el cruce es exacto:
    # U(a) = w + 0.5(1 - w), U(b) = 0.5w + (1 - w), se igualan en w = 0.5.
    p = ProblemaDecision()
    p.agregar_alternativa(Alternativa("a", "A"))
    p.agregar_alternativa(Alternativa("b", "B"))
    p.agregar_criterio(Criterio("c1", "C1"))
    p.agregar_criterio(Criterio("c2", "C2"))
    for alt, c1, c2 in [("a", 10, 5), ("b", 5, 10)]:
        p.evaluar(alt, "c1", c1)
        p.evaluar(alt, "c2", c2)
    ctx = ContextoEvaluacion(p, [0.7, 0.3])
    b = barrido_peso(obtener("ponderacion_lineal"), ctx, 0)
    assert b.ganadora_actual == "A"
    assert len(b.cambios) == 1
    peso, antes, despues = b.cambios[0]
    assert (antes, despues) == ("B", "A")
    assert abs(peso - 0.5) < 1e-3
    desde, hasta = b.intervalo_estable()
    assert abs(desde - 0.5) < 1e-3 and hasta == 1.0


def test_barrido_no_cuenta_un_empate_como_cambio():
    # A y B empatan en c2. Con todo el peso en c2 quedan iguales, pero eso no
    # es que B le gane a A: la ganadora vigente se mantiene.
    p = ProblemaDecision()
    p.agregar_alternativa(Alternativa("b", "B"))
    p.agregar_alternativa(Alternativa("a", "A"))
    p.agregar_criterio(Criterio("c1", "C1"))
    p.agregar_criterio(Criterio("c2", "C2"))
    for alt, c1, c2 in [("b", 5, 10), ("a", 10, 10)]:
        p.evaluar(alt, "c1", c1)
        p.evaluar(alt, "c2", c2)
    b = barrido_peso(obtener("ponderacion_lineal"), ContextoEvaluacion(p, [0.5, 0.5]), 1)
    assert b.ganadora_actual == "A"
    assert b.cambios == []


def test_escenarios_cubren_todas_las_combinaciones():
    ctx = _contexto_caso()
    metodos = {
        "PL": obtener("ponderacion_lineal"),
        "AHP": obtener("ahp"),
        "TOPSIS": obtener("topsis"),
    }
    tabla = escenarios(ctx, metodos)
    # PL varía normalización y política; AHP solo la política; TOPSIS de
    # clase solo la normalización.
    assert len(tabla) == 4 * 3 + 3 + 4
    resumen = resumen_escenarios(tabla, ctx.nombres)
    assert resumen["Veces primera"].sum() == tabla["Ganadora"].notna().sum()


def test_escenarios_registran_las_combinaciones_imposibles():
    # Una columna a minimizar con un cero no admite la inversión: el
    # escenario queda con su motivo en vez de abortar el análisis.
    p = ProblemaDecision()
    for i in (1, 2, 3):
        p.agregar_alternativa(Alternativa(f"a{i}", f"A{i}"))
    p.agregar_criterio(Criterio("c1", "C1", "min"))
    p.agregar_criterio(Criterio("c2", "C2"))
    for i, (c1, c2) in enumerate([(0, 5), (3, 2), (6, 9)], start=1):
        p.evaluar(f"a{i}", "c1", c1)
        p.evaluar(f"a{i}", "c2", c2)
    tabla = escenarios(ContextoEvaluacion(p, [1, 1]), {"PL": obtener("ponderacion_lineal")})
    fallidos = tabla[tabla["Ganadora"].isna()]
    assert len(fallidos) == 4 * 2  # inversión e inversión relativa, en cada normalización
    assert fallidos["Motivo"].str.contains("estrictamente positivos").all()


def _contexto_caso() -> ContextoEvaluacion:
    p = cargar()
    return ContextoEvaluacion(p, del_problema(p).pesos, indices=sobrevivientes(p))


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
