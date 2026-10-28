"""SADOP-Muni — interfaz del sistema de soporte a la decisión.

Siete pasos: definición del problema, matriz de decisión, preanálisis, pesos,
evaluación, comparación de métodos y sensibilidad. Los tres primeros no
dependen del método; los métodos registrados en ``models`` aparecen solos en
el selector del paso 5.

Ejecutar con::

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import a_json, cargar, sobrevivientes
from core.modelo import (
    Alternativa,
    Criterio,
    Escala,
    ModoPesos,
    ProblemaDecision,
    Sentido,
)
from core.normalizacion import (
    ErrorDeNormalizacion,
    PoliticaMinimizar,
    Procedimiento,
    normalizar,
)
from core.pesos import (
    ESCALA_SAATY,
    RC_MAXIMO,
    Consistencia,
    ErrorDePesos,
    Estimacion,
    consistencia,
    del_problema,
    matriz_pareada,
    prioridades,
    sugerencias,
)
from core.preanalisis import dominacion, satisfaccion
from models import (
    ContextoEvaluacion,
    ErrorDeEvaluacion,
    Ideal,
    MetodoDMD,
    Metrica,
    metodos_disponibles,
    obtener,
    usa_politica,
    usa_procedimiento,
)
from models.comparacion import Comparacion, comparar
from models.sensibilidad import Barrido, barrido_peso, escenarios, resumen_escenarios

st.set_page_config(page_title="SADOP-Muni", page_icon="🏗️", layout="wide")

PASOS = [
    "1 · Definición del problema",
    "2 · Matriz de decisión",
    "3 · Preanálisis",
    "4 · Pesos",
    "5 · Evaluación",
    "6 · Comparación de métodos",
    "7 · Sensibilidad",
]

#: Opciones del análisis que comparten los pasos 4 a 6. Viven en
#: ``st.session_state`` con prefijo ``cfg_`` para no perderse al cambiar de
#: paso (Streamlit borra el estado de los widgets que no se dibujan).
CONFIG_INICIAL = {
    "cfg_metodo": "ponderacion_lineal",
    "cfg_metrica": Metrica.EUCLIDEA,
    "cfg_ideal": Ideal.SEGUN_SENTIDO,
    "cfg_procedimiento": Procedimiento.MAXIMO,
    "cfg_politica": PoliticaMinimizar.COMPLEMENTO,
    "cfg_estimacion": Estimacion.APROXIMADA,
    "cfg_solo_eleccion": True,
}

#: Paleta categórica (validada para daltonismo en pares adyacentes). El color
#: sigue a la alternativa por su posición en el problema, no por su puesto en
#: el ranking, para que una misma obra tenga el mismo color en todo el sistema.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#8a8985"

#: Prefijos de las claves de widgets que dependen del caso cargado. Se borran
#: al cambiar de caso para que no pisen los datos nuevos con los anteriores.
CLAVES_DEL_CASO = ("editor_", "ahp_", "ahpalt_", "pareado_")

#: Posiciones de los controles de comparación de a pares: negativo, gana el
#: elemento de la izquierda por ese factor; 1, igual importancia; positivo,
#: gana el de la derecha.
OPCIONES_SAATY = [-k for k in range(9, 1, -1)] + [1] + list(range(2, 10))


def problema() -> ProblemaDecision:
    """Problema activo en la sesión, inicializado con el caso de ejemplo."""
    if "problema" not in st.session_state:
        st.session_state.problema = cargar()
    return st.session_state.problema


def cambiar_problema(nuevo: ProblemaDecision) -> None:
    """Reemplaza el caso activo y olvida el estado de los controles del anterior."""
    for clave in list(st.session_state):
        if str(clave).startswith(CLAVES_DEL_CASO):
            del st.session_state[clave]
    st.session_state.problema = nuevo


# ------------------------------------------------------------------ paso 1


def paso_definicion(p: ProblemaDecision) -> None:
    st.header("Definición del problema")
    st.caption(
        "Qué se decide, entre qué alternativas y con qué criterios. Es la "
        "etapa que condiciona todo lo demás: un criterio mal definido no se "
        "corrige con ningún método."
    )

    p.titulo = st.text_input("Título", p.titulo)
    p.descripcion = st.text_area("Descripción del problema", p.descripcion, height=90)
    p.decisor = st.text_input("Decisor", p.decisor)

    st.subheader("Alternativas")
    st.caption("Las obras públicas entre las que hay que elegir.")
    alts = st.data_editor(
        pd.DataFrame(
            [
                {"id": a.id, "nombre": a.nombre, "descripcion": a.descripcion}
                for a in p.alternativas
            ]
        ),
        num_rows="dynamic",
        width="stretch",
        key="editor_alternativas",
    )

    st.subheader("Criterios")
    st.caption(
        "El **sentido** indica si conviene un valor alto (maximizar) o bajo "
        "(minimizar). El **umbral** es opcional: fija el nivel mínimo "
        "aceptable para el preanálisis de satisfacción."
    )
    crits = st.data_editor(
        pd.DataFrame(
            [
                {
                    "id": c.id,
                    "nombre": c.nombre,
                    "sentido": c.sentido.value,
                    "unidad": c.unidad,
                    "escala": c.escala.value,
                    "umbral": c.umbral,
                    "descripcion": c.descripcion,
                }
                for c in p.criterios
            ]
        ),
        num_rows="dynamic",
        width="stretch",
        key="editor_criterios",
        column_config={
            "sentido": st.column_config.SelectboxColumn(
                options=[s.value for s in Sentido], required=True
            ),
            "escala": st.column_config.SelectboxColumn(
                options=[e.value for e in Escala], required=True
            ),
            "umbral": st.column_config.NumberColumn(help="Vacío = sin restricción"),
        },
    )

    if st.button("Aplicar cambios", type="primary"):
        _aplicar(p, alts, crits)
        st.rerun()


def _aplicar(p: ProblemaDecision, alts: pd.DataFrame, crits: pd.DataFrame) -> None:
    """Reemplaza alternativas y criterios conservando las evaluaciones válidas."""
    evaluaciones = dict(p.evaluaciones)

    p.alternativas = [
        Alternativa(
            id=str(r["id"]).strip(),
            nombre=str(r["nombre"]).strip(),
            descripcion=str(r.get("descripcion") or ""),
        )
        for _, r in alts.iterrows()
        if str(r.get("id") or "").strip()
    ]
    p.criterios = [
        Criterio(
            id=str(r["id"]).strip(),
            nombre=str(r["nombre"]).strip(),
            sentido=str(r["sentido"]),
            unidad=str(r.get("unidad") or ""),
            escala=str(r.get("escala") or "razon"),
            descripcion=str(r.get("descripcion") or ""),
            umbral=None if pd.isna(r.get("umbral")) else float(r["umbral"]),
        )
        for _, r in crits.iterrows()
        if str(r.get("id") or "").strip()
    ]
    # Se descartan las evaluaciones cuyo criterio o alternativa ya no existe.
    validas = {(a, c) for a in p.ids_alternativas for c in p.ids_criterios}
    p.evaluaciones = {k: v for k, v in evaluaciones.items() if k in validas}
    ids = set(p.ids_criterios)
    p.pesos_directos = {k: v for k, v in p.pesos_directos.items() if k in ids}
    p.comparaciones = {
        k: v for k, v in p.comparaciones.items() if k[0] in ids and k[1] in ids
    }


# ------------------------------------------------------------------ paso 2


def paso_matriz(p: ProblemaDecision) -> None:
    st.header("Matriz de decisión")
    st.caption(
        "Evaluación a_ij de cada alternativa respecto de cada criterio, en "
        "sus unidades originales. Todavía no se normaliza nada."
    )

    if not p.alternativas or not p.criterios:
        st.warning("Definí alternativas y criterios en el paso 1.")
        return

    editada = st.data_editor(
        p.tabla(),
        width="stretch",
        key="editor_matriz",
    )

    for i, a in enumerate(p.alternativas):
        for j, c in enumerate(p.criterios):
            valor = editada.iat[i, j]
            if pd.notna(valor):
                p.evaluar(a.id, c.id, float(valor))

    errores = p.validar()
    if errores:
        for e in errores:
            st.error(e)
    else:
        st.success("La matriz está completa y es consistente.")

    with st.expander("Ver la matriz normalizada"):
        st.caption(
            "El capítulo advierte que el resultado final puede verse "
            "considerablemente afectado por el procedimiento elegido. Probá "
            "cambiarlo y mirá cómo se mueven los valores."
        )
        col1, col2 = st.columns(2)
        proc = col1.selectbox(
            "Procedimiento",
            list(Procedimiento),
            format_func=lambda x: x.etiqueta,
        )
        pol = col2.selectbox(
            "Criterios a minimizar",
            list(PoliticaMinimizar),
            format_func=lambda x: x.etiqueta,
        )
        if not errores:
            try:
                R = normalizar(p.matriz(), p.sentidos, proc, pol)
                st.dataframe(
                    pd.DataFrame(
                        R,
                        index=[a.nombre for a in p.alternativas],
                        columns=[c.nombre for c in p.criterios],
                    ).style.format("{:.4f}"),
                    width="stretch",
                )
                st.caption("Tras normalizar, en todas las columnas más es mejor.")
            except ErrorDeNormalizacion as exc:
                st.error(str(exc))


# ------------------------------------------------------------------ paso 3


def paso_preanalisis(p: ProblemaDecision) -> None:
    st.header("Preanálisis")
    st.caption(
        "Filtros previos a cualquier método y a cualquier peso: dominación "
        "(secc. 9) y satisfacción (secc. 10). Reducen el conjunto de elección "
        "dejando constancia de por qué."
    )

    if p.validar():
        st.warning("Completá la matriz de decisión antes de correr el preanálisis.")
        return

    A, sentidos = p.matriz(), p.sentidos
    nombres = [a.nombre for a in p.alternativas]

    st.subheader("Dominación")
    dom = dominacion(A, sentidos)
    if dom.dominadas:
        for k, dominantes in dom.dominada_por.items():
            st.write(
                f"**{nombres[k]}** está dominada por "
                f"{', '.join(nombres[i] for i in dominantes)} "
                "(igual o peor en todos los criterios, peor en alguno)."
            )
    else:
        st.info("Ninguna alternativa está dominada: todas son eficientes.")

    st.subheader("Satisfacción")
    umbrales = [c.umbral for c in p.criterios]
    if any(u is not None for u in umbrales):
        sat = satisfaccion(A, sentidos, umbrales)
        if sat.descartadas:
            for i, fallas in sat.incumple.items():
                detalle = ", ".join(
                    f"{p.criterios[j].nombre} = {A[i, j]:g} "
                    f"(umbral {umbrales[j]:g})"
                    for j in fallas
                )
                st.write(f"**{nombres[i]}** no alcanza: {detalle}.")
        else:
            st.info("Todas las alternativas alcanzan los umbrales fijados.")
    else:
        st.info("No se fijaron umbrales. Se cargan en el paso 1, columna *umbral*.")

    quedan = sobrevivientes(p)
    st.subheader(f"Conjunto de elección: {len(quedan)} de {len(nombres)}")
    st.dataframe(p.tabla().iloc[quedan], width="stretch")


# ------------------------------------------------------------------ paso 4


def paso_pesos(p: ProblemaDecision) -> None:
    st.header("Pesos de los criterios")
    st.caption(
        "La importancia relativa que el decisor le da a cada criterio "
        "(secc. 11). Es independiente del método: Ponderación Lineal y TOPSIS "
        "usan el mismo vector."
    )

    if not p.criterios:
        st.warning("Definí los criterios en el paso 1.")
        return

    modos = list(ModoPesos)
    p.modo_pesos = st.radio(
        "Cómo asignar los pesos",
        modos,
        index=modos.index(p.modo_pesos),
        format_func=lambda m: m.etiqueta,
        horizontal=True,
    )

    if p.modo_pesos is ModoPesos.DIRECTO:
        _pesos_directos(p)
    else:
        _pesos_ahp(p)

    try:
        res = del_problema(p, st.session_state.cfg_estimacion)
    except ErrorDePesos as exc:
        st.error(str(exc))
        return

    if res.consistencia is not None:
        _mostrar_consistencia(
            res.matriz,
            res.consistencia,
            p.ids_criterios,
            [c.nombre for c in p.criterios],
            p.comparaciones,
            "ahp",
        )

    st.subheader("Pesos normalizados")
    nombres = [c.nombre for c in p.criterios]
    st.plotly_chart(
        grafico_barras(nombres, res.pesos, "Peso", ".1%"),
        width="stretch",
    )


def _pesos_directos(p: ProblemaDecision) -> None:
    st.caption(
        "Asignación directa: un valor por criterio en la escala que prefieras "
        "(el ejemplo del capítulo usa 1 a 10). Solo importan las proporciones: "
        "se normalizan para que sumen 1."
    )
    editada = st.data_editor(
        pd.DataFrame(
            {
                "criterio": [c.nombre for c in p.criterios],
                "peso": [p.pesos_directos.get(c.id) for c in p.criterios],
            }
        ),
        disabled=["criterio"],
        hide_index=True,
        width="stretch",
        key="editor_pesos",
        column_config={"peso": st.column_config.NumberColumn(min_value=0.0)},
    )
    for c, valor in zip(p.criterios, editada["peso"], strict=True):
        if pd.notna(valor):
            p.pesos_directos[c.id] = float(valor)
        else:
            p.pesos_directos.pop(c.id, None)


def _pesos_ahp(p: ProblemaDecision) -> None:
    n = len(p.criterios)
    st.caption(
        f"Comparaciones de a pares con la escala de Saaty (Tabla 10): "
        f"{n * (n - 1) // 2} juicios para {n} criterios. Mové cada control "
        "hacia el criterio que consideres más importante."
    )
    with st.expander("Escala Fundamental de Saaty"):
        st.table(
            pd.DataFrame(
                {"Valor": list(ESCALA_SAATY), "Significado": list(ESCALA_SAATY.values())}
            ).set_index("Valor")
        )

    nombres = [c.nombre for c in p.criterios]
    _juicios(p.ids_criterios, nombres, p.comparaciones, "ahp")
    _selector_estimacion()


def _selector_estimacion() -> None:
    st.radio(
        "Estimación de las prioridades",
        list(Estimacion),
        format_func=lambda e: e.etiqueta,
        horizontal=True,
        key="cfg_estimacion",
        help=(
            "El capítulo resuelve el ejemplo con la aproximada (Etapa 3); el "
            "autovector es el cálculo exacto del Anexo 1. Con juicios "
            "consistentes coinciden."
        ),
    )


def _rotulo_juicio(opcion: int, izquierda: str, derecha: str) -> str:
    if opcion == 1:
        return "Igual"
    return f"{izquierda} ×{-opcion}" if opcion < 0 else f"{derecha} ×{opcion}"


def _a_opcion(a_ij: float) -> int:
    """Juicio a_ij (cuánto más importante es i que j) -> posición del control."""
    fuerza = min(round(max(a_ij, 1 / a_ij)), 9)
    return 1 if fuerza <= 1 else -fuerza if a_ij > 1 else fuerza


def _a_juicio(opcion: int) -> float:
    return float(-opcion) if opcion < 0 else 1.0 if opcion == 1 else 1.0 / opcion


def _juicios(ids: list[str], nombres: list[str], juicios: dict, clave: str) -> None:
    """Controles de comparación de a pares; escribe lo elegido en ``juicios``."""
    M = matriz_pareada(ids, juicios)
    columnas = st.columns(2)
    k = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            key = f"{clave}_{ids[a]}_{ids[b]}"
            # Si el control ya existe, manda su estado (puede venir del
            # asistente de consistencia); si no, arranca desde el juicio guardado.
            inicial = {} if key in st.session_state else {"value": _a_opcion(M[a, b])}
            elegido = columnas[k % 2].select_slider(
                f"{nombres[a]}  ↔  {nombres[b]}",
                options=OPCIONES_SAATY,
                format_func=lambda v, x=nombres[a], y=nombres[b]: _rotulo_juicio(v, x, y),
                key=key,
                **inicial,
            )
            juicios.pop((ids[b], ids[a]), None)
            juicios[(ids[a], ids[b])] = _a_juicio(elegido)
            k += 1


def _mostrar_consistencia(
    M: np.ndarray,
    c: Consistencia,
    ids: list[str],
    nombres: list[str],
    juicios: dict,
    clave: str,
) -> None:
    """Métricas de la Etapa 4 y, si hace falta, qué juicios revisar."""
    with st.expander("Matriz de comparaciones"):
        st.dataframe(
            pd.DataFrame(M, index=nombres, columns=nombres).style.format("{:.3f}"),
            width="stretch",
        )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("λ máx", f"{c.lambda_max:.4f}")
    col2.metric("IC", f"{c.ic:.4f}")
    col3.metric("IA", f"{c.ia:.2f}")
    col4.metric("RC", "—" if c.rc is None else f"{c.rc:.4f}")
    if c.aceptable:
        st.success(
            f"✓ Juicios consistentes (RC ≤ {RC_MAXIMO:.2f}): la matriz no "
            "presenta inconsistencias serias."
        )
        return

    st.warning(
        f"⚠ RC = {c.rc:.3f} > {RC_MAXIMO:.2f}: conviene revisar los juicios "
        "antes de seguir. Suele deberse a comparaciones circulares (A > B, "
        "B > C, pero C > A)."
    )
    st.write("**Juicios a revisar.** Cambiando solo uno de estos, el RC queda en:")
    for sug in sugerencias(M, st.session_state.cfg_estimacion):
        x, y = nombres[sug.i], nombres[sug.j]
        col1, col2 = st.columns([5, 1])
        col1.write(
            f"{x} ↔ {y}: de **{_rotulo_juicio(_a_opcion(sug.actual), x, y)}** a "
            f"**{_rotulo_juicio(_a_opcion(sug.sugerido), x, y)}** → RC "
            f"{sug.rc_nuevo:.3f}"
        )
        col2.button(
            "Aplicar",
            key=f"{clave}_sug_{ids[sug.i]}_{ids[sug.j]}",
            on_click=_aplicar_sugerencia,
            args=(juicios, ids[sug.i], ids[sug.j], sug.sugerido, clave),
        )


def _aplicar_sugerencia(juicios: dict, i: str, j: str, valor: float, clave: str) -> None:
    juicios.pop((j, i), None)
    juicios[(i, j)] = valor
    # El control del juicio tiene su propio estado: hay que moverlo también.
    st.session_state[f"{clave}_{i}_{j}"] = _a_opcion(valor)


# ------------------------------------------------------------------ paso 5


def metodo_actual() -> MetodoDMD:
    """Instancia el método elegido en el paso 5 con sus opciones."""
    cfg = st.session_state
    opciones = {
        "topsis": {"metrica": cfg.cfg_metrica, "ideal": cfg.cfg_ideal},
        "ahp": {"estimacion": cfg.cfg_estimacion},
    }.get(cfg.cfg_metodo, {})
    return obtener(cfg.cfg_metodo, **opciones)


def metodos_principales() -> dict[str, MetodoDMD]:
    """Los métodos del capítulo, para comparar y para los escenarios."""
    cfg = st.session_state
    metodos: dict[str, MetodoDMD] = {
        "Ponderación Lineal": obtener("ponderacion_lineal"),
        "AHP": obtener("ahp", estimacion=cfg.cfg_estimacion),
    }
    for m in Metrica:
        metodos[f"TOPSIS {m.etiqueta.split(' ·')[0]}"] = obtener(
            "topsis", metrica=m, ideal=cfg.cfg_ideal
        )
    return metodos


def indices_a_evaluar(p: ProblemaDecision) -> list[int]:
    if st.session_state.cfg_solo_eleccion:
        return sobrevivientes(p)
    return list(range(len(p.alternativas)))


def preparar_evaluacion(p: ProblemaDecision) -> ContextoEvaluacion | None:
    """Arma el contexto con la configuración actual, o explica qué falta."""
    if p.validar():
        st.warning("Completá la matriz de decisión (paso 2) antes de evaluar.")
        return None
    try:
        pesos = del_problema(p, st.session_state.cfg_estimacion).pesos
    except ErrorDePesos as exc:
        st.warning(f"Revisá los pesos (paso 4): {exc}")
        return None

    cfg = st.session_state
    return ContextoEvaluacion(
        p,
        pesos,
        indices=indices_a_evaluar(p),
        procedimiento=cfg.cfg_procedimiento,
        politica=cfg.cfg_politica,
    )


def paso_evaluacion(p: ProblemaDecision) -> None:
    st.header("Evaluación")
    disponibles = metodos_disponibles()

    col1, col2, col3 = st.columns(3)
    col1.selectbox(
        "Método",
        list(disponibles),
        format_func=lambda k: disponibles[k].nombre,
        key="cfg_metodo",
    )
    if st.session_state.cfg_metodo == "topsis":
        col2.selectbox(
            "Métrica de distancia",
            list(Metrica),
            format_func=lambda m: m.etiqueta,
            key="cfg_metrica",
        )
        col3.selectbox(
            "Ideal y anti-ideal",
            list(Ideal),
            format_func=lambda i: i.etiqueta,
            key="cfg_ideal",
            help=(
                "Clase: el ideal es el máximo en los criterios a maximizar y el "
                "mínimo en los de minimizar. Cap. 12: primero se convierte todo "
                "a maximizar con la política elegida."
            ),
        )
    elif st.session_state.cfg_metodo == "ahp":
        col2.selectbox(
            "Estimación de las prioridades",
            list(Estimacion),
            format_func=lambda e: e.etiqueta,
            key="cfg_estimacion",
        )

    metodo = metodo_actual()
    col1, col2 = st.columns(2)
    col1.selectbox(
        "Normalización",
        list(Procedimiento),
        format_func=lambda x: x.etiqueta,
        key="cfg_procedimiento",
        disabled=not usa_procedimiento(metodo),
        help=None if usa_procedimiento(metodo) else "AHP normaliza siempre por la suma.",
    )
    col2.selectbox(
        "Criterios a minimizar",
        list(PoliticaMinimizar),
        format_func=lambda x: x.etiqueta,
        key="cfg_politica",
        disabled=not usa_politica(metodo),
        help=None if usa_politica(metodo) else "Con el ideal de clase no hace falta convertir.",
    )
    st.checkbox(
        "Evaluar solo el conjunto de elección (lo que sobrevivió al preanálisis)",
        key="cfg_solo_eleccion",
    )
    _nota_de_configuracion(p)

    if st.session_state.cfg_metodo == "ahp" and not p.validar():
        _juicios_alternativas(p)

    ctx = preparar_evaluacion(p)
    if ctx is None:
        return
    try:
        res = metodo.evaluar(ctx)
    except (ErrorDeNormalizacion, ErrorDeEvaluacion) as exc:
        st.error(str(exc))
        return

    st.divider()
    ganadora = p.alternativas[res.ganadora].nombre
    st.subheader(f"Resultado: {res.metodo}")
    st.metric("Alternativa preferida", ganadora)

    ranking = pd.DataFrame(res.ranking(), columns=["Posición", "Alternativa", "Puntaje"])
    col1, col2 = st.columns([3, 2])
    col1.plotly_chart(
        grafico_barras(list(ranking["Alternativa"]), ranking["Puntaje"].to_numpy(), "Puntaje", ".3f"),
        width="stretch",
    )
    col2.dataframe(
        ranking.set_index("Posición").style.format({"Puntaje": "{:.4f}"}),
        width="stretch",
    )
    col2.download_button(
        "Descargar ranking (CSV)",
        ranking.to_csv(index=False).encode("utf-8"),
        file_name="ranking.csv",
        mime="text/csv",
    )

    excluidas = sorted(set(range(len(p.alternativas))) - set(ctx.filas))
    if excluidas:
        st.caption(
            "Fuera de la evaluación por el preanálisis: "
            + ", ".join(p.alternativas[i].nombre for i in excluidas)
            + "."
        )

    with st.expander("Paso a paso"):
        st.write("**Pesos normalizados**")
        st.dataframe(
            pd.DataFrame([ctx.pesos], index=["w_j"], columns=[c.nombre for c in p.criterios]).style.format("{:.4f}"),
            width="stretch",
        )
        for titulo, tabla in res.detalle.items():
            st.write(f"**{titulo}**")
            st.dataframe(tabla.style.format("{:.4f}", na_rep="—"), width="stretch")


def _juicios_alternativas(p: ProblemaDecision) -> None:
    """Etapa 2 de AHP sobre alternativas: de dónde sale cada columna de P."""
    st.subheader("Prioridades de las alternativas (AHP)")
    st.caption(
        "Para cada criterio, AHP necesita cuánto se prefiere cada obra. En los "
        "criterios medidos (costo, plazo) alcanza con los datos de la matriz, "
        "normalizados por la suma. En los cualitativos conviene compararlas de "
        "a pares, como en los ejemplos de clase."
    )
    filas = indices_a_evaluar(p)
    ids = [p.alternativas[i].id for i in filas]
    nombres = [p.alternativas[i].nombre for i in filas]

    for c in p.criterios:
        key = f"pareado_{c.id}"
        # El estado del interruptor ya está actualizado al empezar la ejecución:
        # leerlo acá deja el título del desplegable al día en el mismo clic.
        pareado = st.session_state.get(key, c.id in p.comparaciones_alternativas)
        with st.expander(
            f"{c.nombre}: {'de a pares' if pareado else 'datos de la matriz'}",
            expanded=pareado,
        ):
            inicial = {} if key in st.session_state else {"value": pareado}
            if st.toggle("Comparar las alternativas de a pares", key=key, **inicial):
                juicios = p.comparaciones_alternativas.setdefault(c.id, {})
                st.caption(
                    "Mové cada control hacia la obra que **mejor** cumple el "
                    "criterio"
                    + (" (la de menor valor: es a minimizar)." if c.sentido is Sentido.MINIMIZAR else ".")
                )
                _juicios(ids, nombres, juicios, f"ahpalt_{c.id}")
                M = matriz_pareada(ids, juicios)
                w = prioridades(M, st.session_state.cfg_estimacion)
                st.dataframe(
                    pd.DataFrame({"Prioridad": w}, index=nombres).style.format("{:.4f}"),
                    width="stretch",
                )
                if len(ids) <= 10:
                    _mostrar_consistencia(
                        M, consistencia(M, w), ids, nombres, juicios, f"ahpalt_{c.id}"
                    )
            else:
                p.comparaciones_alternativas.pop(c.id, None)
                st.caption("Se usan los datos de la matriz normalizados por la suma.")


def _nota_de_configuracion(p: ProblemaDecision) -> None:
    """Relaciona la combinación elegida con lo que dice el capítulo."""
    cfg = st.session_state
    if cfg.cfg_metodo == "topsis":
        asociada = cfg.cfg_metrica.normalizacion_asociada
        if cfg.cfg_procedimiento is not asociada:
            st.caption(
                f"El capítulo empareja {cfg.cfg_metrica.etiqueta} con la "
                f"normalización *{asociada.etiqueta.lower()}*. Se puede combinar "
                "con otra, pero conviene saberlo."
            )
    elif cfg.cfg_metodo == "ahp" and p.modo_pesos is not ModoPesos.AHP:
        st.caption(
            "En AHP los pesos de los criterios también salen de comparaciones "
            "de a pares. Ahora se están usando pesos directos: se pueden "
            "cambiar en el paso 4."
        )


# ------------------------------------------------------------------ paso 6


def paso_comparacion(p: ProblemaDecision) -> None:
    st.header("Comparación de métodos")
    st.caption(
        "Los mismos pesos, la misma normalización y las mismas alternativas; "
        "solo cambia el método. Si métodos con supuestos distintos coinciden, "
        "el resultado es más fácil de defender."
    )
    ctx = preparar_evaluacion(p)
    if ctx is None:
        return
    comp = comparar(ctx, metodos_principales())
    for metodo, motivo in comp.errores.items():
        st.warning(f"{metodo}: {motivo}")
    if comp.posiciones.empty:
        return

    ganadoras = comp.ganadoras
    if ganadoras.nunique() == 1:
        st.success(f"Los {len(ganadoras)} métodos coinciden: gana **{ganadoras.iloc[0]}**.")
    else:
        detalle = "; ".join(
            f"**{alt}** ({', '.join(ganadoras[ganadoras == alt].index)})"
            for alt in ganadoras.unique()
        )
        st.info(f"Los métodos no coinciden en la ganadora: {detalle}.")

    st.plotly_chart(grafico_posiciones(p, comp), width="stretch")

    consenso = comp.consenso()
    st.subheader("Posiciones")
    st.dataframe(
        comp.posiciones.loc[consenso.index].join(consenso[["Posición media"]]).style.format(
            {"Posición media": "{:.2f}"}
        ),
        width="stretch",
    )

    st.subheader("Acuerdo entre métodos")
    st.caption(
        "Coeficiente de Spearman entre cada par de rankings: 1 si ordenan igual, "
        "0 si no hay relación, -1 si ordenan al revés."
    )
    st.dataframe(comp.acuerdo.style.format("{:.2f}"), width="stretch")
    pares = comp.acuerdo.where(np.triu(np.ones(comp.acuerdo.shape, dtype=bool), k=1)).stack()
    if not pares.empty:
        (a, b), minimo = pares.idxmin(), pares.min()
        st.caption(f"El menor acuerdo es entre {a} y {b} ({minimo:.2f}).")

    with st.expander("Puntajes de cada método"):
        st.caption("Cada método tiene su propia escala: comparar posiciones, no puntajes.")
        st.dataframe(comp.puntajes.style.format("{:.4f}"), width="stretch")


# ------------------------------------------------------------------ paso 7


# ------------------------------------------------------------------ paso 6


def paso_sensibilidad(p: ProblemaDecision) -> None:
    st.header("Análisis de sensibilidad")
    st.caption(
        "¿Qué tan firme es el resultado? Se usa la configuración del paso 5; "
        "cambiala ahí para analizar otra."
    )

    ctx = preparar_evaluacion(p)
    if ctx is None:
        return
    metodo = metodo_actual()
    try:
        actual = metodo.evaluar(ctx)
    except (ErrorDeNormalizacion, ErrorDeEvaluacion) as exc:
        st.error(str(exc))
        return
    ganadora = p.alternativas[actual.ganadora].nombre
    st.write(f"Configuración actual: **{actual.metodo}**. Alternativa preferida: **{ganadora}**.")

    tab_pesos, tab_escenarios = st.tabs(["Frente a los pesos", "Frente a las elecciones del analista"])
    with tab_pesos:
        _sensibilidad_pesos(p, metodo, ctx)
    with tab_escenarios:
        _sensibilidad_escenarios(ctx, ganadora)


def _sensibilidad_pesos(p: ProblemaDecision, metodo: MetodoDMD, ctx: ContextoEvaluacion) -> None:
    st.caption(
        "Se varía el peso de un criterio de 0 a 100 % y los demás se reescalan "
        "conservando sus proporciones. El intervalo estable es el rango en el "
        "que la ganadora actual no cambia."
    )
    barridos = [barrido_peso(metodo, ctx, j) for j in range(len(p.criterios))]

    filas = []
    for b in barridos:
        desde, hasta = b.intervalo_estable()
        # Solo cuentan los bordes donde efectivamente cambia la ganadora: un
        # borde en 0 % o 100 % sin cambio no es un riesgo.
        bordes = {
            peso: (antes if peso == desde else despues)
            for peso, antes, despues in b.cambios
            if peso in (desde, hasta)
        }
        margen, borde, rival = min(
            ((abs(peso - b.peso_actual), peso, otra) for peso, otra in bordes.items()),
            default=(np.inf, np.nan, "—"),
        )
        filas.append(
            {
                "Criterio": b.criterio,
                "Peso actual": b.peso_actual,
                "Estable desde": desde,
                "Estable hasta": hasta,
                "Margen": margen,
                "Cambia con peso": borde,
                "Pasa a ganar": rival,
            }
        )
    resumen = pd.DataFrame(filas).set_index("Criterio")
    st.dataframe(
        pd.DataFrame(
            {
                "Peso actual": resumen["Peso actual"].map("{:.1%}".format),
                "Intervalo estable": [
                    f"{d:.1%} – {h:.1%}"
                    for d, h in zip(resumen["Estable desde"], resumen["Estable hasta"], strict=True)
                ],
                "Cambia con peso": resumen["Cambia con peso"].map(
                    lambda x: "sin cambio" if np.isnan(x) else f"{x:.1%}"
                ),
                "Pasa a ganar": resumen["Pasa a ganar"],
            }
        ),
        width="stretch",
    )
    critico = resumen["Margen"].idxmin()
    if np.isfinite(resumen.loc[critico, "Margen"]):
        fila = resumen.loc[critico]
        sentido = "baja" if fila["Cambia con peso"] < fila["Peso actual"] else "sube"
        st.info(
            f"El criterio más sensible es **{critico}**: si su peso {sentido} de "
            f"{fila['Peso actual']:.1%} a {fila['Cambia con peso']:.1%}, la "
            f"preferida pasa a ser **{fila['Pasa a ganar']}**."
        )
    else:
        st.success(
            "La ganadora se mantiene para cualquier peso de cualquier criterio "
            "(variando uno a la vez)."
        )

    nombres = [c.nombre for c in p.criterios]
    j = st.selectbox("Ver el detalle de un criterio", range(len(nombres)), format_func=lambda k: nombres[k])
    b = barridos[j]
    st.plotly_chart(grafico_barrido(p, b), width="stretch")
    for peso, antes, despues in b.cambios:
        st.write(f"- Con peso **{peso:.1%}**: {antes} → **{despues}**")


def _sensibilidad_escenarios(ctx: ContextoEvaluacion, ganadora: str) -> None:
    st.caption(
        "Se repite la evaluación con cada combinación de método, normalización "
        "y política para los criterios a minimizar, con los mismos pesos (las "
        "opciones que un método no usa no se varían). El capítulo advierte que "
        "estas elecciones pueden cambiar el resultado."
    )
    tabla = escenarios(ctx, metodos_principales())
    validos = tabla["Ganadora"].notna().sum()
    gana = int((tabla["Ganadora"] == ganadora).sum())
    st.metric(f"Escenarios en los que gana {ganadora}", f"{gana} de {validos}")

    resumen = resumen_escenarios(tabla, ctx.nombres)
    st.dataframe(resumen.style.format({"Posición media": "{:.2f}"}), width="stretch")

    with st.expander(f"Los {len(tabla)} escenarios"):
        st.dataframe(tabla, width="stretch", hide_index=True)
    if validos < len(tabla):
        st.caption(
            f"{len(tabla) - validos} combinaciones no se pudieron evaluar con "
            "este caso; el motivo figura en la tabla."
        )


# ---------------------------------------------------------------- gráficos


def color_de(p: ProblemaDecision, nombre: str) -> str:
    """Color fijo de una alternativa, según su posición en el problema."""
    k = [a.nombre for a in p.alternativas].index(nombre)
    return PALETA[k] if k < len(PALETA) else GRIS


def grafico_barras(etiquetas: list[str], valores: np.ndarray, medida: str, formato: str) -> go.Figure:
    """Barras horizontales de una sola serie, de mayor a menor."""
    orden = np.argsort(valores)
    fig = go.Figure(
        go.Bar(
            x=np.asarray(valores)[orden],
            y=[etiquetas[k] for k in orden],
            orientation="h",
            marker_color=PALETA[0],
            texttemplate=f"%{{x:{formato}}}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{medida}: %{{x:{formato}}}<extra></extra>",
        )
    )
    fig.update_layout(
        barcornerradius=4,
        bargap=0.35,
        height=60 + 44 * len(etiquetas),
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def grafico_posiciones(p: ProblemaDecision, comp: Comparacion) -> go.Figure:
    """Posición de cada alternativa según cada método (1 arriba)."""
    pos = comp.posiciones
    metodos = list(pos.columns)
    etiquetar = len(pos.index) <= 4
    fig = go.Figure()
    for nombre, fila in pos.iterrows():
        fig.add_trace(
            go.Scatter(
                x=metodos,
                y=fila.values,
                name=nombre,
                mode="lines+markers",
                line=dict(color=color_de(p, nombre), width=2),
                marker=dict(size=10, line=dict(width=2, color="rgba(0,0,0,0)")),
                hovertemplate=f"{nombre}<br>%{{x}}: puesto %{{y}}<extra></extra>",
            )
        )
        if etiquetar:
            fig.add_annotation(
                x=metodos[-1], y=fila.values[-1], text=nombre, showarrow=False,
                xanchor="left", xshift=12, font=dict(size=11),
            )
    n = len(pos.index)
    fig.update_layout(
        height=120 + 50 * n,
        margin=dict(l=0, r=200 if etiquetar else 10, t=10, b=10),
        yaxis=dict(
            title="Posición", autorange="reversed", tickmode="array",
            tickvals=list(range(1, n + 1)), showgrid=False, zeroline=False,
        ),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def grafico_barrido(p: ProblemaDecision, b: Barrido) -> go.Figure:
    """Puntaje de cada alternativa en función del peso de un criterio."""
    fig = go.Figure()
    desde, hasta = b.intervalo_estable()
    fig.add_vrect(
        x0=desde, x1=hasta, fillcolor=GRIS, opacity=0.12, line_width=0,
        annotation_text="estable", annotation_position="top left",
    )
    fig.add_vline(
        x=b.peso_actual, line_dash="dot", line_color=GRIS, line_width=1.5,
        annotation_text=f"actual {b.peso_actual:.0%}", annotation_position="top right",
    )
    etiquetar = len(b.puntajes.columns) <= 4
    for nombre in b.puntajes.columns:
        serie = b.puntajes[nombre]
        fig.add_trace(
            go.Scatter(
                x=serie.index,
                y=serie.values,
                name=nombre,
                mode="lines",
                line=dict(color=color_de(p, nombre), width=2),
                hovertemplate="%{y:.3f}",
            )
        )
        if etiquetar:
            fig.add_annotation(
                x=1.0, y=serie.iloc[-1], text=nombre, showarrow=False,
                xanchor="left", xshift=6, font=dict(size=11),
            )
    fig.update_layout(
        hovermode="x unified",
        height=420,
        margin=dict(l=0, r=180 if etiquetar else 10, t=30, b=10),
        xaxis=dict(title=f"Peso de {b.criterio}", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="Puntaje"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig

# ------------------------------------------------------------------- shell


def main() -> None:
    p = problema()
    for clave, valor in CONFIG_INICIAL.items():
        # Reasignar el valor evita que Streamlit lo descarte cuando el widget
        # que lo controla no se dibuja en esta ejecución.
        st.session_state[clave] = st.session_state.get(clave, valor)

    st.sidebar.title("SADOP-Muni")
    st.sidebar.caption("Soporte a la decisión para priorizar obras públicas")
    paso = st.sidebar.radio("Pasos", PASOS, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.metric("Alternativas", len(p.alternativas))
    st.sidebar.metric("Criterios", len(p.criterios))
    faltan = len(p.faltantes())
    if faltan:
        st.sidebar.warning(f"{faltan} evaluaciones sin cargar")

    st.sidebar.divider()
    st.sidebar.download_button(
        "Descargar caso (JSON)",
        a_json(p),
        file_name="caso.json",
        mime="application/json",
        width="stretch",
    )
    subido = st.sidebar.file_uploader("Cargar caso (JSON)", type="json")
    # El archivo queda en el control después de cargarlo: sin esta marca se
    # volvería a cargar en cada interacción, descartando las ediciones.
    if subido is not None and st.session_state.get("archivo_cargado") != subido.file_id:
        import json

        from core.io_casos import desde_dict

        st.session_state.archivo_cargado = subido.file_id
        cambiar_problema(desde_dict(json.load(subido)))
        st.rerun()
    if st.sidebar.button("Restaurar caso de ejemplo", width="stretch"):
        cambiar_problema(cargar())
        st.rerun()

    {
        PASOS[0]: paso_definicion,
        PASOS[1]: paso_matriz,
        PASOS[2]: paso_preanalisis,
        PASOS[3]: paso_pesos,
        PASOS[4]: paso_evaluacion,
        PASOS[5]: paso_comparacion,
        PASOS[6]: paso_sensibilidad,
    }[paso](p)


if __name__ == "__main__":
    main()
