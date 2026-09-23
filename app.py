"""SADOP-Muni — interfaz del sistema de soporte a la decisión.

Cubre los pasos que **no dependen del método de evaluación**: definición del
problema, carga de la matriz de decisión y preanálisis. El paso de evaluación
queda deliberadamente pendiente hasta que se decida el método; en cuanto haya
uno registrado en ``models``, aparece solo en el selector.

Ejecutar con::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import a_json, cargar, sobrevivientes
from core.modelo import Alternativa, Criterio, Escala, ProblemaDecision, Sentido
from core.normalizacion import (
    ErrorDeNormalizacion,
    PoliticaMinimizar,
    Procedimiento,
    normalizar,
)
from core.preanalisis import dominacion, satisfaccion
from models import metodos_disponibles

st.set_page_config(page_title="SADOP-Muni", page_icon="🏗️", layout="wide")

PASOS = [
    "1 · Definición del problema",
    "2 · Matriz de decisión",
    "3 · Preanálisis",
    "4 · Evaluación",
]


def problema() -> ProblemaDecision:
    """Problema activo en la sesión, inicializado con el caso de ejemplo."""
    if "problema" not in st.session_state:
        st.session_state.problema = cargar()
    return st.session_state.problema


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
    st.session_state.sobrevivientes = quedan


# ------------------------------------------------------------------ paso 4


def paso_evaluacion(p: ProblemaDecision) -> None:
    st.header("Evaluación")
    disponibles = metodos_disponibles()

    if not disponibles:
        st.info(
            "**Método de evaluación pendiente de definición.**\n\n"
            "La base del sistema (definición del problema, matriz, manejo de "
            "criterios a minimizar, normalizaciones y preanálisis) ya está "
            "implementada y es independiente del método. Cuando se decida "
            "cuál usar, se implementa contra la interfaz de `models/base.py` "
            "y este paso se habilita solo."
        )
        st.markdown(
            "Candidatos del Cap. 12:\n"
            "- **Ponderación Lineal** — suma ponderada; simple, compensatorio, "
            "muy sensible a la normalización.\n"
            "- **AHP** — pesos por comparaciones pareadas de Saaty, con ratio "
            "de consistencia.\n"
            "- **TOPSIS** — distancia a la solución ideal y anti-ideal."
        )
        return

    st.selectbox(
        "Método",
        list(disponibles),
        format_func=lambda k: disponibles[k].nombre,
    )
    st.warning("Falta conectar la asignación de pesos y el cálculo del ranking.")


# ------------------------------------------------------------------- shell


def main() -> None:
    p = problema()

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
    if subido is not None:
        import json

        from core.io_casos import desde_dict

        st.session_state.problema = desde_dict(json.load(subido))
        st.rerun()
    if st.sidebar.button("Restaurar caso de ejemplo", width="stretch"):
        st.session_state.problema = cargar()
        st.rerun()

    {
        PASOS[0]: paso_definicion,
        PASOS[1]: paso_matriz,
        PASOS[2]: paso_preanalisis,
        PASOS[3]: paso_evaluacion,
    }[paso](p)


if __name__ == "__main__":
    main()
