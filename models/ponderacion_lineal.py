"""Ponderación Lineal (Cap. 12, sección 13).

La preferencia global de cada alternativa es la suma ponderada de sus
evaluaciones normalizadas:

    U(A_i) = sum_j w_j r_ij

Es compensatorio, requiere evaluaciones y pesos cardinales y es muy sensible
al procedimiento de normalización, que por eso se recibe del contexto en vez
de fijarse acá.

Con pesos obtenidos por comparaciones pareadas y normalización por la suma,
es la Etapa 5 de AHP: el capítulo presenta AHP como "una variante del método
de Ponderación Lineal en la cual los pesos se asignan por comparaciones de a
pares".
"""

from __future__ import annotations

from .base import ContextoEvaluacion, Resultado, ordenar_por_puntaje, registrar


@registrar
class PonderacionLineal:
    id = "ponderacion_lineal"
    nombre = "Ponderación Lineal"

    def evaluar(self, ctx: ContextoEvaluacion) -> Resultado:
        R = ctx.matriz_normalizada()
        V = R * ctx.pesos
        U = V.sum(axis=1)
        return Resultado(
            metodo=self.nombre,
            puntajes=U,
            orden=ordenar_por_puntaje(ctx.filas, U),
            nombres=ctx.nombres,
            detalle={
                "Matriz normalizada (r_ij)": ctx.rotular(R),
                "Contribución de cada criterio (w_j · r_ij)": ctx.rotular(V),
            },
        )
