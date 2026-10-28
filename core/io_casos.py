"""Persistencia de casos de decisión en JSON.

Formato del archivo::

    {
      "titulo": "...", "descripcion": "...", "decisor": "...",
      "alternativas": [{"id": "...", "nombre": "...", "descripcion": "..."}],
      "criterios":    [{"id": "...", "nombre": "...", "sentido": "min|max",
                        "unidad": "...", "escala": "...", "umbral": 0}],
      "evaluaciones": {"<id_alt>": {"<id_crit>": 0.0}},
      "pesos": {"modo": "directo|ahp",
                "directos": {"<id_crit>": 0.0},
                "comparaciones": {"<id_crit_i>": {"<id_crit_j>": 0.0}},
                "alternativas": {"<id_crit>": {"<id_alt_i>": {"<id_alt_j>": 0.0}}}}
    }

El bloque ``pesos`` es opcional, para seguir leyendo casos guardados antes de
que existiera la asignación de pesos.
"""

from __future__ import annotations

import json
from pathlib import Path

from .modelo import Alternativa, Criterio, ModoPesos, ProblemaDecision

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
CASO_EJEMPLO = DIR_DATOS / "caso_ejemplo.json"


def desde_dict(datos: dict) -> ProblemaDecision:
    """Reconstruye un problema a partir de un diccionario ya parseado."""
    problema = ProblemaDecision(
        titulo=datos.get("titulo", "Problema de decisión"),
        descripcion=datos.get("descripcion", ""),
        decisor=datos.get("decisor", ""),
    )
    for a in datos.get("alternativas", []):
        problema.agregar_alternativa(
            Alternativa(
                id=a["id"], nombre=a["nombre"], descripcion=a.get("descripcion", "")
            )
        )
    for c in datos.get("criterios", []):
        problema.agregar_criterio(
            Criterio(
                id=c["id"],
                nombre=c["nombre"],
                sentido=c.get("sentido", "max"),
                unidad=c.get("unidad", ""),
                escala=c.get("escala", "razon"),
                descripcion=c.get("descripcion", ""),
                umbral=c.get("umbral"),
            )
        )
    for id_alt, fila in datos.get("evaluaciones", {}).items():
        for id_crit, valor in fila.items():
            problema.evaluar(id_alt, id_crit, valor)

    pesos = datos.get("pesos", {})
    problema.modo_pesos = ModoPesos(pesos.get("modo", "directo"))
    problema.pesos_directos = {
        k: float(v) for k, v in pesos.get("directos", {}).items()
    }
    problema.comparaciones = {
        (i, j): float(v)
        for i, fila in pesos.get("comparaciones", {}).items()
        for j, v in fila.items()
    }
    problema.comparaciones_alternativas = {
        id_crit: {(i, j): float(v) for i, fila in juicios.items() for j, v in fila.items()}
        for id_crit, juicios in pesos.get("alternativas", {}).items()
    }
    return problema


def a_dict(problema: ProblemaDecision) -> dict:
    """Serializa el problema a un diccionario JSON-compatible."""
    return {
        "titulo": problema.titulo,
        "descripcion": problema.descripcion,
        "decisor": problema.decisor,
        "alternativas": [
            {"id": a.id, "nombre": a.nombre, "descripcion": a.descripcion}
            for a in problema.alternativas
        ],
        "criterios": [
            {
                "id": c.id,
                "nombre": c.nombre,
                "sentido": c.sentido.value,
                "unidad": c.unidad,
                "escala": c.escala.value,
                "descripcion": c.descripcion,
                "umbral": c.umbral,
            }
            for c in problema.criterios
        ],
        "evaluaciones": {
            a.id: {
                c.id: problema.evaluaciones[(a.id, c.id)]
                for c in problema.criterios
                if (a.id, c.id) in problema.evaluaciones
            }
            for a in problema.alternativas
        },
        "pesos": {
            "modo": problema.modo_pesos.value,
            "directos": dict(problema.pesos_directos),
            "comparaciones": _anidar(problema.comparaciones),
            "alternativas": {
                c: _anidar(juicios)
                for c, juicios in problema.comparaciones_alternativas.items()
            },
        },
    }


def _anidar(pares: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    anidado: dict[str, dict[str, float]] = {}
    for (i, j), valor in pares.items():
        anidado.setdefault(i, {})[j] = valor
    return anidado


def cargar(ruta: str | Path = CASO_EJEMPLO) -> ProblemaDecision:
    """Lee un caso desde un archivo JSON."""
    ruta = Path(ruta)
    with ruta.open(encoding="utf-8") as f:
        return desde_dict(json.load(f))


def guardar(problema: ProblemaDecision, ruta: str | Path) -> Path:
    """Escribe el caso a un archivo JSON."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        json.dump(a_dict(problema), f, ensure_ascii=False, indent=2)
    return ruta


def a_json(problema: ProblemaDecision) -> str:
    """Serializa a texto JSON (para el botón de descarga de la interfaz)."""
    return json.dumps(a_dict(problema), ensure_ascii=False, indent=2)
