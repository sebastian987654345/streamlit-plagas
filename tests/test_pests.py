"""Verifica que el catálogo de recomendaciones siga sincronizado con el modelo."""

import pytest

from src import pests

# Clases con las que fue entrenado models/best6.pt.
CLASES_DEL_MODELO = {
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
}


def test_catalogo_cubre_todas_las_clases():
    faltantes = CLASES_DEL_MODELO - set(pests.CATALOGO)
    assert not faltantes, f"Clases sin ficha de manejo: {sorted(faltantes)}"


def test_catalogo_no_tiene_clases_inventadas():
    sobrantes = set(pests.CATALOGO) - CLASES_DEL_MODELO
    assert not sobrantes, f"Fichas que el modelo no puede detectar: {sorted(sobrantes)}"


@pytest.mark.parametrize("clase", sorted(CLASES_DEL_MODELO))
def test_cada_ficha_esta_completa(clase):
    especie = pests.CATALOGO[clase]
    assert especie.nombre_es.strip()
    assert especie.emoji.strip()
    assert len(especie.impacto) > 20
    assert len(especie.recomendacion) > 20


def test_clase_desconocida_tiene_fallback():
    especie = pests.obtener("Xenomorfo")
    assert especie is pests.ESPECIE_DESCONOCIDA
    assert especie.nivel is pests.Nivel.VIGILANCIA


def test_especies_beneficas_identificadas():
    beneficas = {c for c, e in pests.CATALOGO.items() if e.nivel is pests.Nivel.BENEFICO}
    assert beneficas == {"Bees", "Earthworms"}
