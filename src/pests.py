"""Catálogo de especies detectables y sus recomendaciones de manejo.

Las claves coinciden exactamente con los nombres de clase del modelo
(`models/best6.pt`, 12 clases). El test `tests/test_pests.py` verifica que
no se desincronicen.

Las recomendaciones se redactan en tercera persona formal (usted), que es el
registro habitual en el campo colombiano y el que evita sonar extranjero.
"""

from dataclasses import dataclass
from enum import Enum


class Nivel(str, Enum):
    """Qué tan urgente es actuar sobre la especie detectada."""

    BENEFICO = "benefico"
    VIGILANCIA = "vigilancia"
    CONTROL = "control"


@dataclass(frozen=True)
class Especie:
    nombre_es: str
    emoji: str
    nivel: Nivel
    impacto: str
    recomendacion: str


CATALOGO: dict[str, Especie] = {
    "Ants": Especie(
        nombre_es="Hormigas",
        emoji="🐜",
        nivel=Nivel.VIGILANCIA,
        impacto=(
            "Rara vez dañan la planta directamente, pero protegen y trasladan "
            "pulgones y cochinillas a cambio de melaza."
        ),
        recomendacion=(
            "Ataque la causa: controle primero los pulgones. Aplique barreras "
            "pegajosas en el tronco y cebos con ácido bórico lejos del cultivo."
        ),
    ),
    "Bees": Especie(
        nombre_es="Abejas",
        emoji="🐝",
        nivel=Nivel.BENEFICO,
        impacto="Polinizador clave: su presencia mejora el cuajado y el rendimiento.",
        recomendacion=(
            "No aplique insecticidas. Si es indispensable fumigar, hágalo al "
            "atardecer, cuando ya no hay actividad de vuelo."
        ),
    ),
    "Beetles": Especie(
        nombre_es="Escarabajos",
        emoji="🪲",
        nivel=Nivel.CONTROL,
        impacto="Defoliación y perforación de hojas, flores y frutos.",
        recomendacion=(
            "Recolección manual temprano en la mañana, trampas de luz y "
            "aplicaciones puntuales de Beauveria bassiana."
        ),
    ),
    "Caterpillars": Especie(
        nombre_es="Orugas",
        emoji="🐛",
        nivel=Nivel.CONTROL,
        impacto=(
            "Una de las plagas más destructivas: consumen follaje y perforan "
            "frutos en pocos días."
        ),
        recomendacion=(
            "Aplique Bacillus thuringiensis (Bt) en estadios larvales tempranos "
            "y revise el envés de las hojas para eliminar posturas de huevos."
        ),
    ),
    "Earthworms": Especie(
        nombre_es="Lombrices de tierra",
        emoji="🪱",
        nivel=Nivel.BENEFICO,
        impacto="Mejoran la estructura del suelo, la aireación y la disponibilidad de nutrientes.",
        recomendacion=(
            "No requieren control. Favorézcalas incorporando materia orgánica y "
            "evitando la labranza excesiva."
        ),
    ),
    "Earwigs": Especie(
        nombre_es="Tijeretas",
        emoji="✂️",
        nivel=Nivel.VIGILANCIA,
        impacto=(
            "Ambivalentes: depredan pulgones, pero en altas poblaciones roen "
            "brotes tiernos, flores y frutos blandos."
        ),
        recomendacion=(
            "Monitoree con trampas de cartón enrollado o macetas con paja. "
            "Controle solo si el daño en brotes es visible."
        ),
    ),
    "Grasshoppers": Especie(
        nombre_es="Saltamontes",
        emoji="🦗",
        nivel=Nivel.CONTROL,
        impacto="Defoliación rápida y extensa; en brotes poblacionales arrasan lotes enteros.",
        recomendacion=(
            "Mallas antiinsecto en cultivos pequeños y aplicaciones de Nosema "
            "locustae o Metarhizium en los focos detectados."
        ),
    ),
    "Moths": Especie(
        nombre_es="Polillas",
        emoji="🦋",
        nivel=Nivel.CONTROL,
        impacto="Los adultos no dañan, pero cada hembra deja posturas que derivan en orugas.",
        recomendacion=(
            "Instale trampas de feromonas para medir el vuelo y aplique extracto "
            "de neem o ajo antes de que eclosionen los huevos."
        ),
    ),
    "Slugs": Especie(
        nombre_es="Babosas",
        emoji="🐌",
        nivel=Nivel.CONTROL,
        impacto="Perforan hojas y plántulas durante la noche; graves en suelos húmedos.",
        recomendacion=(
            "Cebos de fosfato férrico, trampas de cerveza y riego por la mañana "
            "para que el suelo llegue seco a la noche."
        ),
    ),
    "Snails": Especie(
        nombre_es="Caracoles",
        emoji="🐌",
        nivel=Nivel.CONTROL,
        impacto="Consumen hojas tiernas y plántulas, sobre todo tras lluvias.",
        recomendacion=(
            "Barreras de cáscara de huevo triturada o ceniza alrededor del "
            "tallo, más recolección manual nocturna."
        ),
    ),
    "Wasps": Especie(
        nombre_es="Avispas",
        emoji="🐝",
        nivel=Nivel.VIGILANCIA,
        impacto=(
            "Muchas especies son depredadoras de orugas y ayudan al cultivo, "
            "pero los nidos cercanos son un riesgo para el personal."
        ),
        recomendacion=(
            "No destruya nidos sin necesidad. Si hay riesgo de picaduras, "
            "reubique el nido con personal capacitado y use trampas cebadas."
        ),
    ),
    "Weevils": Especie(
        nombre_es="Gorgojos",
        emoji="🪲",
        nivel=Nivel.CONTROL,
        impacto="Dañan raíces, brotes y grano almacenado; sus larvas actúan bajo tierra.",
        recomendacion=(
            "Controle la humedad del almacenamiento, use trampas de feromonas "
            "y rote cultivos para cortar el ciclo de las larvas."
        ),
    ),
}


ESPECIE_DESCONOCIDA = Especie(
    nombre_es="Especie no catalogada",
    emoji="❓",
    nivel=Nivel.VIGILANCIA,
    impacto="El modelo detectó una clase que aún no tiene ficha de manejo.",
    recomendacion="Consulte con un ingeniero agrónomo antes de aplicar cualquier tratamiento.",
)


def obtener(nombre_clase: str) -> Especie:
    """Devuelve la ficha de la especie, con un fallback seguro si no existe."""
    return CATALOGO.get(nombre_clase, ESPECIE_DESCONOCIDA)
