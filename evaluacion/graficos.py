"""Gráficos y hoja de contacto a partir de los resultados de `evaluar.py`.

Se separa de la evaluación para poder re-renderizar los artefactos visuales sin
repetir la inferencia sobre las 546 imágenes.

    python evaluacion/graficos.py                      # matriz + F1 por clase
    python evaluacion/graficos.py --data <data.yaml>   # + hoja de contacto

La hoja de contacto necesita el dataset porque vuelve a anotar las imágenes de
ejemplo; los dos gráficos salen del CSV cacheado.

Las imágenes se escriben en `docs/assets/` —no en `evaluacion/resultados/`—
porque las consume la landing page publicada con GitHub Pages, que sirve `docs/`
como raíz del sitio y no puede referenciar rutas fuera de esa carpeta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre en CI o headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import detector, pests  # noqa: E402

DATOS = Path(__file__).parent / "resultados"
SALIDA = RAIZ / "docs" / "assets"
SIN_DETECCION = "(sin detección)"

# Paleta: rampa secuencial de un solo tono (claro -> oscuro) para magnitud, y
# tinta recesiva para ejes y grillas. Un solo color por rol, sin arcoíris.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
TINTA_MUTE = "#898781"
AZUL = "#2a78d6"
RAMPA = LinearSegmentedColormap.from_list(
    "azul_secuencial",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"],
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SEC,
        "xtick.color": TINTA_MUTE,
        "ytick.color": TINTA_MUTE,
    }
)


def _es(clase: str) -> str:
    return SIN_DETECCION if clase == SIN_DETECCION else pests.obtener(clase).nombre_es


def matriz_confusion(df: pd.DataFrame) -> Path:
    """Heatmap real vs. predicho, normalizado por fila (% de cada especie)."""
    clases = sorted(df["real"].unique())
    columnas = clases + [SIN_DETECCION]

    conteo = np.array(
        [[len(df[(df["real"] == r) & (df["predicho"] == c)]) for c in columnas] for r in clases],
        dtype=float,
    )
    porcentaje = conteo / conteo.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.imshow(porcentaje, cmap=RAMPA, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(columnas)), [_es(c) for c in columnas], rotation=45, ha="right")
    ax.set_yticks(range(len(clases)), [_es(c) for c in clases])
    ax.set_xlabel("Predicción del modelo", labelpad=12)
    ax.set_ylabel("Especie real", labelpad=12)
    ax.set_title(
        "Matriz de confusión — 546 imágenes de test",
        loc="left", pad=18, fontsize=14, fontweight="bold", color=TINTA,
    )

    # Separación de 2 px del color de fondo entre celdas: los bloques se leen
    # como marcas independientes, no como una mancha continua.
    ax.set_xticks(np.arange(-0.5, len(columnas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clases), 1), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)

    for i in range(len(clases)):
        for j in range(len(columnas)):
            if not conteo[i, j]:
                continue
            # Tinta clara sobre celdas oscuras, oscura sobre celdas claras.
            ax.text(
                j, i, f"{porcentaje[i, j]:.0f}",
                ha="center", va="center", fontsize=9,
                color="#ffffff" if porcentaje[i, j] > 45 else TINTA_SEC,
                fontweight="bold" if i == j else "normal",
            )

    fig.text(
        0.01, 0.02,
        "Valores en % de las imágenes de cada especie. La diagonal es el acierto.",
        fontsize=9, color=TINTA_MUTE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    destino = SALIDA / "matriz_confusion.png"
    fig.savefig(destino, dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)
    return destino


def barras_f1(metricas: pd.DataFrame) -> Path:
    """F1 por especie, ordenado. Serie única: sin leyenda, con etiqueta directa."""
    datos = metricas.sort_values("f1")

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(datos))
    ax.barh(y, datos["f1"], height=0.62, color=AZUL, zorder=3)

    ax.set_yticks(y, datos["especie"])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,50", "0,75", "1"])
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)

    for yi, (f1, aciertos, total) in enumerate(
        zip(datos["f1"], datos["aciertos"], datos["imagenes"])
    ):
        ax.text(f1 + 0.015, yi, f"{f1:.2f}".replace(".", ","), va="center",
                fontsize=9, color=TINTA_SEC, fontweight="bold")
        ax.text(0.012, yi, f"{aciertos}/{total}", va="center", fontsize=8, color="#ffffff")

    ax.set_title(
        "F1 por especie", loc="left", pad=16, fontsize=14, fontweight="bold", color=TINTA
    )
    fig.text(
        0.01, 0.02,
        "Media armónica de precisión y recall. Dentro de cada barra: aciertos sobre imágenes evaluadas.",
        fontsize=9, color=TINTA_MUTE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    destino = SALIDA / "f1_por_clase.png"
    fig.savefig(destino, dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)
    return destino


def hoja_contacto(df: pd.DataFrame, data_yaml: Path, ruta_modelo: Path | None) -> Path:
    """Un acierto representativo por especie más los errores más ilustrativos."""
    import yaml

    datos = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dir_imagenes = Path(datos["test"])

    aciertos = df[df["real"] == df["predicho"]]
    elegidas: list[tuple[str, str, bool]] = []

    for clase in sorted(df["real"].unique()):
        candidatas = aciertos[aciertos["real"] == clase].sort_values("confianza")
        if candidatas.empty:
            continue
        # La mediana de confianza: un caso típico, ni el mejor ni el peor.
        fila = candidatas.iloc[len(candidatas) // 2]
        elegidas.append(
            (fila["archivo"], f"{_es(clase)} · {fila['confianza']:.0%}", True)
        )

    fallos = df[(df["real"] != df["predicho"]) & (df["predicho"] != SIN_DETECCION)]
    for _, fila in fallos.sort_values("confianza", ascending=False).head(4).iterrows():
        elegidas.append(
            (
                fila["archivo"],
                f"{_es(fila['real'])} → {_es(fila['predicho'])} · {fila['confianza']:.0%}",
                False,
            )
        )

    modelo = detector.cargar_modelo(ruta_modelo)
    celda, barra, columnas = 300, 30, 4
    filas = -(-len(elegidas) // columnas)
    lienzo = Image.new("RGB", (columnas * celda, filas * (celda + barra)), SUPERFICIE)
    dibujo = ImageDraw.Draw(lienzo)
    try:
        fuente = ImageFont.truetype("segoeui.ttf", 15)
    except OSError:
        fuente = ImageFont.load_default()

    for i, (archivo, etiqueta, ok) in enumerate(elegidas):
        resultado = detector.detectar(
            modelo, Image.open(dir_imagenes / archivo).convert("RGB")
        )
        x, y = (i % columnas) * celda, (i // columnas) * (celda + barra)
        lienzo.paste(resultado.imagen_anotada.resize((celda, celda)), (x, y))
        dibujo.rectangle(
            [x, y + celda, x + celda, y + celda + barra],
            fill="#0ca30c" if ok else "#d03b3b",  # verde/rojo de estado
        )
        dibujo.text((x + 8, y + celda + 7), etiqueta, fill="#ffffff", font=fuente)

    destino = SALIDA / "muestras.jpg"
    lienzo.save(destino, quality=88)
    return destino


def ejemplos_diagnostico(
    df: pd.DataFrame, data_yaml: Path, ruta_modelo: Path | None
) -> list[Path]:
    """Una imagen anotada por nivel de riesgo, para las fichas de la landing.

    Se eligen tres casos —benéfica, vigilancia y control— porque el argumento
    del producto no es "detecta bichos" sino "distingue qué hacer con cada uno".
    """
    import yaml

    dir_imagenes = Path(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["test"])
    modelo = detector.cargar_modelo(ruta_modelo)
    aciertos = df[df["real"] == df["predicho"]]
    generadas = []

    for clase in ("Bees", "Ants", "Caterpillars"):
        candidatas = aciertos[aciertos["real"] == clase].sort_values(
            "confianza", ascending=False
        )
        if candidatas.empty:
            continue
        archivo = candidatas.iloc[0]["archivo"]
        resultado = detector.detectar(
            modelo, Image.open(dir_imagenes / archivo).convert("RGB")
        )
        destino = SALIDA / f"ejemplo-{clase.lower()}.jpg"
        resultado.imagen_anotada.resize((520, 520)).save(destino, quality=88)
        generadas.append(destino)

    return generadas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="data.yaml (para la hoja de contacto)")
    parser.add_argument("--modelo", type=Path, default=None)
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATOS / "predicciones.csv")
    metricas = pd.read_csv(DATOS / "metricas_por_clase.csv")

    print("Generado:", matriz_confusion(df))
    print("Generado:", barras_f1(metricas))
    if args.data:
        print("Generado:", hoja_contacto(df, args.data, args.modelo))
        for ruta in ejemplos_diagnostico(df, args.data, args.modelo):
            print("Generado:", ruta)


if __name__ == "__main__":
    main()
