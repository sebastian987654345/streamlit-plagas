"""Evaluación del modelo sobre el conjunto de test.

Mide el mismo código que corre en producción: importa `src.detector`, así que
lo que se reporta acá es exactamente lo que devuelve la aplicación, no una
inferencia paralela con otros parámetros.

Uso:

    python evaluacion/evaluar.py --data F:/entrenamiento_cnn/data-set/data.yaml

El dataset no se versiona en el repositorio (son ~1.5 GB); se pasa por
parámetro. Los artefactos generados sí se versionan, en `resultados/`.

La inferencia se cachea en `predicciones.csv`: volver a correr el script
regenera las métricas y los gráficos sin repetir los ~5 minutos de cómputo.
Use `--forzar` para reprocesar las imágenes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

# Permite ejecutar el script directamente (`python evaluacion/evaluar.py`)
# resolviendo `src` desde la raíz del proyecto.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import config, detector, pests  # noqa: E402

SALIDA = Path(__file__).parent / "resultados"
SIN_DETECCION = "(sin detección)"


@dataclass(frozen=True)
class Prediccion:
    archivo: str
    real: str
    predicho: str
    confianza: float
    n_detecciones: int
    ms: float


def _clase_real(ruta_etiqueta: Path, nombres: dict[int, str]) -> str | None:
    """Clase mayoritaria de una anotación YOLO.

    Las imágenes del dataset son de un único espécimen, pero algunas traen
    varias cajas del mismo individuo; se toma la clase más frecuente.
    """
    if not ruta_etiqueta.exists():
        return None

    ids = [
        int(linea.split()[0])
        for linea in ruta_etiqueta.read_text().splitlines()
        if linea.strip()
    ]
    return nombres[max(set(ids), key=ids.count)] if ids else None


def _rutas_test(data_yaml: Path) -> tuple[Path, Path]:
    datos = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    imagenes = Path(datos["test"])
    if not imagenes.is_absolute():
        imagenes = (data_yaml.parent / imagenes).resolve()
    return imagenes, imagenes.parent / "labels"


def inferir(modelo, data_yaml: Path, conf: float, iou: float) -> list[Prediccion]:
    """Corre el detector sobre todas las imágenes de test."""
    dir_imagenes, dir_etiquetas = _rutas_test(data_yaml)
    imagenes = sorted(
        p for p in dir_imagenes.iterdir()
        if p.suffix.lower().lstrip(".") in config.SUPPORTED_FORMATS
    )
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en '{dir_imagenes}'.")

    predicciones: list[Prediccion] = []
    for i, ruta in enumerate(imagenes, start=1):
        real = _clase_real(dir_etiquetas / f"{ruta.stem}.txt", modelo.names)
        if real is None:  # imagen sin anotación: no aporta a la métrica
            continue

        inicio = time.perf_counter()
        resultado = detector.detectar(
            modelo, Image.open(ruta).convert("RGB"), conf=conf, iou=iou
        )
        ms = (time.perf_counter() - inicio) * 1000

        mejor = max(resultado.detecciones, key=lambda d: d.confianza, default=None)
        predicciones.append(
            Prediccion(
                archivo=ruta.name,
                real=real,
                predicho=mejor.especie if mejor else SIN_DETECCION,
                confianza=mejor.confianza if mejor else 0.0,
                n_detecciones=resultado.total,
                ms=ms,
            )
        )

        if i % 25 == 0 or i == len(imagenes):
            print(f"  {i}/{len(imagenes)} imágenes procesadas", flush=True)

    return predicciones


def metricas_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    """Precisión, recall y F1 por especie, sobre la clase de mayor confianza.

    Es una lectura de clasificación, no de detección: responde "¿acertó la
    especie?", que es la pregunta que le importa a quien usa la app. Las
    métricas de detección (mAP) se calculan aparte con `model.val`.
    """
    filas = []
    for clase in sorted(df["real"].unique()):
        tp = len(df[(df["real"] == clase) & (df["predicho"] == clase)])
        fp = len(df[(df["real"] != clase) & (df["predicho"] == clase)])
        fn = len(df[(df["real"] == clase) & (df["predicho"] != clase)])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        filas.append(
            {
                "clase": clase,
                "especie": pests.obtener(clase).nombre_es,
                "imagenes": tp + fn,
                "aciertos": tp,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "sin_deteccion": len(
                    df[(df["real"] == clase) & (df["predicho"] == SIN_DETECCION)]
                ),
                "confianza_media": round(
                    df[(df["real"] == clase) & (df["predicho"] == clase)]["confianza"].mean(), 3
                )
                if tp
                else 0.0,
            }
        )

    return pd.DataFrame(filas)


def metricas_deteccion(ruta_modelo: Path, data_yaml: Path, imgsz: int) -> dict[str, float]:
    """mAP e IoU sobre el split de test, con la métrica estándar de Ultralytics."""
    from ultralytics import YOLO

    resultados = YOLO(str(ruta_modelo)).val(
        data=str(data_yaml), split="test", imgsz=imgsz, batch=4,
        device="cpu", verbose=False, plots=False,
    )
    return {
        "map50": round(float(resultados.box.map50), 3),
        "map50_95": round(float(resultados.box.map), 3),
        "precision": round(float(resultados.box.mp), 3),
        "recall": round(float(resultados.box.mr), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Ruta al data.yaml del dataset")
    parser.add_argument("--modelo", type=Path, default=None, help="Checkpoint a evaluar")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU)
    parser.add_argument("--forzar", action="store_true", help="Reprocesa aunque exista el caché")
    parser.add_argument("--sin-map", action="store_true", help="Omite la validación de Ultralytics")
    args = parser.parse_args()

    SALIDA.mkdir(exist_ok=True)
    ruta_modelo = args.modelo or config.resolver_modelo()
    cache = SALIDA / "predicciones.csv"

    if cache.exists() and not args.forzar:
        print(f"Usando predicciones cacheadas de '{cache.name}' (--forzar para rehacerlas).")
        df = pd.read_csv(cache)
    else:
        print(f"Cargando modelo '{ruta_modelo.name}'…")
        modelo = detector.cargar_modelo(ruta_modelo)
        print(f"Infiriendo sobre el test set (conf={args.conf}, iou={args.iou})…")
        df = pd.DataFrame(inferir(modelo, args.data, args.conf, args.iou))
        df.to_csv(cache, index=False)

    por_clase = metricas_por_clase(df)
    por_clase.to_csv(SALIDA / "metricas_por_clase.csv", index=False)

    aciertos = int((df["real"] == df["predicho"]).sum())
    resumen = {
        "modelo": ruta_modelo.name,
        "imagenes": len(df),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": config.IMAGE_SIZE,
        "aciertos_top1": aciertos,
        "exactitud_top1": round(aciertos / len(df), 3),
        "sin_deteccion": int((df["predicho"] == SIN_DETECCION).sum()),
        "confianza_media_aciertos": round(
            df[df["real"] == df["predicho"]]["confianza"].mean(), 3
        ),
        "ms_por_imagen": round(df["ms"].mean(), 1),
        "f1_macro": round(por_clase["f1"].mean(), 3),
    }

    if not args.sin_map:
        print("Calculando mAP con la validación de Ultralytics…")
        resumen["deteccion"] = metricas_deteccion(ruta_modelo, args.data, config.IMAGE_SIZE)

    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + por_clase.to_string(index=False))
    print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nArtefactos escritos en '{SALIDA}'.")
    print("Genere los gráficos con: python evaluacion/graficos.py")


if __name__ == "__main__":
    main()
