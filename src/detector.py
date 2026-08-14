"""Carga del modelo YOLO y ejecución de inferencia.

Este módulo no depende de Streamlit: recibe y devuelve tipos estándar para
poder reutilizarse desde un script, un test o una API.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src import config


class ModeloNoDisponibleError(RuntimeError):
    """El archivo de pesos no existe o no se pudo cargar."""


@dataclass(frozen=True)
class Deteccion:
    especie: str
    confianza: float
    caja: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Resultado:
    detecciones: list[Deteccion]
    imagen_anotada: Image.Image

    @property
    def total(self) -> int:
        return len(self.detecciones)

    @property
    def especies(self) -> list[str]:
        """Especies únicas, ordenadas por su detección de mayor confianza."""
        mejor: dict[str, float] = {}
        for d in self.detecciones:
            mejor[d.especie] = max(mejor.get(d.especie, 0.0), d.confianza)
        return sorted(mejor, key=lambda e: mejor[e], reverse=True)


def cargar_modelo(ruta: Path | None = None):
    """Carga los pesos YOLO desde disco.

    Raises:
        ModeloNoDisponibleError: si el archivo no existe o está corrupto.
    """
    ruta = ruta or config.resolver_modelo()

    if not ruta.exists():
        raise ModeloNoDisponibleError(
            f"No se encontró el modelo en '{ruta}'. Deje el archivo de pesos en "
            "la raíz del proyecto o en 'models/', o defina PEST_MODEL_PATH."
        )

    try:
        # Import diferido: ultralytics arrastra torch y tarda varios segundos.
        from ultralytics import YOLO
    except ImportError as exc:
        # Se incluye el error original: en un despliegue lo que falla casi
        # siempre es una librería del sistema (libGL, que necesita OpenCV), no
        # un paquete de Python, y el mensaje genérico manda a revisar el lado
        # equivocado.
        raise ModeloNoDisponibleError(
            f"No se pudieron cargar las dependencias de inferencia: {exc}. "
            "Si falta una librería del sistema, declárela en 'packages.txt'; "
            "si falta un paquete de Python, instálelo con "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        return YOLO(str(ruta))
    except Exception as exc:  # noqa: BLE001 - se re-expone con contexto útil
        raise ModeloNoDisponibleError(f"No se pudo cargar el modelo: {exc}") from exc


def detectar(
    modelo,
    imagen: Image.Image,
    conf: float = config.DEFAULT_CONF,
    iou: float = config.DEFAULT_IOU,
    imgsz: int = config.IMAGE_SIZE,
) -> Resultado:
    """Ejecuta la detección sobre una imagen PIL ya cargada en memoria."""
    salida = modelo.predict(source=imagen, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = salida[0]

    detecciones = [
        Deteccion(
            especie=r.names[int(box.cls[0])],
            confianza=float(box.conf[0]),
            caja=tuple(float(v) for v in box.xyxy[0].tolist()),
        )
        for box in (r.boxes if r.boxes is not None else [])
    ]

    # r.plot() devuelve un array BGR (convención de OpenCV): hay que invertir
    # los canales antes de mostrarlo o los colores salen cambiados.
    anotada = Image.fromarray(r.plot()[:, :, ::-1])

    return Resultado(detecciones=detecciones, imagen_anotada=anotada)
