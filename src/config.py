"""Configuración central de la aplicación."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Ubicaciones donde se busca el checkpoint, en orden de preferencia. Permite
# mantener los pesos en 'models/' sin romper despliegues que los dejan en la
# raíz del proyecto.
RUTAS_MODELO = (
    BASE_DIR / "models" / "best6.pt",
    BASE_DIR / "best6.pt",
)


def resolver_modelo() -> Path:
    """Ruta del modelo a cargar.

    La variable de entorno PEST_MODEL_PATH tiene prioridad; si no está
    definida se usa la primera ubicación conocida que exista.
    """
    if (ruta_env := os.getenv("PEST_MODEL_PATH")) is not None:
        return Path(ruta_env)

    return next((r for r in RUTAS_MODELO if r.exists()), RUTAS_MODELO[0])


# Parámetros de inferencia
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50
IMAGE_SIZE = 640

# Formatos aceptados en el cargador de imágenes
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp"]

# Tamaño máximo tolerado del archivo subido (MB)
MAX_UPLOAD_MB = 20
