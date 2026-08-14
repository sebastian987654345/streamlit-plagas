"""Detector de plagas agrícolas — interfaz Streamlit."""

import io

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src import config, pests
from src.detector import ModeloNoDisponibleError, Resultado, cargar_modelo, detectar

st.set_page_config(
    page_title="Detector de Plagas Agrícolas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ESTILO_NIVEL = {
    pests.Nivel.BENEFICO: ("#1a7f4b", "Especie benéfica"),
    pests.Nivel.VIGILANCIA: ("#b27300", "Requiere vigilancia"),
    pests.Nivel.CONTROL: ("#c0392b", "Requiere control"),
}


@st.cache_resource(show_spinner="Cargando modelo de detección…")
def _modelo():
    return cargar_modelo()


@st.cache_data(show_spinner=False)
def _detectar_cacheado(_modelo_yolo, imagen_bytes: bytes, conf: float, iou: float) -> Resultado:
    """Cachea la inferencia por (imagen, umbrales) para que mover un slider
    no reprocese lo ya calculado."""
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    return detectar(_modelo_yolo, imagen, conf=conf, iou=iou)


def _a_png(imagen: Image.Image) -> bytes:
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


def barra_lateral() -> tuple[float, float]:
    with st.sidebar:
        st.header("⚙️ Parámetros de análisis")

        conf = st.slider(
            "Confianza mínima",
            min_value=0.05,
            max_value=0.95,
            value=config.DEFAULT_CONF,
            step=0.05,
            help="Umbral por debajo del cual se descartan las detecciones. "
            "Bajalo para encontrar plagas pequeñas; subilo para reducir falsos positivos.",
        )
        iou = st.slider(
            "Solapamiento máximo (IoU)",
            min_value=0.10,
            max_value=0.90,
            value=config.DEFAULT_IOU,
            step=0.05,
            help="Controla cuánto pueden solaparse dos cajas antes de considerarse "
            "la misma detección.",
        )

        st.divider()
        with st.expander("🧠 Sobre el modelo"):
            st.markdown(
                f"""
                **Arquitectura:** YOLO (Ultralytics)
                **Resolución de entrada:** {config.IMAGE_SIZE}×{config.IMAGE_SIZE} px
                **Clases entrenadas:** {len(pests.CATALOGO)}
                """
            )
            st.caption(
                " · ".join(
                    f"{e.emoji} {e.nombre_es}" for e in pests.CATALOGO.values()
                )
            )

        return conf, iou


def elegir_imagen() -> tuple[bytes, str] | None:
    """Imagen a analizar, desde archivo o desde la cámara del dispositivo.

    Devuelve los bytes y un nombre base para los archivos descargables, o None
    mientras no haya imagen. Se usa un selector en vez de mostrar los dos
    widgets a la vez para que nunca haya dos imágenes candidatas en pantalla.
    """
    origen = st.radio(
        "¿De dónde viene la imagen?",
        ("📁 Subir una foto", "📷 Tomar una foto ahora"),
        horizontal=True,
    )

    if origen.startswith("📁"):
        archivo = st.file_uploader(
            "Imagen del cultivo",
            type=config.SUPPORTED_FORMATS,
            help=f"Formatos: {', '.join(config.SUPPORTED_FORMATS).upper()} · "
            f"Máx. {config.MAX_UPLOAD_MB} MB",
        )
        if archivo is None:
            return None
        return archivo.getvalue(), archivo.name.rsplit(".", 1)[0]

    st.caption(
        "El navegador va a pedir permiso para usar la cámara. En el celular "
        "conviene acercarse al insecto hasta que ocupe buena parte del encuadre."
    )
    # camera_input entrega siempre un PNG y no trae nombre de archivo.
    foto = st.camera_input("Apunte al insecto y tome la foto")
    return (foto.getvalue(), "captura") if foto is not None else None


def estado_inicial() -> None:
    st.info(
        "Suba una fotografía del cultivo o tómela con la cámara para comenzar. "
        "El sistema identifica la especie, marca su ubicación en la imagen y "
        "entrega una recomendación de manejo integrado."
    )

    cols = st.columns(4)
    tarjetas = [
        ("🎯", "Detección precisa", "Localiza cada individuo con su nivel de confianza."),
        ("🌱", "Criterio agronómico", "Distingue plagas de especies benéficas."),
        ("📋", "Plan de acción", "Recomendaciones de manejo por especie detectada."),
        ("⚡", "Resultado inmediato", "Análisis en segundos, sin instalación."),
    ]
    for col, (icono, titulo, detalle) in zip(cols, tarjetas):
        with col:
            st.markdown(f"### {icono}\n**{titulo}**")
            st.caption(detalle)


def resumen(resultado: Resultado) -> None:
    plagas = [
        e for e in resultado.especies
        if pests.obtener(e).nivel is not pests.Nivel.BENEFICO
    ]
    beneficas = [
        e for e in resultado.especies
        if pests.obtener(e).nivel is pests.Nivel.BENEFICO
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Individuos detectados", resultado.total)
    c2.metric("Especies plaga", len(plagas))
    c3.metric("Especies benéficas", len(beneficas))


def recomendaciones(resultado: Resultado) -> None:
    st.subheader("📋 Diagnóstico y recomendaciones")

    for nombre_clase in resultado.especies:
        especie = pests.obtener(nombre_clase)
        color, etiqueta = ESTILO_NIVEL[especie.nivel]
        cantidad = sum(1 for d in resultado.detecciones if d.especie == nombre_clase)

        with st.container(border=True):
            st.markdown(
                f"#### {especie.emoji} {especie.nombre_es} "
                f"<span style='color:{color};font-size:0.7em;'>· {etiqueta}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{cantidad} individuo{'s' if cantidad != 1 else ''} · clase del modelo: `{nombre_clase}`"
            )
            st.markdown(f"**Impacto:** {especie.impacto}")
            st.markdown(f"**Manejo sugerido:** {especie.recomendacion}")


def tabla_detalle(resultado: Resultado) -> None:
    with st.expander("🔬 Detalle técnico de las detecciones"):
        df = pd.DataFrame(
            [
                {
                    "#": i,
                    "Especie": pests.obtener(d.especie).nombre_es,
                    "Clase": d.especie,
                    "Confianza": round(d.confianza * 100, 1),
                    "x1": round(d.caja[0]),
                    "y1": round(d.caja[1]),
                    "x2": round(d.caja[2]),
                    "y2": round(d.caja[3]),
                }
                for i, d in enumerate(resultado.detecciones, start=1)
            ]
        )
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confianza": st.column_config.ProgressColumn(
                    "Confianza", min_value=0.0, max_value=100.0, format="%.1f%%"
                )
            },
        )
        st.download_button(
            "⬇️ Descargar detecciones (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="detecciones.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("🌿 Detección Inteligente de Plagas en Cultivos")
    st.caption(
        "Visión por computadora aplicada al manejo integrado de plagas: identifique "
        "la especie y reciba una recomendación de control en segundos."
    )

    conf, iou = barra_lateral()

    try:
        modelo = _modelo()
    except ModeloNoDisponibleError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()

    entrada = elegir_imagen()

    if entrada is None:
        estado_inicial()
        return

    imagen_bytes, nombre_base = entrada

    if len(imagen_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"La imagen supera el límite de {config.MAX_UPLOAD_MB} MB.")
        return

    try:
        original = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    except UnidentifiedImageError:
        st.error("No se pudo leer el archivo: ¿es una imagen válida?")
        return

    with st.spinner("Analizando imagen…"):
        resultado = _detectar_cacheado(modelo, imagen_bytes, conf, iou)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(
            resultado.imagen_anotada,
            caption="Detecciones del modelo",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Descargar imagen analizada",
            data=_a_png(resultado.imagen_anotada),
            file_name=f"analisis_{nombre_base}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.divider()

    if resultado.total == 0:
        st.warning(
            "No se detectaron plagas con los parámetros actuales. Pruebe bajando la "
            "confianza mínima en el panel lateral o usando una foto más cercana."
        )
        return

    resumen(resultado)
    recomendaciones(resultado)
    tabla_detalle(resultado)


if __name__ == "__main__":
    main()
