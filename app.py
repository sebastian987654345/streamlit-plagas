import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Detector de Plagas Inteligente 🐛", layout="centered")
st.title("🌿 Detección Inteligente de Plagas en Cultivos")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("best6.pt")  # Asegúrate de que 'best.pt' esté en la carpeta del proyecto

model = load_model()

# Subida de imagen
uploaded_file = st.file_uploader("📤 Sube una imagen del cultivo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar temporalmente la imagen
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Slider de confianza en porcentaje (0 a 100)
    conf_percent = st.slider("Nivel de confianza mínima (%)", 0, 100, 30, 5)
    conf = conf_percent / 100  # Convertir a decimal para YOLO

    # Botón para detectar
    if st.button("🔍 Detectar plagas"):
        with st.spinner("Analizando imagen... ⏳"):
            results = model.predict(source=temp_path, conf=conf, imgsz=640)
        
        detecciones = 0
        for r in results:
            # Crear columnas: izquierda imagen original, derecha imagen con detecciones
            col1, col2 = st.columns(2)

            # Imagen original
            with col1:
                st.image(Image.open(temp_path), caption="🖼️ Imagen original", use_column_width=True)
            
            # Imagen con detecciones
            with col2:
                if r.boxes is not None and len(r.boxes) > 0:
                    annotated_img = r.plot()
                    st.image(annotated_img, caption="🧩 Detección", use_column_width=True)
                else:
                    st.warning("⚠️ No se detectaron plagas en la imagen.")

            # Mostrar información de cada detección
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    plaga = r.names[cls_id]
                    detecciones += 1
                    st.success(f"🐞 {plaga} detectada con {conf_score*100:.1f}% de confianza")

                    # 🔎 Recomendaciones automáticas
                    if plaga == "Ants":
                        st.info("🐜 Usa barreras naturales de limón o vinagre alrededor de las plantas.")
                    elif plaga == "Bees":
                        st.info("🐝 No eliminar, las abejas son beneficiosas para la polinización.")
                    elif plaga == "Wasps":
                        st.warning("⚠️ Usa trampas y evita manipular nidos directamente.")
                    elif plaga == "Moths":
                        st.info("🦋 Aplica extractos naturales de ajo o neem.")
                    elif plaga == "Snails":
                        st.info("🐌 Coloca cáscaras de huevo trituradas alrededor del tallo.")
                    elif plaga == "Weevils":
                        st.warning("🪲 Controla la humedad y usa trampas de feromonas.")
            
        if detecciones == 0:
            st.warning("⚠️ No se detectaron plagas en la imagen. Intenta con otra o baja el nivel de confianza.")
