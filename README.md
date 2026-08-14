# 🌿 Detector Inteligente de Plagas Agrícolas

Aplicación web que identifica plagas en fotografías de cultivos mediante visión por
computadora (YOLO) y devuelve una recomendación de manejo integrado para cada
especie detectada.

No se limita a marcar insectos: distingue **plagas** de **especies benéficas** —una
abeja o una lombriz no deben tratarse como amenaza— y prioriza la respuesta según
el nivel de riesgo agronómico.

🌐 **[Página de presentación](https://sebastian987654345.github.io/streamlit-plagas/)** ·
📊 **[Informe de validación](evaluacion/RESULTADOS.md)**

---

## Características

| | |
|---|---|
| 🎯 **Detección por individuo** | Localiza cada espécimen con su caja delimitadora y nivel de confianza. |
| 🌱 **Criterio agronómico** | Clasifica en benéfica, vigilancia o control, en lugar de tratar todo como plaga. |
| 📋 **Recomendación accionable** | Ficha de impacto y manejo integrado por especie, no un simple nombre. |
| 🎚️ **Umbrales ajustables** | Confianza e IoU configurables en vivo, con la inferencia cacheada. |
| 📤 **Resultados exportables** | Descarga de la imagen anotada (PNG) y del detalle de detecciones (CSV). |
| ⚡ **Sin instalación para el usuario** | Corre en el navegador; el modelo se carga una sola vez por sesión. |

## Especies detectadas

El modelo reconoce **12 clases**:

| Especie | Clasificación | Especie | Clasificación |
|---|---|---|---|
| 🐜 Hormigas | Vigilancia | 🦗 Saltamontes | Control |
| 🐝 Abejas | Benéfica | 🦋 Polillas | Control |
| 🪲 Escarabajos | Control | 🐌 Babosas | Control |
| 🐛 Orugas | Control | 🐌 Caracoles | Control |
| 🪱 Lombrices | Benéfica | 🐝 Avispas | Vigilancia |
| ✂️ Tijeretas | Vigilancia | 🪲 Gorgojos | Control |

## Validación del modelo

Medido sobre las **546 imágenes del conjunto de test** —imágenes que el modelo
nunca vio durante el entrenamiento— con los mismos parámetros que usa la
aplicación (confianza 0,30 · IoU 0,50 · 640 px).

| Métrica | Valor |
|---|---|
| Especie correcta (top-1) | **455/546 · 83,3 %** |
| F1 macro (promedio de las 12 clases) | **0,86** |
| Confianza media en los aciertos | 0,69 |
| Imágenes sin ninguna detección | 39 · 7,1 % |
| Tiempo por imagen (CPU) | ~206 ms |

Seis de las doce especies superan **F1 0,89**; las mejores son caracoles (0,97) y
avispas (0,97). El detalle por clase, la matriz de confusión y el análisis de los
errores están en **[evaluacion/RESULTADOS.md](evaluacion/RESULTADOS.md)**.

Un caso típico de cada especie y los cuatro errores más marcados de la prueba
(verde: acierto · rojo: confusión):

![Muestras analizadas](docs/assets/muestras.jpg)

![Matriz de confusión](docs/assets/matriz_confusion.png)

### Reproducir la evaluación

```bash
python evaluacion/evaluar.py --data ruta/al/data-set/data.yaml
python evaluacion/graficos.py --data ruta/al/data-set/data.yaml
```

El dataset no se versiona (pesa ~1,5 GB), por eso se pasa por parámetro. Las
predicciones quedan cacheadas en `evaluacion/resultados/predicciones.csv`, de
modo que regenerar métricas o gráficos no repite la inferencia.

## Estructura del proyecto

```
streamlit-plagas/
├── app.py                  # Interfaz Streamlit (capa de presentación)
├── best6.pt                # Pesos entrenados (YOLO, 12 clases)
├── src/
│   ├── config.py           # Rutas y parámetros de inferencia
│   ├── detector.py         # Carga del modelo y ejecución de YOLO
│   └── pests.py            # Catálogo de especies y recomendaciones
├── evaluacion/
│   ├── evaluar.py          # Métricas sobre el conjunto de test
│   ├── graficos.py         # Matriz de confusión, F1 y hoja de contacto
│   ├── RESULTADOS.md       # Informe de validación
│   └── resultados/         # Predicciones y métricas (CSV/JSON)
├── tests/
│   └── test_pests.py       # Verifica catálogo ↔ clases del modelo
├── docs/
│   ├── index.html          # Landing page (GitHub Pages)
│   └── assets/             # Gráficos de la evaluación
├── .streamlit/config.toml  # Tema y configuración del servidor
├── packages.txt            # Librerías del sistema (libGL, para OpenCV)
└── requirements.txt
```

La lógica de dominio (`src/`) no depende de Streamlit: el detector y el catálogo
pueden reutilizarse desde un script, un test o una API sin tocar la UI.

## Puesta en marcha

```bash
git clone https://github.com/sebastian987654345/streamlit-plagas.git
cd streamlit-plagas

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

### Usar otro modelo

Los pesos se buscan en `models/best6.pt` y, si no está, en `best6.pt` (raíz del
proyecto). Para apuntar a otro checkpoint, defina `PEST_MODEL_PATH`:

```bash
PEST_MODEL_PATH=/ruta/a/otro_modelo.pt streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

La suite valida que toda clase que el modelo puede predecir tenga su ficha de
manejo correspondiente, de modo que agregar clases al modelo sin documentarlas
haga fallar el build.

## Despliegue

**Streamlit Community Cloud** (gratuito):

1. Conecte el repositorio en [share.streamlit.io](https://share.streamlit.io).
2. Archivo principal: `app.py`.
3. Las dependencias se instalan solas desde `requirements.txt` y `packages.txt`.

`packages.txt` es necesario: Ultralytics depende de `opencv-python`, que exige
`libGL` a nivel de sistema operativo. Sin ese archivo el contenedor arranca pero
falla al importar el modelo.

**Landing page**: en *Settings → Pages* del repositorio, elija la rama `main` y la
carpeta `/docs`. La página queda publicada en
`https://sebastian987654345.github.io/streamlit-plagas/`.

## Notas técnicas

- La imagen subida se procesa **en memoria**; no se escribe nada a disco ni se
  almacena la fotografía del usuario.
- La inferencia está cacheada por `(imagen, confianza, IoU)`: ajustar un umbral
  no reprocesa un resultado ya calculado.
- El modelo se carga una única vez por sesión mediante `st.cache_resource`.

## Licencia

El código de esta aplicación es de uso libre. Tenga en cuenta que
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) se distribuye bajo
**AGPL-3.0**: un despliegue comercial cerrado requiere adquirir una licencia
comercial de Ultralytics.
