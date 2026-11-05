import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Detección de Dígitos Escritos a Mano",
    page_icon="✍️",
    layout="wide",
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; color: #F8FAFC; }
    .stSidebar { background-color: #161B22 !important; }
    .hero-title { color: #FFFFFF; font-weight: 800; font-size: 2.2rem; }
    .status-box {
        background-color: #161B22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-bottom: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- FUNCION PARA CREAR/CARGAR UN MODELO MNIST SIMPLE ---
@st.cache_resource
def get_mnist_model():
  model_path = "mnist_model.h5"
  if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    return model, f"Modelo cargado exitosamente desde: {model_path}"
  else:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=3, verbose=0)
    model.save(model_path)
    return model, f"Modelo entrenado y guardado en: {model_path}"


# --- BARRA LATERAL ---
st.sidebar.markdown("### ⚙️ Entorno")
st.sidebar.markdown(f"**TensorFlow:** {tf.__version__}")
st.sidebar.markdown(f"**Keras:** {tf.keras.__version__}")
st.sidebar.markdown("---")
st.sidebar.info("Dibuja un número del 0 al 9 en el lienzo central.")

# --- CONTENIDO PRINCIPAL ---
st.markdown(
    '<p class="hero-title">✍️ Detección de Dígitos Escritos a Mano</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "Modelo CNN/DNN entrenado con el dataset MNIST para reconocimiento en"
    " tiempo real."
)
st.markdown("---")

# Cargar modelo
with st.spinner("Intentando cargar el modelo..."):
  model, status_msg = get_mnist_model()

st.markdown(
    f"""
    <div class="status-box">
        📂 Intentando cargar: <b>mnist_model.h5</b><br>
        <span style="color: #10B981;">✅ {status_msg}</span>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown("### 🎨 Dibuja un dígito del 0 al 9")

col_canvas, col_controls = st.columns([1.2, 1])

with col_canvas:
  # Lienzo con update_streamlit=True para evitar el error de lectura de datos
  canvas_result = st_canvas(
      fill_color="#000000",
      stroke_width=18,
      stroke_color="#FFFFFF",
      background_color="#000000",
      width=280,
      height=280,
      drawing_mode="freedraw",
      update_streamlit=True,
      key="canvas",
  )

with col_controls:
  st.markdown("### 🎮 Controles")
  predecir_btn = st.button("🔍 Predecir")

# --- PROCESAMIENTO Y PREDICCIÓN ---
if predecir_btn:
  if canvas_result.image_data is not None:
    img_array = canvas_result.image_data.astype("uint8")
    img_pil = Image.fromarray(img_array).convert("L")
    img_resized = img_pil.resize((28, 28))

    img_normalized = np.array(img_resized) / 255.0
    img_reshaped = img_normalized.reshape(1, 28, 28)

    predictions = model.predict(img_reshaped)
    digito_predicho = np.argmax(predictions[0])
    confianza = np.max(predictions[0]) * 100

    st.markdown("---")
    st.markdown("### 📊 Resultado de la Predicción")
    st.success(
        f"🎯 **Dígito Detectado:** {digito_predicho} (Confianza:"
        f" {confianza:.2f}%)"
    )
  else:
    st.warning("Por favor dibuja un dígito en el lienzo antes de predecir.")
