☀️ TinyML Solar Forecasting (ESP32-S3)
Implementación de Redes Neuronales Recurrentes (GRU) para la predicción de potencia fotovoltaica en el Edge.

Este repositorio contiene el desarrollo técnico de mi Trabajo de Fin de Grado. El objetivo es predecir la producción de energía solar con un horizonte de 1 hora utilizando datos en tiempo real de inversores Fronius, procesados íntegramente en un microcontrolador ESP32-S3.

🚀 Características Principales
Inteligencia Artificial en el Edge: Inferencia local sin dependencia de la nube (Cloudless AI).

Modelo GRU (Gated Recurrent Unit): Arquitectura optimizada para series temporales y bajo consumo de memoria.

Ingeniería de Características: Codificación cíclica del tiempo (Seno/Coseno) para capturar la estacionalidad solar.

Pipeline Completo: Desde la extracción de datos vía API/Modbus hasta la conversión a cabeceras de C++ para TinyML.

📂 Estructura del Proyecto
Plaintext
├── training/            # Notebooks y scripts de entrenamiento (Keras/TensorFlow)
├── model/               # Modelos exportados (.h5, .tflite, model_data.h)
├── src_esp32/           # Código fuente C++ para la placa ESP32-S3
├── data/                # Datasets de entrenamiento y validación
└── docs/                # Gráficas de rendimiento y documentación del TFG
🛠️ Stack Tecnológico
Entrenamiento: Python, TensorFlow/Keras, Scikit-learn, Pandas.

Inferencia Embebida: C++, TensorFlow Lite for Microcontrollers.

Hardware: ESP32-S3 (Dual-core, 512KB SRAM).

Visualización: Matplotlib, Seaborn.

📈 Resultados del Modelo
El modelo utiliza una ventana deslizante de 6 horas de historia para predecir el siguiente intervalo. Se incluyen análisis de:

Matriz de Correlación: Impacto de la temperatura y humedad en la producción.

Curvas de Pérdida: Visualización de la convergencia (Train vs Validation Loss).

Predicción Real: Comparativa de la potencia predicha frente a la lectura real del inversor.

🔧 Instalación y Uso rápido
Clonar el repo:

Bash
git clone https://github.com/acasado05/TinyML-for-Solar-Forecasting-using-ESP32-S3.git
Instalar dependencias:

Bash
pip install -r requirements.txt
Ejecutar entrenamiento:
Ver training/gru_training.py para generar el modelo optimizado.

✍️ Autor
Aitor Casado - Estudiante de Ingeniería Electrónica - UPM

LinkedIn: [Tu enlace aquí]