import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ==============================================================================
CSV_PATH = 'modelos_tfg/datos_10min_modelos.csv'
RUTA_LSTM = 'modelos_tfg/exportacion_micro/LSTM_int8.tflite'
RUTA_GRU  = 'modelos_tfg/exportacion_micro/GRU_int8.tflite'

SEQ_LENGTH = 18
LOOK_AHEAD = 6
FEATURES = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv']

# ==============================================================================
# 2. CARGA Y PREPROCESADO EXACTO AL ENTRENAMIENTO
# ==============================================================================
print(f"[*] Cargando dataset original: {CSV_PATH}...")
data = pd.read_csv(CSV_PATH, sep=';', decimal=',')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)

# Limpieza y variables cíclicas
data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True, errors='ignore')
horas = data.index.hour
meses = data.index.month
data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
data['mes_sin']  = np.sin(meses * (2 * np.pi / 12))
data['mes_cos']  = np.cos(meses * (2 * np.pi / 12))

data_selected = data[FEATURES]

# ==============================================================================
# 3. SPLIT 80/20 Y ESCALADO
# ==============================================================================
train_split = int(0.8 * len(data_selected))
train_df = data_selected.iloc[:train_split]
val_df   = data_selected.iloc[train_split:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Ajustamos solo con Train, transformamos ambos
val_X = scaler_X.fit(train_df.drop(columns=['Pot_inv'])).transform(val_df.drop(columns=['Pot_inv']))
val_y = scaler_y.fit(train_df[['Pot_inv']]).transform(val_df[['Pot_inv']]).flatten()

# ==============================================================================
# 4. CREACIÓN DE VENTANAS DESLIZANTES PARA VALIDACIÓN
# ==============================================================================
def make_sequences(X, y, timestamps, seq_length, look_ahead):
    Xs, ys, ts_y = [], [], []
    limite = len(X) - seq_length - look_ahead + 1
    tiempo_esperado = pd.Timedelta(minutes=10 * (seq_length + look_ahead - 1))

    for i in range(limite):
        t_inicio = timestamps[i]
        t_fin = timestamps[i + seq_length + look_ahead - 1]
        if (t_fin - t_inicio) == tiempo_esperado:
            Xs.append(X[i : i + seq_length])
            ys.append(y[i + seq_length + look_ahead - 1])
            ts_y.append(t_fin)
            
    return np.array(Xs, dtype=np.float32), np.array(ys), ts_y

print("[*] Generando secuencias temporales para el 20% de validación...")
X_val_seq, y_val_seq, ts_val = make_sequences(val_X, val_y, val_df.index, SEQ_LENGTH, LOOK_AHEAD)

# Desescalar los valores reales para tener la referencia en W
y_val_real = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).flatten()

# ==============================================================================
# 5. INFERENCIA CON TFLITE CUANTIZADO
# ==============================================================================
def predecir_tflite(modelo_path, X_input):
    interp = tf.lite.Interpreter(model_path=modelo_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]['index']
    out = interp.get_output_details()[0]['index']
    
    preds = []
    for seq in X_input:
        interp.set_tensor(inp, np.expand_dims(seq, axis=0))
        interp.invoke()
        pred_norm = interp.get_tensor(out)[0][0]
        preds.append(pred_norm)
        
    preds_w = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return np.maximum(preds_w, 0) # Recorte a 0W mínimo

print("[*] Ejecutando modelo LSTM INT8...")
preds_lstm = predecir_tflite(RUTA_LSTM, X_val_seq)

print("[*] Ejecutando modelo GRU INT8...")
preds_gru = predecir_tflite(RUTA_GRU, X_val_seq)

# ==============================================================================
# 6. MÉTRICAS Y VISUALIZACIÓN
# ==============================================================================
print("\n" + "="*65)
print(f" RESULTADOS FINALES DE VALIDACIÓN (INT8) - {len(y_val_real)} secuencias")
print("="*65)
print(f" LSTM  -> MAE: {mean_absolute_error(y_val_real, preds_lstm):.1f} W  | R²: {r2_score(y_val_real, preds_lstm):.4f}")
print(f" GRU   -> MAE: {mean_absolute_error(y_val_real, preds_gru):.1f} W  | R²: {r2_score(y_val_real, preds_gru):.4f}")
print("="*65)

# Visualizar un tramo bonito (ej: 4 días despejados/nublados)
INICIO = 280
FIN = 850 

plt.figure(figsize=(16, 6))
plt.plot(ts_val[INICIO:FIN], y_val_real[INICIO:FIN], label='Real Medida', color='black', linewidth=3, zorder=3)
plt.plot(ts_val[INICIO:FIN], preds_lstm[INICIO:FIN], label='LSTM INT8 (Ganador)', color='blue', linewidth=2, zorder=2)
plt.plot(ts_val[INICIO:FIN], preds_gru[INICIO:FIN], label='GRU INT8', color='red', linewidth=2, linestyle='--', zorder=1)

plt.title('Evaluación en Set de Validación: LSTM vs GRU (TFLite Hardware)', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=13)
plt.ylabel('Potencia Generada (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()