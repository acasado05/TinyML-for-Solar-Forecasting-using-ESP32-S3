import numpy as np
import pandas as pd
import os
import re
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# CONFIGURACIÓN V13 — VERSIÓN DEFINITIVA DE SELECCIÓN DE MODELO
# Hiperparámetros óptimos individuales por modelo, derivados de la
# campaña de entrenamiento completa (V1-V12).
# Dataset completo con noches. Split temporal 80/20. seq_length=18.
# Métricas duales: dataset completo + solo instantes diurnos.
# =====================================================================

CONFIG = {
    'version'       : 'V13_DEFINITIVA',
    'fecha'         : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    'descripcion'   : 'Versión final de selección de modelo. Hiperparámetros '
                      'óptimos individuales por arquitectura derivados de la '
                      'campaña V1-V12. Métricas duales (completo y diurno).',
    'dataset'       : 'datos_10min_modelos.csv',
    'features'      : ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos',
                       'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv'],
    'target'        : 'Pot_inv',
    'seq_length'    : 18,       # 3 horas de contexto (18 pasos x 10 min)
    'look_ahead'    : 6,        # horizonte de predicción: 1 hora (6 pasos x 10 min)
    'train_split'   : 0.80,     # split temporal secuencial 80/20
    'n_epochs'      : 200,
    'batch_size'    : 64,
    'patience_es'   : 50,       # EarlyStopping
    'patience_lr'   : 10,       # ReduceLROnPlateau
    'lr_factor'     : 0.5,
    'lr_min'        : 1e-6,
    'umbral_diurno' : 10.0,     # W/m² mínimo de G_Glob para considerar instante diurno
    'modelos': {
        'RNN' : {'units': 64, 'lr': 0.001,  'dropout': 0.1,
                 'justificacion': 'Mejor resultado en V9/V11. 64 unidades óptimas.'},
        'LSTM': {'units': 32, 'lr': 0.001,  'dropout': 0.1,
                 'justificacion': 'Mejor resultado en V12 con 32 unidades. '
                                  'Mayor capacidad individual no mejora resultados.'},
        'GRU' : {'units': 32, 'lr': 0.0005, 'dropout': 0.1,
                 'justificacion': 'Mejor resultado en V11 con lr reducido. '
                                  'GRU sensible a lr alto por estructura de compuertas.'},
        'CNN' : {'filters': 64, 'kernel': 3, 'lr': 0.001, 'dropout': 0.1,
                 'justificacion': 'Arquitectura de referencia no recurrente.'},
    }
}

# =====================================================================
# GESTIÓN DE VERSIONES Y CARPETA DE SALIDA
# =====================================================================
ruta_base = 'modelos_tfg'
prefijo   = 'entrenamiento_v'
os.makedirs(ruta_base, exist_ok=True)

carpetas_existentes = [
    d for d in os.listdir(ruta_base)
    if os.path.isdir(os.path.join(ruta_base, d)) and d.startswith(prefijo)
]
if not carpetas_existentes:
    nueva_version = 1
else:
    numeros = []
    for c in carpetas_existentes:
        try:
            num = int(re.findall(r'^' + prefijo + r'(\d+)', c)[0])
            numeros.append(num)
        except (IndexError, ValueError):
            continue
    nueva_version = max(numeros) + 1 if numeros else 1

carpeta_salida = os.path.join(ruta_base, f'{prefijo}{nueva_version}')
os.makedirs(carpeta_salida, exist_ok=True)
print(f"\n{'='*70}")
print(f"  ENTRENAMIENTO {CONFIG['version']} — Carpeta: {carpeta_salida}")
print(f"{'='*70}\n")

# =====================================================================
# 1. CARGA Y PREPROCESADO DE DATOS
# =====================================================================
csv_path = f"modelos_tfg/{CONFIG['dataset']}"
try:
    data = pd.read_csv(csv_path, sep=';', decimal=',')
    print(f"[OK] Dataset cargado: {len(data)} filas — {csv_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)
data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True)

# Codificación cíclica de hora y mes
horas = data.index.hour
meses = data.index.month
data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
data['mes_sin']  = np.sin(meses * (2 * np.pi / 12))
data['mes_cos']  = np.cos(meses * (2 * np.pi / 12))

data_selected = data[CONFIG['features']]
print(f"[OK] Features seleccionadas ({len(CONFIG['features'])}): {CONFIG['features']}")
print(f"     Filas totales (con noches): {len(data_selected)}\n")

# =====================================================================
# 2. SPLIT TEMPORAL 80/20
# =====================================================================
train_split = int(CONFIG['train_split'] * len(data_selected))
train_df    = data_selected.iloc[:train_split]
val_df      = data_selected.iloc[train_split:]

print(f"[OK] Split temporal 80/20:")
print(f"     Entrenamiento : {len(train_df)} filas "
      f"({train_df.index[0].date()} → {train_df.index[-1].date()})")
print(f"     Validación    : {len(val_df)} filas "
      f"({val_df.index[0].date()} → {val_df.index[-1].date()})\n")

# =====================================================================
# 3. NORMALIZACIÓN — MinMaxScaler separado para X e y
# =====================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X_df = train_df.drop(columns=[CONFIG['target']])
val_X_df   = val_df.drop(columns=[CONFIG['target']])

transformed_train_X = scaler_X.fit_transform(train_X_df)
transformed_val_X   = scaler_X.transform(val_X_df)

transformed_train_y = scaler_y.fit_transform(train_df[[CONFIG['target']]])
transformed_val_y   = scaler_y.transform(val_df[[CONFIG['target']]])

# G_Glob sin escalar para la máscara diurna (valores físicos en W/m²)
g_glob_train_raw = train_df['G_Glob'].values
g_glob_val_raw   = val_df['G_Glob'].values

print(f"[OK] Normalización MinMaxScaler completada.")
print(f"     Pot_inv — min: {train_df[CONFIG['target']].min():.1f} W  "
      f"max: {train_df[CONFIG['target']].max():.1f} W\n")

# =====================================================================
# 4. CREACIÓN DE SECUENCIAS CON VENTANA DESLIZANTE
#    Devuelve también G_Glob del instante objetivo para la máscara diurna
# =====================================================================
def create_multivariate_sequences(X, y, g_glob_raw, timestamps,
                                   seq_length, look_ahead):
    """
    Ventana deslizante con validación temporal estricta.
    Descarta secuencias que cruzan discontinuidades (noche→día, cambios de día).

    Retorna:
        X_seq  : (N, seq_length, n_features)
        y_seq  : (N,)
        g_seq  : (N,) — G_Glob del instante objetivo (W/m², sin escalar)
    """
    Xs, ys, gs = [], [], []
    limite           = len(X) - seq_length - look_ahead + 1
    minutos_esperados = 10 * (seq_length + look_ahead - 1)
    tiempo_esperado  = pd.Timedelta(minutes=minutos_esperados)
    saltos_ignorados = 0

    for i in range(limite):
        t_inicio = timestamps[i]
        t_fin    = timestamps[i + seq_length + look_ahead - 1]

        if (t_fin - t_inicio) == tiempo_esperado:
            Xs.append(X[i : i + seq_length])
            ys.append(y[i + seq_length + look_ahead - 1])
            gs.append(g_glob_raw[i + seq_length + look_ahead - 1])
        else:
            saltos_ignorados += 1

    print(f"   Secuencias válidas: {len(Xs):>6}  |  "
          f"Descartadas (discontinuidades): {saltos_ignorados}")
    return np.array(Xs), np.array(ys), np.array(gs)

seq_length = CONFIG['seq_length']
look_ahead = CONFIG['look_ahead']

print(f"[INFO] Creando secuencias — seq_length={seq_length} pasos "
      f"({seq_length*10} min)  look_ahead={look_ahead} pasos "
      f"({look_ahead*10} min)...")

print("  → Train:")
X_train, y_train, g_train = create_multivariate_sequences(
    transformed_train_X, transformed_train_y.flatten(),
    g_glob_train_raw, train_df.index, seq_length, look_ahead)

print("  → Val:")
X_val, y_val, g_val = create_multivariate_sequences(
    transformed_val_X, transformed_val_y.flatten(),
    g_glob_val_raw, val_df.index, seq_length, look_ahead)

print(f"\n[OK] Forma de X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"     Forma de X_val:   {X_val.shape}  |  y_val:   {y_val.shape}\n")

# Máscara diurna para evaluación honesta (sin ceros nocturnos)
mascara_dia_val = g_val > CONFIG['umbral_diurno']
print(f"[INFO] Instantes diurnos en validación "
      f"(G_Glob > {CONFIG['umbral_diurno']} W/m²): "
      f"{mascara_dia_val.sum()} / {len(mascara_dia_val)} "
      f"({100*mascara_dia_val.mean():.1f}%)\n")

# =====================================================================
# 5. DEFINICIÓN DE ARQUITECTURAS
#    Hiperparámetros óptimos individuales por modelo (derivados de V1-V12)
# =====================================================================
n_features   = X_train.shape[2]
forma_entrada = (seq_length, n_features)

def create_model(model_type, input_shape):
    cfg = CONFIG['modelos'][model_type]

    if model_type == 'RNN':
        model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(cfg['units'], activation='tanh', return_sequences=False),
            Dropout(cfg['dropout']),
            Dense(16, activation='relu'),
            Dense(1)
        ], name='Simple_RNN')
        lr = cfg['lr']

    elif model_type == 'LSTM':
        model = Sequential([
            Input(shape=input_shape),
            LSTM(cfg['units'], activation='tanh', return_sequences=False),
            Dropout(cfg['dropout']),
            Dense(16, activation='relu'),
            Dense(1)
        ], name='LSTM')
        lr = cfg['lr']

    elif model_type == 'GRU':
        model = Sequential([
            Input(shape=input_shape),
            GRU(cfg['units'], activation='tanh', return_sequences=False),
            Dropout(cfg['dropout']),
            Dense(16, activation='relu'),
            Dense(1)
        ], name='GRU')
        lr = cfg['lr']

    elif model_type == 'CNN':
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=cfg['filters'], kernel_size=cfg['kernel'],
                   activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(1)
        ], name='CNN')
        lr = cfg['lr']

    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

K.clear_session()
print("[INFO] Construyendo modelos...\n")

model_RNN  = create_model('RNN',  forma_entrada)
model_LSTM = create_model('LSTM', forma_entrada)
model_GRU  = create_model('GRU',  forma_entrada)
model_CNN  = create_model('CNN',  forma_entrada)

# Resumen de parámetros
for nombre, modelo in [('RNN', model_RNN), ('LSTM', model_LSTM),
                        ('GRU', model_GRU), ('CNN',  model_CNN)]:
    params = modelo.count_params()
    flash  = (params * 4) / 1024
    print(f"  {nombre:4s} — Parámetros: {params:>6}  |  Flash estimada: {flash:.1f} KB")
print()

# =====================================================================
# 6. ENTRENAMIENTO
# =====================================================================
def entrenar_modelo(model, model_name):
    print(f"\n{'─'*60}")
    print(f"  ENTRENANDO: {model_name}")
    print(f"{'─'*60}")

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=CONFIG['patience_es'],
        restore_best_weights=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=CONFIG['lr_factor'],
        patience=CONFIG['patience_lr'], min_lr=CONFIG['lr_min'], verbose=1)

    ruta_guardado = os.path.join(carpeta_salida,
                                  f"{model_name.replace(' ', '_')}_mejor.h5")
    checkpoint = ModelCheckpoint(
        filepath=ruta_guardado, monitor='val_loss',
        save_best_only=True, save_weights_only=False, verbose=0)

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG['n_epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1)
    t1 = time.time()

    epocas_reales = len(history.history['loss'])
    print(f"\n  [{model_name}] Entrenado en {t1-t0:.1f} s  |  "
          f"Épocas hasta convergencia: {epocas_reales}")
    return history, round(t1 - t0, 1), epocas_reales

hist_RNN,  t_RNN,  ep_RNN  = entrenar_modelo(model_RNN,  'Simple RNN')
hist_LSTM, t_LSTM, ep_LSTM = entrenar_modelo(model_LSTM, 'LSTM')
hist_GRU,  t_GRU,  ep_GRU  = entrenar_modelo(model_GRU,  'GRU')
hist_CNN,  t_CNN,  ep_CNN  = entrenar_modelo(model_CNN,  'CNN')

print(f"\n[OK] Todos los modelos entrenados.\n")

# =====================================================================
# 7. EVALUACIÓN — MÉTRICAS DUALES (COMPLETO + DIURNO)
# =====================================================================
y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

def evaluar_modelo(model, model_name, tiempo_train, epocas):
    preds_scaled = model.predict(X_val, verbose=0)
    preds_real   = scaler_y.inverse_transform(preds_scaled).flatten()
    preds_real   = np.maximum(preds_real, 0)   # clipping físico

    # --- Métricas sobre dataset COMPLETO ---
    mae_c  = mean_absolute_error(y_val_real, preds_real)
    rmse_c = np.sqrt(mean_squared_error(y_val_real, preds_real))
    r2_c   = r2_score(y_val_real, preds_real)
    n, p   = X_val.shape[0], X_val.shape[2]
    r2a_c  = 1 - (1 - r2_c) * (n - 1) / (n - p - 1)

    # --- Métricas sobre instantes DIURNOS únicamente ---
    y_dia    = y_val_real[mascara_dia_val]
    p_dia    = preds_real[mascara_dia_val]
    mae_d    = mean_absolute_error(y_dia, p_dia)
    rmse_d   = np.sqrt(mean_squared_error(y_dia, p_dia))
    r2_d     = r2_score(y_dia, p_dia)
    n_d      = mascara_dia_val.sum()
    r2a_d    = 1 - (1 - r2_d) * (n_d - 1) / (n_d - p - 1)

    # Flash
    params   = model.count_params()
    flash_kb = (params * 4) / 1024

    return {
        'nombre'      : model_name,
        'preds_real'  : preds_real,
        # Completo
        'mae_c'  : mae_c,  'rmse_c' : rmse_c,
        'r2_c'   : r2_c,   'r2a_c'  : r2a_c,
        # Diurno
        'mae_d'  : mae_d,  'rmse_d' : rmse_d,
        'r2_d'   : r2_d,   'r2a_d'  : r2a_d,
        # Hardware
        'flash_kb'    : flash_kb,
        'params'      : params,
        'tiempo_train': tiempo_train,
        'epocas'      : epocas,
    }

res_RNN  = evaluar_modelo(model_RNN,  'Simple RNN', t_RNN,  ep_RNN)
res_LSTM = evaluar_modelo(model_LSTM, 'LSTM',       t_LSTM, ep_LSTM)
res_GRU  = evaluar_modelo(model_GRU,  'GRU',        t_GRU,  ep_GRU)
res_CNN  = evaluar_modelo(model_CNN,  'CNN',         t_CNN,  ep_CNN)

resultados = [res_RNN, res_LSTM, res_GRU, res_CNN]

# =====================================================================
# 8. TABLA DE RESULTADOS EN CONSOLA
# =====================================================================
sep  = '=' * 105
sep2 = '-' * 105

print(f"\n{sep}")
print(f"  RESULTADOS V13 — MÉTRICAS SOBRE DATASET COMPLETO (con noches)")
print(sep)
print(f"{'MODELO':12} | {'MAE (W)':>10} | {'RMSE (W)':>10} | "
      f"{'R²':>7} | {'R² Aj.':>7} | {'Flash KB':>9} | "
      f"{'Tiempo (s)':>11} | {'Épocas':>7}")
print(sep2)
for r in resultados:
    print(f"{r['nombre']:12} | {r['mae_c']:>10.3f} | {r['rmse_c']:>10.3f} | "
          f"{r['r2_c']:>7.4f} | {r['r2a_c']:>7.4f} | {r['flash_kb']:>9.1f} | "
          f"{r['tiempo_train']:>11.1f} | {r['epocas']:>7}")
print(sep)

print(f"\n{sep}")
print(f"  RESULTADOS V13 — MÉTRICAS SOLO INSTANTES DIURNOS "
      f"(G_Glob > {CONFIG['umbral_diurno']} W/m²)")
print(sep)
print(f"{'MODELO':12} | {'MAE (W)':>10} | {'RMSE (W)':>10} | "
      f"{'R²':>7} | {'R² Aj.':>7}")
print(sep2)
for r in resultados:
    print(f"{r['nombre']:12} | {r['mae_d']:>10.3f} | {r['rmse_d']:>10.3f} | "
          f"{r['r2_d']:>7.4f} | {r['r2a_d']:>7.4f}")
print(sep)

# Modelo ganador por R² total
ganador = max(resultados, key=lambda x: x['r2_c'])
print(f"\n  ★ Modelo ganador por R² (dataset completo): "
      f"{ganador['nombre']} — R²={ganador['r2_c']:.4f}  "
      f"MAE={ganador['mae_c']:.1f} W  Flash={ganador['flash_kb']:.1f} KB\n")

# =====================================================================
# 9. GUARDAR RESULTADOS EN JSON (TRAZABILIDAD TOTAL)
# =====================================================================
resultados_json = []
for r in resultados:
    entrada = {k: v for k, v in r.items() if k != 'preds_real'}
    resultados_json.append(entrada)

CONFIG['resultados'] = resultados_json
CONFIG['modelo_ganador'] = ganador['nombre']

with open(os.path.join(carpeta_salida, 'configuracion_y_resultados_v13.json'),
          'w', encoding='utf-8') as f:
    json.dump(CONFIG, f, indent=4, ensure_ascii=False)

print(f"[OK] Configuración y resultados guardados en JSON.\n")

# =====================================================================
# 10. VISUALIZACIONES
# =====================================================================
COLOR_REAL = '#000000'
COLOR_RNN  = '#E53935'
COLOR_LSTM = '#1E88E5'
COLOR_GRU  = '#43A047'
COLOR_CNN  = '#8E24AA'

colores_modelos = {
    'Simple RNN': COLOR_RNN,
    'LSTM'      : COLOR_LSTM,
    'GRU'       : COLOR_GRU,
    'CNN'       : COLOR_CNN,
}

preds_rnn_real  = res_RNN['preds_real']
preds_lstm_real = res_LSTM['preds_real']
preds_gru_real  = res_GRU['preds_real']
preds_cnn_real  = res_CNN['preds_real']

# --- Gráfica 1: Val Loss unificado ---
plt.figure(figsize=(12, 5))
for hist, nombre, color in [
        (hist_RNN,  'Simple RNN', COLOR_RNN),
        (hist_LSTM, 'LSTM',       COLOR_LSTM),
        (hist_GRU,  'GRU',        COLOR_GRU),
        (hist_CNN,  'CNN',        COLOR_CNN)]:
    plt.plot(hist.history['val_loss'], label=nombre, color=color, linewidth=2)

plt.title('Evolución del Error de Validación durante el Entrenamiento',
          fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '1_val_loss_unificado.png'), dpi=300)
plt.show()

# --- Gráfica 1b: Train vs Val Loss por modelo ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Diagnóstico de Entrenamiento: Train Loss vs Validation Loss',
             fontsize=16, fontweight='bold')

historiales = [hist_RNN, hist_LSTM, hist_GRU, hist_CNN]
nombres_plot = ['Simple RNN', 'LSTM', 'GRU', 'CNN']
colores_plot = [COLOR_RNN, COLOR_LSTM, COLOR_GRU, COLOR_CNN]
axes_flat    = axes.flatten()

for i in range(4):
    ax  = axes_flat[i]
    ax.plot(historiales[i].history['loss'],
            label='Training Loss', color='gray', linestyle='--', linewidth=2)
    ax.plot(historiales[i].history['val_loss'],
            label='Validation Loss', color=colores_plot[i], linewidth=2.5)
    ax.set_title(f'Modelo: {nombres_plot[i]}', fontsize=13)
    ax.set_xlabel('Épocas', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(os.path.join(carpeta_salida, '1b_diagnostico_train_val.png'), dpi=300)
plt.show()

# --- Gráfica 2: Dispersión del modelo ganador ---
mejor_preds = ganador['preds_real']
mejor_color = colores_modelos[ganador['nombre']]

plt.figure(figsize=(8, 8))
max_val = np.max(y_val_real) * 1.05
plt.scatter(y_val_real, mejor_preds, alpha=0.5, color=mejor_color,
            s=15, label=f"Predicciones {ganador['nombre']}")
plt.plot([0, max_val], [0, max_val], color=COLOR_REAL,
         linestyle='--', linewidth=2.5, label='Ideal')
plt.title(f"Dispersión del Modelo Óptimo ({ganador['nombre']}): "
          f"Real vs. Predicción", fontsize=14, fontweight='bold')
plt.xlabel('Potencia Real (W)', fontsize=12)
plt.ylabel('Potencia Predicha (W)', fontsize=12)
plt.xlim(0, max_val); plt.ylim(0, max_val)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '2_dispersion_ganador.png'), dpi=300)
plt.show()

# --- Gráfica 3: Detalle día despejado ---
D_SOL_INI, D_SOL_FIN = 0, 144
plt.figure(figsize=(14, 5))
plt.plot(y_val_real[D_SOL_INI:D_SOL_FIN],
         label='Real', color=COLOR_REAL, linewidth=3)
plt.plot(preds_rnn_real[D_SOL_INI:D_SOL_FIN],
         label='RNN',  color=COLOR_RNN,  linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[D_SOL_INI:D_SOL_FIN],
         label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[D_SOL_INI:D_SOL_FIN],
         label='GRU',  color=COLOR_GRU,  linewidth=2)
plt.plot(preds_cnn_real[D_SOL_INI:D_SOL_FIN],
         label='CNN',  color=COLOR_CNN,  linewidth=2)
plt.title('Detalle de Predicción: Día Despejado (Curva de Campana)',
          fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '3_zoom_soleado.png'), dpi=300)
plt.show()

# --- Gráfica 4: Detalle día nublado ---
D_NUB_INI, D_NUB_FIN = 4170, 4320
plt.figure(figsize=(14, 5))
plt.plot(y_val_real[D_NUB_INI:D_NUB_FIN],
         label='Real', color=COLOR_REAL, linewidth=3)
plt.plot(preds_rnn_real[D_NUB_INI:D_NUB_FIN],
         label='RNN',  color=COLOR_RNN,  linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[D_NUB_INI:D_NUB_FIN],
         label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[D_NUB_INI:D_NUB_FIN],
         label='GRU',  color=COLOR_GRU,  linewidth=2)
plt.plot(preds_cnn_real[D_NUB_INI:D_NUB_FIN],
         label='CNN',  color=COLOR_CNN,  linewidth=2)
plt.title('Detalle de Predicción: Día Nublado (Alta Variabilidad)',
          fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '4_zoom_nublado.png'), dpi=300)
plt.show()

# --- Gráfica 5: Comparativa temporal ---
T_INI, T_FIN = 280, 1300
plt.figure(figsize=(16, 6))
plt.plot(y_val_real[T_INI:T_FIN],
         label='Potencia Real', color=COLOR_REAL, linewidth=3, zorder=5)
plt.plot(preds_rnn_real[T_INI:T_FIN],
         label='RNN',  color=COLOR_RNN,  linewidth=2, linestyle='--', alpha=0.9)
plt.plot(preds_lstm_real[T_INI:T_FIN],
         label='LSTM', color=COLOR_LSTM, linewidth=2, alpha=0.9)
plt.plot(preds_gru_real[T_INI:T_FIN],
         label='GRU',  color=COLOR_GRU,  linewidth=2, alpha=0.9)
plt.plot(preds_cnn_real[T_INI:T_FIN],
         label='CNN',  color=COLOR_CNN,  linewidth=2, alpha=0.9)
plt.title('Comparativa Temporal de Potencia Generada (Finales Octubre)',
          fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo (Intervalos de 10 min)', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.margins(x=0)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '5_comparativa_temporal.png'), dpi=300)
plt.show()

# --- Gráfica 6: Biobjetivo MAE vs Flash ---
etiquetas   = ['Simple RNN', 'LSTM', 'GRU', 'CNN']
valores_mae = [r['mae_c']    for r in resultados]
valores_kb  = [r['flash_kb'] for r in resultados]
x     = np.arange(len(etiquetas))
width = 0.35

fig, ax1 = plt.subplots(figsize=(11, 6))
bar1 = ax1.bar(x - width/2, valores_mae, width,
               color='#F57C00', edgecolor='black', linewidth=1.2,
               label='MAE (W)')
ax1.set_ylabel('Error Promedio MAE (W)', fontsize=12,
               fontweight='bold', color='#F57C00')
ax1.tick_params(axis='y', labelcolor='#F57C00')
ax1.set_xticks(x)
ax1.set_xticklabels(etiquetas, fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(valores_mae) * 1.20)

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, valores_kb, width,
               color='#4527A0', edgecolor='black', linewidth=1.2,
               label='Flash (KB)')
ax2.set_ylabel('Memoria Flash Estimada (KB)', fontsize=12,
               fontweight='bold', color='#4527A0')
ax2.tick_params(axis='y', labelcolor='#4527A0')
ax2.set_ylim(0, max(valores_kb) * 1.20)

for bar in bar1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2,
             h + max(valores_mae)*0.02,
             f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bar2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2,
             h + max(valores_kb)*0.02,
             f'{h:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Comparativa TinyML: Precisión vs. Ligereza de Hardware',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(os.path.join(carpeta_salida, '6_barras_biobjetivo.png'), dpi=300)
plt.show()

# --- Gráfica 7 (NUEVA): R² diurno vs R² completo — comparativa de sesgo nocturno ---
r2_completo = [r['r2_c'] for r in resultados]
r2_diurno   = [r['r2_d'] for r in resultados]

x2    = np.arange(len(etiquetas))
width2 = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x2 - width2/2, r2_completo, width2,
            color='#546E7A', edgecolor='black', linewidth=1.2,
            label='R² — Dataset completo (con noches)')
b2 = ax.bar(x2 + width2/2, r2_diurno, width2,
            color='#FF8F00', edgecolor='black', linewidth=1.2,
            label=f'R² — Solo diurno (G_Glob > {CONFIG["umbral_diurno"]} W/m²)')

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
            f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
            f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Coeficiente de Determinación R²', fontsize=12)
ax.set_xticks(x2)
ax.set_xticklabels(etiquetas, fontsize=12, fontweight='bold')
ax.set_ylim(min(r2_diurno) * 0.95, 1.01)
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.title('R² con y sin datos nocturnos — Impacto del sesgo nocturno',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, '7_r2_completo_vs_diurno.png'), dpi=300)
plt.show()

# =====================================================================
# 11. RESUMEN FINAL EN CONSOLA
# =====================================================================
print(f"\n{'='*70}")
print(f"  RESUMEN FINAL — {CONFIG['version']}")
print(f"{'='*70}")
print(f"  Carpeta de salida : {carpeta_salida}")
print(f"  JSON de trazabilidad guardado.")
print(f"  Gráficas generadas: 7")
print(f"\n  MODELO GANADOR: {ganador['nombre']}")
print(f"    R²  (completo) : {ganador['r2_c']:.4f}")
print(f"    MAE (completo) : {ganador['mae_c']:.1f} W")
print(f"    R²  (diurno)   : {ganador['r2_d']:.4f}")
print(f"    MAE (diurno)   : {ganador['mae_d']:.1f} W")
print(f"    Flash estimada : {ganador['flash_kb']:.1f} KB")
print(f"    Parámetros     : {ganador['params']}")
print(f"{'='*70}\n")