import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json
import os

# ─── Parámetros (mismos que V13) ──────────────────────────────────────
EXPORT_DIR   = 'modelos_tfg/exportacion_micro'
CSV_PATH     = 'modelos_tfg/datos_10min_modelos.csv'
SEQ_LENGTH   = 18
LOOK_AHEAD   = 6
FEATURES     = ['hora_sin','hora_cos','mes_sin','mes_cos',
                'G_Glob','Ta','Hum_Rel','Tc','Pot_inv']
TARGET       = 'Pot_inv'
UMBRAL_DIA   = 10.0    # W/m²
N_CALIB      = 3000    # muestras para calibrar la cuantización

# ─── DIRECTORIO DE EXPORTACIÓN ────────────────────────────────────────
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f'[*] Todos los archivos se guardarán en: {EXPORT_DIR}')

# ─── Carga y preprocesado (idéntico a V13) ────────────────────────────
data = pd.read_csv(CSV_PATH, sep=';', decimal=',')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)
data.drop(columns=['Gefsaypce','EDC','EACAC','Vmpp_panel'], inplace=True)

horas = data.index.hour
meses = data.index.month
data['hora_sin'] = np.sin(horas * (2*np.pi/24))
data['hora_cos'] = np.cos(horas * (2*np.pi/24))
data['mes_sin']  = np.sin(meses * (2*np.pi/12))
data['mes_cos']  = np.cos(meses * (2*np.pi/12))

data_sel = data[FEATURES]
train_split = int(0.8 * len(data_sel))
train_df = data_sel.iloc[:train_split]
val_df   = data_sel.iloc[train_split:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
train_X  = scaler_X.fit_transform(train_df.drop(columns=[TARGET]))
val_X    = scaler_X.transform(val_df.drop(columns=[TARGET]))
train_y  = scaler_y.fit_transform(train_df[[TARGET]]).flatten()
val_y    = scaler_y.transform(val_df[[TARGET]]).flatten()
g_val    = val_df['G_Glob'].values

def make_sequences(X, y, g, ts, seq_len, look_ahead):
    Xs, ys, gs = [], [], []
    dt = pd.Timedelta(minutes=10*(seq_len+look_ahead-1))
    for i in range(len(X) - seq_len - look_ahead + 1):
        if (ts[i+seq_len+look_ahead-1] - ts[i]) == dt:
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len+look_ahead-1])
            gs.append(g[i+seq_len+look_ahead-1])
    return np.array(Xs), np.array(ys), np.array(gs)

X_val, y_val, g_val_seq = make_sequences(
    val_X, val_y, g_val, val_df.index, SEQ_LENGTH, LOOK_AHEAD)
X_train_seq, y_train_seq, _ = make_sequences(
    train_X, train_y, train_df['G_Glob'].values,
    train_df.index, SEQ_LENGTH, LOOK_AHEAD)

y_val_real = scaler_y.inverse_transform(
    y_val.reshape(-1,1)).flatten()
mascara_dia = g_val_seq > UMBRAL_DIA

calib_idx = np.random.choice(len(X_train_seq), N_CALIB, replace=False)
X_calib   = X_train_seq[calib_idx].astype(np.float32)

def representative_dataset():
    for i in range(len(X_calib)):
        yield [X_calib[i:i+1]]

def evaluar(nombre, preds_real):
    preds_real = np.maximum(preds_real, 0)
    mae  = mean_absolute_error(y_val_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_val_real, preds_real))
    r2   = r2_score(y_val_real, preds_real)
    mae_d  = mean_absolute_error(y_val_real[mascara_dia], preds_real[mascara_dia])
    r2_d   = r2_score(y_val_real[mascara_dia], preds_real[mascara_dia])
    print(f'\n  {nombre}')
    print(f'    MAE={mae:.1f}W  RMSE={rmse:.1f}W  R²={r2:.4f}')
    print(f'    MAE diurno={mae_d:.1f}W  R² diurno={r2_d:.4f}')
    return {'mae':mae,'rmse':rmse,'r2':r2,'mae_d':mae_d,'r2_d':r2_d}

modelos = {
    'LSTM': 'modelos_tfg/entrenamiento_v13/LSTM_mejor.h5',
    'GRU':  'nuevo_GRU/GRU_mejor.h5',
}

resultados = {}

for nombre, ruta_h5 in modelos.items():
    print(f'\n{"="*55}')
    print(f'  Procesando: {nombre}')
    print(f'{"="*55}')

    # ── Cargar modelo float32 original (el que tiene bucles WHILE) ──
    model_original = tf.keras.models.load_model(ruta_h5, compile=False)
    
    # ── Reconstruir modelo DESENROLLADO (unroll=True) para microcontroladores ──
    print(f'  [INFO] Desenrollando arquitectura (unroll=True) para evitar error INT32...')
    
    # 1. Creamos un modelo vacío
    model_unrolled = tf.keras.Sequential()
    
    # 2. Fijamos la entrada rígidamente (obligatorio para hacer unroll)
    model_unrolled.add(tf.keras.layers.InputLayer(input_shape=(SEQ_LENGTH, len(FEATURES)-1)))

    # 3. Clonamos tu arquitectura exacta capa por capa
    for capa in model_original.layers:
        # Nos saltamos la entrada original porque ya la hemos puesto fija arriba
        if isinstance(capa, tf.keras.layers.InputLayer):
            continue
            
        config = capa.get_config()
        
        # Si es la LSTM o GRU, le inyectamos la magia
        if isinstance(capa, (tf.keras.layers.LSTM, tf.keras.layers.GRU)):
            config['unroll'] = True
            model_unrolled.add(capa.__class__.from_config(config))
        # Si es cualquier otra cosa (Dense, Dropout...), la copiamos tal cual
        else:
            model_unrolled.add(capa.__class__.from_config(config))

    # 4. Inyectamos los pesos (ahora sí, encajarán perfectamente)
    model_unrolled.set_weights(model_original.get_weights())
    
    # A partir de aquí, usamos el modelo desenrollado como el modelo principal
    model = model_unrolled

    params_f32 = model.count_params()
    flash_f32  = params_f32 * 4 / 1024
    print(f'  Float32 — params: {params_f32}  Flash est.: {flash_f32:.1f} KB')

    # ── Evaluar float32 como referencia ─────────────────────────────
    preds_f32 = scaler_y.inverse_transform(
        model.predict(X_val, verbose=0)).flatten()
    res_f32 = evaluar(f'{nombre} Float32 Unrolled', preds_f32)

    # ── Convertir a TFLite Float32 (sin cuantizar) ───────────────────
    conv_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = conv_f32.convert()
    ruta_f32 = os.path.join(EXPORT_DIR, f'{nombre}_float32.tflite')
    with open(ruta_f32, 'wb') as f:
        f.write(tflite_f32)
    size_f32 = len(tflite_f32) / 1024
    print(f'\n  TFLite Float32: {size_f32:.1f} KB → {ruta_f32}')

# ── Convertir a TFLite INT8 (FULL INTEGER QUANTIZATION - TFG) ──────────────
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 1. Activamos la calibración con el dataset para cuantizar las activaciones
    conv_int8.representative_dataset = representative_dataset
    
    # 2. Obligamos a que TODAS las operaciones internas de la red sean INT8 puras
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 3. TRUCO MAGISTRAL: Mantenemos las puertas de entrada/salida en Float32
    # Esto permite que tu código C++ (main.cpp) no tenga que cambiar y siga 
    # enviando los datos y recibiendo los vatios como siempre. El micro hará 
    # la conversión Float->INT8->Float de forma transparente en las fronteras.
    conv_int8.inference_input_type = tf.float32
    conv_int8.inference_output_type = tf.float32

    tflite_int8 = conv_int8.convert()
    ruta_int8 = os.path.join(EXPORT_DIR, f'{nombre}_int8.tflite')
    with open(ruta_int8, 'wb') as f:
        f.write(tflite_int8)
    size_int8 = len(tflite_int8) / 1024
    print(f'  TFLite INT8 (Full):    {size_int8:.1f} KB → {ruta_int8}')
    print(f'  Reducción:      {(1 - size_int8/size_f32)*100:.1f}%')

    # ── Evaluar TFLite INT8 ──────────────────────────────────────────
    interp = tf.lite.Interpreter(model_content=tflite_int8)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds_int8 = []

    print(f'  [INFO] Evaluando {len(X_val)} secuencias en INT8 Full...')
    for i in range(len(X_val)):
        if i > 0 and i % 1000 == 0:
            print(f'    ... Procesadas {i} de {len(X_val)} muestras')
            
        interp.set_tensor(inp['index'], X_val[i:i+1].astype(np.float32))
        interp.invoke()
        valor_predicho = interp.get_tensor(out['index']).flatten()[0]
        preds_int8.append(valor_predicho)
    print(f'  [INFO] Evaluación INT8 completada.')

    preds_int8 = scaler_y.inverse_transform(
        np.array(preds_int8).reshape(-1,1)).flatten()
    res_int8 = evaluar(f'{nombre} INT8 Full', preds_int8)

    delta_mae = res_int8['mae'] - res_f32['mae']
    delta_r2  = res_f32['r2']  - res_int8['r2']
    print(f'\n  Degradación: ΔMAE={delta_mae:+.1f}W  ΔR²={delta_r2:+.4f}')

    # ── Generar archivo .h de los pesos para ESP32 ──────────────────
    nombre_var = f'{nombre.lower()}_model'
    lineas = [
        f'// Modelo {nombre} FULL INT8 para ESP32-S3 (Requisito TFG)',
        f'// Generado automáticamente — NO EDITAR',
        f'// Tamaño: {size_int8:.1f} KB',
        f'',
        f'#pragma once',
        f'#include <stdint.h>',
        f'',
        f'alignas(16) const uint8_t {nombre_var}_data[] = {{',
    ]
    hex_bytes = [f'  0x{b:02x}' for b in tflite_int8]
    for j in range(0, len(hex_bytes), 12):
        lineas.append(', '.join(hex_bytes[j:j+12]) + ',')
    lineas += [
        '};',
        f'const int {nombre_var}_data_len = {len(tflite_int8)};',
    ]
    ruta_h = os.path.join(EXPORT_DIR, f'{nombre}_model.h')
    with open(ruta_h, 'w') as f:
        f.write('\n'.join(lineas))
    print(f'  Header C generado: {ruta_h}')

    resultados[nombre] = {
        'flash_f32_kb': flash_f32,
        'tflite_f32_kb': size_f32,
        'tflite_int8_kb': size_int8,
        'res_f32': res_f32,
        'res_int8': res_int8,
        'delta_mae': delta_mae,
        'delta_r2': delta_r2,
    }

scaler_params = {
    'y_min': float(scaler_y.data_min_[0]),
    'y_max': float(scaler_y.data_max_[0]),
    'X_min': scaler_X.data_min_.tolist(),
    'X_max': scaler_X.data_max_.tolist(),
    'features': FEATURES[:-1], 
}
ruta_json = os.path.join(EXPORT_DIR, 'scaler_params.json')
with open(ruta_json, 'w') as f:
    json.dump(scaler_params, f, indent=2)
print(f'\n[OK] Parámetros guardados en: {ruta_json}')

texto_tabla = f'\n{"="*70}\n'
texto_tabla += f'  TABLA COMPARATIVA FINAL\n'
texto_tabla += f'{"="*70}\n'
texto_tabla += f'{"Modelo":12} {"TFLite F32":>12} {"TFLite INT8":>12} {"Reduc.":>8} {"ΔMAE":>8} {"ΔR²":>8}\n'
texto_tabla += '-'*70 + '\n'
for nombre, r in resultados.items():
    texto_tabla += (f'{nombre:12} {r["tflite_f32_kb"]:>10.1f}KB {r["tflite_int8_kb"]:>10.1f}KB '
                    f'{(1-r["tflite_int8_kb"]/r["tflite_f32_kb"])*100:>7.1f}% '
                    f'{r["delta_mae"]:>+7.1f}W {r["delta_r2"]:>+8.4f}\n')
texto_tabla += f'{"="*70}\n'

print(texto_tabla)

ruta_txt = os.path.join(EXPORT_DIR, 'resultados_cuantizacion.txt')
with open(ruta_txt, 'w', encoding='utf-8') as f:
    f.write(texto_tabla)
print(f'[OK] Tabla de resultados exportada a: {ruta_txt}')