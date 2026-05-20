import serial
import time
import os
import datetime as datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ─── Configuración ────────────────────────────────────────────────────
PORT          = 'COM13'          # Windows: COM3  |  Linux: /dev/ttyUSB0
BAUDRATE      = 115200
TIMEOUT       = 5
MODELO_ACTUAL = 'LSTM_cmpt'  # Cuando evalue GRU, poner GRU
CSV_PATH      = 'modelos_tfg/datos_10min_modelos.csv'
SEQ_LEN       = 18
LOOK_AHEAD    = 6
FEATURES      = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos',
                    'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv']
TARGET        = 'Pot_inv'
UMBRAL_DIA    = 10.0   # W/m² — umbral G_Glob para métrica diurna

# ─── Carga y preprocesado (idéntico a V13) ────────────────────────────
data = pd.read_csv(CSV_PATH, sep=';', decimal=',')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)
data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True)

horas = data.index.hour
meses = data.index.month
data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
data['mes_sin']  = np.sin(meses * (2 * np.pi / 12))
data['mes_cos']  = np.cos(meses * (2 * np.pi / 12))

data_sel    = data[FEATURES]
train_split = int(0.8 * len(data_sel))
train_df    = data_sel.iloc[:train_split]
val_df      = data_sel.iloc[train_split:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(train_df.drop(columns=[TARGET]))
scaler_y.fit(train_df[[TARGET]])

# Arrays del conjunto de validación (sin normalizar, para enviar al ESP32
# tal cual — el ESP32 normaliza internamente con scaler_params.h)
val_X_raw  = val_df.drop(columns=[TARGET]).values   # shape (N, 8)
val_y_raw  = val_df[TARGET].values                  # shape (N,)
val_g_raw  = val_df['G_Glob'].values                # shape (N,) para máscara diurna
val_ts     = val_df.index                           # timestamps para validar continuidad

N_VAL = len(val_df)
print(f'Conjunto de validación: {N_VAL} filas')
print(f'  Desde: {val_ts[0]}')
print(f'  Hasta: {val_ts[-1]}')

# ─── Precalcular máscara de ventanas temporalmente válidas ────────────
# Una predicción en el índice i (que usa pasos i-(SEQ_LEN+LOOK_AHEAD-1)
# a i) es válida si todos los pasos de la ventana son consecutivos.
# Condición: ts[i] - ts[i - (SEQ_LEN + LOOK_AHEAD - 1)] == 
#            (SEQ_LEN + LOOK_AHEAD - 1) * 10 minutos
dt_ventana = pd.Timedelta(minutes=10 * (SEQ_LEN + LOOK_AHEAD - 1))

# ventana_valida[i] = True si la predicción recibida tras enviar la fila i
# corresponde a una ventana sin saltos temporales.
# El ESP32 emite predicción después de recibir SEQ_LEN pasos,
# con el target en el paso i + LOOK_AHEAD - 1 (look_ahead pasos adelante
# desde el último paso de la ventana).
# Índice del target para la predicción emitida al enviar fila i:
#   target_idx = i + LOOK_AHEAD - 1  (si i >= SEQ_LEN - 1)
# La ventana que usó el ESP32: filas [i - SEQ_LEN + 1 .. i]
# Validez: ts[i] - ts[i - SEQ_LEN + 1 + 0] == (SEQ_LEN-1)*10min
#          Y además ts[target_idx] - ts[i] == (LOOK_AHEAD)*10min
# Simplificado: ts[target_idx] - ts[i - SEQ_LEN + 1] == dt_ventana

ventana_valida = np.zeros(N_VAL, dtype=bool)
target_idx_arr = np.full(N_VAL, -1, dtype=int)

for i in range(SEQ_LEN - 1, N_VAL):
    target_idx = i + LOOK_AHEAD - 1
    if target_idx >= N_VAL:
        break
    ini_ventana = i - SEQ_LEN + 1
    if (val_ts[target_idx] - val_ts[ini_ventana]) == dt_ventana:
        ventana_valida[i]    = True
        target_idx_arr[i]    = target_idx

n_validas = ventana_valida.sum()
n_total   = (N_VAL - SEQ_LEN - LOOK_AHEAD + 2)
print(f'\nVentanas válidas (temporalmente continuas): '
      f'{n_validas} / {n_total} '
      f'({100*n_validas/max(n_total,1):.1f}%)')
print(f'Ventanas inválidas (saltos noche→día etc.): {n_total - n_validas}')

# ─── Conectar al ESP32 y Verificar (Handshake) ────────────────────────
print(f'\nConectando a {PORT} @ {BAUDRATE} bps...')
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
except serial.SerialException as e:
    print(f"[ERROR CRÍTICO] No se pudo abrir el puerto {PORT}. ¿Está conectado o en uso?")
    exit()

time.sleep(3) # Dar tiempo al reinicio del ESP32 al conectar el puerto

# Vaciar mensajes de arranque del ESP32
while ser.in_waiting:
    msg = ser.readline().decode('utf-8', errors='ignore').strip()
    if msg:
        print(f'  ESP32 boot: {msg}')

print('  Enviando PING de comprobación...')
ser.write(b'PING\n')
ser.flush()

# Esperar respuesta PONG con un timeout de 3 segundos
t_inicio = time.time()
conexion_ok = False

while time.time() - t_inicio < 3:
    if ser.in_waiting:
        respuesta = ser.readline().decode('utf-8', errors='ignore').strip()
        if respuesta == 'PONG':
            conexion_ok = True
            break
        elif respuesta:
            # Si el ESP32 escupe un error de TFLite o reinicio, lo mostramos
            print(f'  [ESP32 LOG] {respuesta}')

if not conexion_ok:
    print('\n[FATAL] El ESP32 no ha respondido al PING.')
    print('Posibles causas:')
    print(' 1. TENSOR_ARENA_SIZE es muy pequeño y hay un bootloop (fíjate en los logs arriba).')
    print(' 2. El firmware no se ha subido correctamente.')
    ser.close()
    exit() # Abortamos ejecución al instante

print('  [OK] Respuesta PONG recibida. Enlace establecido.\n')

# ─── Enviar TODAS las filas y recoger respuestas ──────────────────────
# Se envían las 8 features sin normalizar (el ESP32 normaliza)
# El ESP32 responde:
#   - "BUFFERING:k/18"  mientras llena el buffer (primeros 17 pasos)
#   - "PRED:X.XX,LAT:Y" a partir del paso 18 en adelante

raw_preds     = np.full(N_VAL, np.nan)   # predicción recibida para cada fila i
raw_latencias = np.full(N_VAL, -1, dtype=int)

print(f'Enviando {N_VAL} filas al ESP32...')
t_inicio_total = time.time()

for i in range(N_VAL):
    # Enviar fila i (8 features sin normalizar)
    linea = ','.join([f'{v:.6f}' for v in val_X_raw[i]])
    ser.write((linea + '\n').encode())
    ser.flush()

    # Leer respuesta con timeout
    respuesta = ser.readline().decode('utf-8', errors='ignore').strip()

    if respuesta.startswith('PRED:'):
        # Formato: "PRED:1234.56,LAT:1823"
        try:
            partes   = respuesta.replace('PRED:', '').split(',LAT:')
            pred_raw = float(partes[0])   # valor normalizado [0,1]
            lat_us   = int(partes[1])
            raw_preds[i]     = pred_raw
            raw_latencias[i] = lat_us
        except (ValueError, IndexError):
            print(f'  [WARN] Fila {i}: respuesta mal formada: {respuesta}')

    elif respuesta.startswith('BUFFERING'):
        pass   # normal durante el llenado del buffer

    elif respuesta == '' or respuesta is None:
        print(f'  [WARN] Fila {i}: timeout sin respuesta del ESP32')

    else:
        print(f'  [INFO] Fila {i}: {respuesta}')

    # Progreso cada 500 filas
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t_inicio_total
        pct     = 100 * (i + 1) / N_VAL
        print(f'  {i+1}/{N_VAL} ({pct:.0f}%) — {elapsed:.0f}s transcurridos')

ser.close()
t_total = time.time() - t_inicio_total
print(f'\nEnvío completado en {t_total:.1f} s '
      f'({t_total/N_VAL*1000:.1f} ms por fila)')

# ─── Desescalar predicciones ─────────────────────────────────────────
# El ESP32 devuelve el valor normalizado [0,1].
# Python lo desescala con scaler_y.
# NOTA: si el ESP32 ya devuelve W directamente (con denormalize_output
# implementado en C++), omite este paso.
preds_w = raw_preds.copy()
mask_recibido = ~np.isnan(preds_w)
preds_w = np.maximum(preds_w, 0)   # clipping físico

# ─── Aplicar máscara de ventanas válidas ──────────────────────────────
# Solo conservamos predicciones donde:
# 1. La ventana es temporalmente continua (sin saltos noche→día)
# 2. Se recibió predicción del ESP32 (no NaN)
# 3. El índice del target está dentro del rango
mask_final = (
    ventana_valida &
    mask_recibido &
    (target_idx_arr >= 0)
)

# Extraer valores finales alineados
idx_validos   = np.where(mask_final)[0]
preds_final   = preds_w[idx_validos]
targets_final = val_y_raw[target_idx_arr[idx_validos]]
g_final       = val_g_raw[target_idx_arr[idx_validos]]
lats_final    = raw_latencias[idx_validos]
ts_final      = [val_ts[target_idx_arr[i]] for i in idx_validos]

print(f'\nPredicciones válidas para métricas: {len(preds_final)}')
print(f'  (de {N_VAL} filas enviadas, {mask_final.sum()} válidas)')

# ─── Métricas completas (con noches) ─────────────────────────────────
mae_c  = mean_absolute_error(targets_final, preds_final)
rmse_c = np.sqrt(mean_squared_error(targets_final, preds_final))
r2_c   = r2_score(targets_final, preds_final)
n_c, p_c = len(preds_final), val_X_raw.shape[1]
r2a_c  = 1 - (1 - r2_c) * (n_c - 1) / (n_c - p_c - 1)

# ─── Métricas diurnas (G_Glob del instante target > 10 W/m²) ─────────
# mascara_dia = g_final > UMBRAL_DIA
# preds_dia   = preds_final[mascara_dia]
# reals_dia   = targets_final[mascara_dia]

# mae_d  = mean_absolute_error(reals_dia, preds_dia)
# rmse_d = np.sqrt(mean_squared_error(reals_dia, preds_dia))
# r2_d   = r2_score(reals_dia, preds_dia)
# n_d    = mascara_dia.sum()
# r2a_d  = 1 - (1 - r2_d) * (n_d - 1) / (n_d - p_c - 1)

# ─── Latencias ────────────────────────────────────────────────────────
lats_validas = lats_final[lats_final > 0]

# ─── Imprimir resultados ──────────────────────────────────────────────
sep = '=' * 60
print(f'\n{sep}')
print(f'  RESULTADOS EN HARDWARE — ESP32-S3')
print(sep)
print(f'\n  MÉTRICAS DATASET COMPLETO (con noches):')
print(f'    MAE   = {mae_c:.3f} W')
print(f'    RMSE  = {rmse_c:.3f} W')
print(f'    R²    = {r2_c:.4f}')
print(f'    R² Aj.= {r2a_c:.4f}')

print(f'\n  LATENCIA DE INFERENCIA:')
if len(lats_validas) > 0:
    print(f'    Media  = {lats_validas.mean():.0f} µs  '
          f'({lats_validas.mean()/1000:.2f} ms)')
    print(f'    Máxima = {lats_validas.max():.0f} µs')
    print(f'    Mínima = {lats_validas.min():.0f} µs')
    print(f'    P95    = {np.percentile(lats_validas, 95):.0f} µs')
print(sep)

# ─── Comparativa con V13 Python ───────────────────────────────────────
V13_LSTM = {'mae_c': 51.74, 'r2_c': 0.9495}
V13_GRU  = {'mae_c': 48.08, 'r2_c': 0.9386}
V13_REF  = V13_LSTM   # Cambiar a V13_GRU en caso de evaluar la otra

print(f'\n  DEGRADACIÓN POR CUANTIZACIÓN (vs V13 Python Float32):')
print(f'    ΔMAE  completo  = {mae_c  - V13_REF["mae_c"]:+.1f} W')
print(f'    ΔR²   completo  = {r2_c   - V13_REF["r2_c"]:+.4f}')

# ─── Guardar resultados CSV ───────────────────────────────────────────
# Generar nombre de carpeta único con la fecha y hora actual
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("eval_embebidos", f"run_{timestamp}_{MODELO_ACTUAL}")
os.makedirs(out_dir, exist_ok=True)

df_out = pd.DataFrame({
    'timestamp':     ts_final,
    'real_W':        targets_final,
    'pred_W':        preds_final,
    'G_Glob':        g_final,
    'latencia_us':   lats_final,
})

# Guardamos el CSV dentro de la nueva carpeta
csv_path = os.path.join(out_dir, 'resultados_hardware.csv')
df_out.to_csv(csv_path, index=False)
print(f'\n[OK] Resultados guardados en: {csv_path}')

# ─── Gráficas (Calidad Imprenta TFG) ──────────────────────────────────

# 1. Gráfica de Dispersión
plt.figure(figsize=(8, 6))
plt.scatter(targets_final, preds_final, alpha=0.4, s=15, color='#2E5090')
mx = max(targets_final.max(), preds_final.max()) * 1.05
plt.plot([0, mx], [0, mx], 'k--', lw=2.5, label='Ideal')
plt.xlabel('Potencia Real (W)', fontsize=12, fontweight='bold')
plt.ylabel('Potencia Predicha (W)', fontsize=12, fontweight='bold')
plt.title(f'Dispersión en Hardware - {MODELO_ACTUAL} (Completo)\nMAE={mae_c:.1f}W  R²={r2_c:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'evaluacion_dispersion_hardware_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Serie Temporal (últimos 1000 pasos válidos)
plt.figure(figsize=(10, 5))
n_plot = min(1000, len(preds_final))
plt.plot(targets_final[-n_plot:], 'k-',  lw=2.5, label='Real', zorder=5)
plt.plot(preds_final[-n_plot:],   'b-',  lw=2.0, label='Predicción', alpha=0.9)
plt.xlabel('Pasos de tiempo', fontsize=12, fontweight='bold')
plt.ylabel('Potencia (W)', fontsize=12, fontweight='bold')
plt.title(f'Serie Temporal {MODELO_ACTUAL} (Últimos 1000 pasos)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'evaluacion_serie_hardware_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Histograma de Latencias
if len(lats_validas) > 0:
    plt.figure(figsize=(8, 6))
    plt.hist(lats_validas / 1000, bins=60,
             color='#43A047', edgecolor='black', lw=0.8) # Borde negro para imprimir bien
    plt.axvline(lats_validas.mean() / 1000, color='black', ls='--', lw=2.5,
                label=f'Media: {lats_validas.mean()/1000:.2f} ms')
    plt.axvline(np.percentile(lats_validas, 95) / 1000,
                color='red', ls=':', lw=2.5,
                label=f'P95: {np.percentile(lats_validas,95)/1000:.2f} ms')
    plt.xlabel('Latencia de inferencia (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Frecuencia', fontsize=12, fontweight='bold')
    plt.title(f'Distribución de Latencia en ESP32-S3 {MODELO_ACTUAL}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'evaluacion_latencia_hardware_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print('\n[OK] Gráficas generadas por separado a 300 DPI y guardadas con éxito.')