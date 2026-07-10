import serial
import time
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ─── Configuración ────────────────────────────────────────────────────
PORT          = 'COM14'          
BAUDRATE      = 115200
TIMEOUT       = 2
MODELO_ACTUAL = 'LSTM_cmpt'  # Cuando evalue GRU, poner GRU
CSV_PATH      = 'train_LSTM/datos_10min_4549W_horabuena.csv'
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

# Imputar NaN en val_X_raw con la media de cada columna del conjunto de entrenamiento
col_means = np.nanmean(train_df.drop(columns=[TARGET]).values, axis=0)
for col_idx in range(val_X_raw.shape[1]):
    nan_mask = np.isnan(val_X_raw[:, col_idx])
    if nan_mask.sum() > 0:
        print(f"  [FIX] Imputando {nan_mask.sum()} NaN en columna {col_idx} "
              f"({val_df.drop(columns=[TARGET]).columns[col_idx]}) "
              f"con media={col_means[col_idx]:.4f}")
        val_X_raw[nan_mask, col_idx] = col_means[col_idx]

# Verificar que no queden NaN
assert not np.isnan(val_X_raw).any(), "[ERROR] Siguen habiendo NaN tras imputación"
print("[OK] val_X_raw sin NaN")

# ─── Precalcular máscara de ventanas temporalmente válidas ────────────
# CORRECCIÓN 1: Ajuste del desfase temporal ("Off-by-one error")
saltos_totales = (SEQ_LEN - 1) + LOOK_AHEAD 
dt_ventana = pd.Timedelta(minutes=10 * saltos_totales)

ventana_valida = np.zeros(N_VAL, dtype=bool)
target_idx_arr = np.full(N_VAL, -1, dtype=int)

for i in range(SEQ_LEN - 1, N_VAL):
    target_idx = i + LOOK_AHEAD
    if target_idx >= N_VAL:
        break
        
    ini_ventana = i - SEQ_LEN + 1
    
    # Comprobación de integridad temporal
    if (val_ts[target_idx] - val_ts[ini_ventana]) == dt_ventana:
        ventana_valida[i]    = True
        target_idx_arr[i]    = target_idx

n_validas = ventana_valida.sum()
n_total   = (N_VAL - SEQ_LEN - LOOK_AHEAD + 1)
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
            print(f'  [ESP32 LOG] {respuesta}')

if not conexion_ok:
    print('\n[FATAL] El ESP32 no ha respondido al PING.')
    print('Posibles causas:')
    print(' 1. TENSOR_ARENA_SIZE es muy pequeño y hay un bootloop (fíjate en los logs arriba).')
    print(' 2. El firmware no se ha subido correctamente.')
    ser.close()
    exit()

print('  [OK] Respuesta PONG recibida. Enlace establecido.\n')

# ─── Enviar TODAS las filas y recoger respuestas ──────────────────────
raw_preds     = np.full(N_VAL, np.nan)   
raw_latencias = np.full(N_VAL, -1, dtype=int)

print(f'Enviando {N_VAL} filas al ESP32...')
t_inicio_total = time.time()

for i in range(N_VAL):
        # 1. Enviar la fila con su número de secuencia
        linea = f'SEQ:{i},' + ','.join([f'{v:.6f}' for v in val_X_raw[i]])
        ser.write((linea + '\n').encode())
        ser.flush()

        # 2. LECTURA DINÁMICA
        while True:
            respuesta = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if not respuesta:
                print(f"  [ERROR] Timeout en la fila {i}. El ESP32 no respondió a tiempo.")
                break 
                
            if respuesta.startswith('PRED:'):
                match = re.search(r'PRED:SEQ:(\d+):([a-zA-Z0-9.-]+),LAT:(\d+)', respuesta)
                if match:
                    seq_recv = int(match.group(1))
                    pred_str = match.group(2)
                    lat_us   = int(match.group(3))
                    
                    pred_raw = np.nan if pred_str.lower() == 'nan' else float(pred_str)
                    
                    if 0 <= seq_recv < N_VAL:
                        raw_preds[seq_recv]     = pred_raw
                        raw_latencias[seq_recv] = lat_us
                
                break 
                
            elif respuesta.startswith('BUFFERING'):
                break 

        # 4. Imprimir progreso
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_inicio_total
            pct     = 100 * (i + 1) / N_VAL
            print(f'  {i+1}/{N_VAL} ({pct:.0f}%) — {elapsed:.0f}s transcurridos')

ser.close()
t_total = time.time() - t_inicio_total
print(f'\nEnvío completado en {t_total:.1f} s '
      f'({t_total/N_VAL*1000:.1f} ms por fila)')

# ─── Desescalar predicciones ─────────────────────────────────────────
# CORRECCIÓN 2: Aplicación del inverse_transform para tener los datos en W
preds_w = raw_preds.copy()
mask_recibido = ~np.isnan(preds_w)

# idx_recibidos = np.where(mask_recibido)[0]
# if len(idx_recibidos) > 0:
#     # Desescalamos usando el scaler que entrenamos al principio
#     preds_w[idx_recibidos] = scaler_y.inverse_transform(preds_w[idx_recibidos].reshape(-1, 1)).flatten()

preds_w = np.maximum(preds_w, 0)   # clipping físico (nunca tendremos potencia negativa)

# ─── Aplicar máscara de ventanas válidas ──────────────────────────────
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
mascara_dia = g_final > UMBRAL_DIA
preds_dia   = preds_final[mascara_dia]
reals_dia   = targets_final[mascara_dia]

mae_d  = mean_absolute_error(reals_dia, preds_dia)
rmse_d = np.sqrt(mean_squared_error(reals_dia, preds_dia))
r2_d   = r2_score(reals_dia, preds_dia)
n_d    = mascara_dia.sum()
r2a_d  = 1 - (1 - r2_d) * (n_d - 1) / (n_d - p_c - 1)

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

print(f'\n  MÉTRICAS DIURNAS (G_Glob > {UMBRAL_DIA} W/m²) - {n_d} muestras:')
print(f'    MAE   = {mae_d:.3f} W')
print(f'    RMSE  = {rmse_d:.3f} W')
print(f'    R²    = {r2_d:.4f}')
print(f'    R² Aj.= {r2a_d:.4f}')
# ──────────────────────────────────────────────────────────────────────

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
V13_REF  = V13_LSTM   # Cambiar a V13_LSTM en caso de evaluar la otra

print(f'\n  DEGRADACIÓN POR CUANTIZACIÓN (vs V13 Python Float32):')
print(f'    ΔMAE  completo  = {mae_c  - V13_REF["mae_c"]:+.1f} W')
print(f'    ΔR²   completo  = {r2_c   - V13_REF["r2_c"]:+.4f}')

# ─── Guardar resultados CSV ───────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("train_LSTM/Pmax_4549W/eval_emb_lstm", f"run_{timestamp}_{MODELO_ACTUAL}")
os.makedirs(out_dir, exist_ok=True)

df_out = pd.DataFrame({
    'timestamp':     ts_final,
    'real_W':        targets_final,
    'pred_W':        preds_final,
    'G_Glob':        g_final,
    'latencia_us':   lats_final,
})

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
             color='#43A047', edgecolor='black', lw=0.8)
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

# 1. GRÁFICA DE JUSTIFICACIÓN DEL UMBRAL (Irradiancia)
plt.figure(figsize=(10, 4))
tramo = 400 
plt.plot(g_final[:tramo], color='#F39C12', lw=2, label='Irradiancia Global (G_Glob)')
plt.axhline(y=UMBRAL_DIA, color='red', linestyle='--', lw=2, 
            label=f'Umbral de Máscara ({UMBRAL_DIA} W/m²)')
plt.fill_between(range(tramo), 0, UMBRAL_DIA, color='red', alpha=0.1, label='Zona descartada (Noche)')

plt.xlabel('Pasos de tiempo (Tramo de 3 días)', fontsize=12, fontweight='bold')
plt.ylabel('Irradiancia (W/m²)', fontsize=12, fontweight='bold')
plt.title('Justificación Física de la Máscara Diurna', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'justificacion_umbral_G_Glob.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. COMPARATIVA DE DISPERSIÓN (El engaño de la noche)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
mx_val = max(targets_final.max(), preds_final.max()) * 1.05

# Izquierda: 24 Horas Completas
ax1.scatter(targets_final, preds_final, alpha=0.3, s=15, color='#2E5090')
ax1.plot([0, mx_val], [0, mx_val], 'k--', lw=2)
ax1.set_title(f'Evaluación Completa (24h)\nMancha artificial en (0,0)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Potencia Real (W)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Potencia Predicha (W)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.5)

# Derecha: Solo Día (Máscara Aplicada)
mascara_dia = g_final > UMBRAL_DIA
ax2.scatter(targets_final[mascara_dia], preds_final[mascara_dia], alpha=0.3, s=15, color='#D35400')
ax2.plot([0, mx_val], [0, mx_val], 'k--', lw=2)
ax2.set_title(f'Evaluación Estricta Diurna (G > {UMBRAL_DIA} W/m²)\nDispersión real bajo carga', fontsize=13, fontweight='bold')
ax2.set_xlabel('Potencia Real (W)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Potencia Predicha (W)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.5)

plt.suptitle('Efecto del Filtrado Nocturno en las Métricas del Modelo', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'comparativa_dispersion_mascara.png'), dpi=300, bbox_inches='tight')
plt.close()

print('\n[OK] Gráficas de justificación para la memoria generadas.')