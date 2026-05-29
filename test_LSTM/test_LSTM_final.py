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
import matplotlib.dates as mdates

# ─── Configuración ────────────────────────────────────────────────────
PORT          = 'COM14'          
BAUDRATE      = 115200
TIMEOUT       = 2
MODELO_ACTUAL = 'LSTM_cmpt'
CSV_PATH      = 'test_LSTM/datos_2sem_arreglado.csv'
SEQ_LEN       = 18
LOOK_AHEAD    = 6
FEATURES      = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos',
                    'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv']
TARGET        = 'Pot_inv'
UMBRAL_DIA    = 40.0   # W/m² — umbral G_Glob para métrica diurna

# ─── Carga y preprocesado (idéntico a V13) ────────────────────────────
df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
df.set_index('Timestamp', inplace=True)

# Variables temporales
horas = df.index.hour
meses = df.index.month
df['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
df['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
df['mes_sin']  = np.sin(meses * (2 * np.pi / 12))
df['mes_cos']  = np.cos(meses * (2 * np.pi / 12))

# ¡TODO EL ARCHIVO ES VALIDACIÓN! 
val_df = df[FEATURES]

# Preparamos los arrays crudos para enviarlos al ESP32
# (El ESP32 se encarga de normalizar usando scaler_params.h)
val_X_raw  = val_df.drop(columns=[TARGET]).values   # shape (N, 8)
val_y_raw  = val_df[TARGET].values                  # shape (N,)
val_g_raw  = val_df['G_Glob'].values                # shape (N,) para máscara diurna
val_ts     = val_df.index                           # timestamps para validar continuidad

N_VAL = len(val_df)
print(f'Conjunto de validación HIL: {N_VAL} filas (100% del dataset)')
print(f'  Desde: {val_ts[0]}')
print(f'  Hasta: {val_ts[-1]}')
print(f'  Hasta: {val_ts[-1]}')

# ─── Precalcular máscara de ventanas temporalmente válidas ────────────
# 1. Calculamos cuántos "saltos" (deltas) reales hay entre el índice de inicio 
# y el índice del target.
saltos_totales = (SEQ_LEN - 1) + (LOOK_AHEAD - 1) 

# 2. Convertimos esos saltos a tiempo teórico (a razón de 10 mins por salto)f
dt_ventana = pd.Timedelta(minutes=10 * saltos_totales)

ventana_valida = np.zeros(N_VAL, dtype=bool)
target_idx_arr = np.full(N_VAL, -1, dtype=int)

for i in range(SEQ_LEN - 1, N_VAL):
    target_idx = i + LOOK_AHEAD - 1
    if target_idx >= N_VAL:
        break
        
    ini_ventana = i - SEQ_LEN + 1
    
    # Comprobación de integridad temporal
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
        # 1. Enviar la fila con su número de secuencia
        linea = f'SEQ:{i},' + ','.join([f'{v:.6f}' for v in val_X_raw[i]])
        ser.write((linea + '\n').encode())
        ser.flush()

        # 2. LECTURA DINÁMICA (Adiós al time.sleep)
        # Nos quedamos escuchando hasta que el ESP32 hable o salte el timeout
        while True:
            respuesta = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Si la respuesta está vacía, es que han pasado los 2 segundos de timeout
            if not respuesta:
                print(f"  [ERROR] Timeout en la fila {i}. El ESP32 no respondió a tiempo.")
                break 
                
            # Si recibimos predicción, la guardamos y rompemos el bucle de escucha
            if respuesta.startswith('PRED:'):
                match = re.search(r'PRED:SEQ:(\d+):([a-zA-Z0-9.-]+),LAT:(\d+)', respuesta)
                if match:
                    seq_recv = int(match.group(1))
                    pred_str = match.group(2)
                    lat_us   = int(match.group(3))
                    
                    pred_raw = np.nan if pred_str.lower() == 'nan' else float(pred_str)
                    
                    # Guardamos la predicción en su índice exacto
                    if 0 <= seq_recv < N_VAL:
                        raw_preds[seq_recv]     = pred_raw
                        raw_latencias[seq_recv] = lat_us
                
                break # Rompemos el while True para enviar la siguiente fila
                
            # Si está llenando la ventana de los 18 pasos, también rompemos y avanzamos
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

# ─── DIAGNÓSTICO DEFINITIVO ───
# print(f"\n--- DIAGNÓSTICO DE FILTROS ---")
# print(f"1. Ventanas correctas (Fechas CSV Python): {ventana_valida.sum()}")
# print(f"2. Predicciones recibidas (Enviadas por ESP32): {mask_recibido.sum()}")
# print(f"3. Intersección final (Válidas para gráfica): {mask_final.sum()}")
# import sys; sys.exit() # Cortamos la ejecución aquí para que no salte el error
# ──────────────────────────────

# Extraer valores finales alineados
idx_validos   = np.where(mask_final)[0]
preds_final   = preds_w[idx_validos]
targets_final = val_y_raw[target_idx_arr[idx_validos]]
g_final       = val_g_raw[target_idx_arr[idx_validos]]
lats_final    = raw_latencias[idx_validos]
ts_final      = [val_ts[target_idx_arr[i]] for i in idx_validos]

print(f'\nPredicciones válidas para métricas: {len(preds_final)}')
print(f'  (de {N_VAL} filas enviadas, {mask_final.sum()} válidas)')

# ─── CÁLCULO DEL FACTOR DE REDUCCIÓN DE PLANTA (K) ───────────────
mascara_dia = g_final > UMBRAL_DIA
preds_dia_bruto = preds_final[mascara_dia]
reals_dia_bruto = targets_final[mascara_dia]

# Fórmula de mínimos cuadrados para superponer las curvas perfectamente
# (Evitamos división por cero por si la máscara está vacía)
if np.sum(preds_dia_bruto**2) > 0:
    K_opt = np.sum(reals_dia_bruto * preds_dia_bruto) / np.sum(preds_dia_bruto**2)
else:
    K_opt = 1.0

print(f"\n{'='*60}")
print(f"[AJUSTE] Factor de Derate (K) calculado: {K_opt:.4f}")
print(f"         (La planta actual rinde al {K_opt*100:.1f}% de la original)")
print(f"{'='*60}")

# ¡Aplicamos el factor K a TODAS las predicciones antes de evaluar!
preds_final = preds_final * K_opt

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
# Generar nombre de carpeta único con la fecha y hora actual
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
directorio_script = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(directorio_script, "test_LSTM", f"run_{timestamp}_{MODELO_ACTUAL}")
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

print('\nGenerando gráficas de calidad profesional (TFG)...')

# Configuraciones generales de estética para imprenta
COLOR_REAL = '#2C3E50'   # Gris Oscuro/Azul marino
COLOR_PRED = '#E67E22'   # Naranja intenso (buen contraste)
COLOR_DIA  = '#F1C40F'   # Amarillo para el sombreado de irradiancia
FONT_TITLE = 14
FONT_LABEL = 12

# 1. SERIE TEMPORAL DE CICLOS DIURNOS (Zoom del 20 al 25 de Mayo)
plt.figure(figsize=(12, 5))

ts_idx = pd.DatetimeIndex(ts_final)
mask_fechas = (ts_idx >= '2026-05-20') & (ts_idx <= '2026-05-25 23:59:59')

time_slice = ts_idx[mask_fechas]
real_slice = targets_final[mask_fechas]
pred_slice = preds_final[mask_fechas]
max_y = max(real_slice.max(), pred_slice.max()) * 1.1

plt.plot(time_slice, real_slice, color=COLOR_REAL, lw=2.5, label='Potencia Real', zorder=3)
plt.plot(time_slice, pred_slice, color=COLOR_PRED, lw=2.0, label='Predicción LSTM', alpha=0.9, zorder=4)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.xticks(rotation=0)
plt.xlabel('Fecha', fontsize=FONT_LABEL, fontweight='bold')
plt.ylabel('Potencia (W)', fontsize=FONT_LABEL, fontweight='bold')
plt.title('Dinámica Predictiva HIL (Detalle del 20 al 25 de Mayo)', fontsize=FONT_TITLE, fontweight='bold')
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.ylim(0, max_y)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'1_serie_temporal_HIL_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. DISPERSIÓN ESTRICTAMENTE DIURNA
plt.figure(figsize=(8, 7))
mx_val = max(reals_dia.max(), preds_dia.max()) * 1.05
plt.scatter(reals_dia, preds_dia, alpha=0.7, s=30, c='#3498DB', edgecolors='#154360', linewidths=0.8, zorder=2)
plt.plot([0, mx_val], [0, mx_val], 'k--', lw=2.5, label='Predicción Ideal (y = x)', zorder=3)

plt.xlabel('Potencia Real (W)', fontsize=FONT_LABEL, fontweight='bold')
plt.ylabel('Potencia Predicha (W)', fontsize=FONT_LABEL, fontweight='bold')
plt.title(f'Correlación bajo Carga (Periodo Diurno)\nMAE: {mae_d:.1f} W | R²: {r2_d:.4f}', fontsize=FONT_TITLE, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7, zorder=0)
plt.axis('equal') # Ejes proporcionales
plt.xlim(0, mx_val)
plt.ylim(0, mx_val)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'2_dispersion_diurna_HIL_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. HISTOGRAMA DE RESIDUOS (Análisis de Error)
plt.figure(figsize=(9, 5))
residuos = preds_dia - reals_dia
mean_res = np.mean(residuos)
std_res = np.std(residuos)

plt.hist(residuos, bins=60, color='#8E44AD', edgecolor='black', lw=0.5, alpha=0.8, zorder=2)
plt.axvline(0, color='black', lw=2, zorder=3)
plt.axvline(mean_res, color='#F39C12', ls='--', lw=2.5, label=f'Media: {mean_res:+.1f} W', zorder=3)
plt.axvline(mean_res + std_res, color='#E74C3C', ls=':', lw=2, label=f'Desv. Est. (±{std_res:.1f} W)', zorder=3)
plt.axvline(mean_res - std_res, color='#E74C3C', ls=':', lw=2, zorder=3)

plt.xlabel('Error Residual (W) [Predicción - Real]', fontsize=FONT_LABEL, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=FONT_LABEL, fontweight='bold')
plt.title('Distribución de Errores Diurnos (Normalidad y Sesgo)', fontsize=FONT_TITLE, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'3_residuos_HIL_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. ENERGÍA DIARIA ACUMULADA (KPI para mercado eléctrico)
df_energy = pd.DataFrame({'ts': ts_final, 'real': targets_final, 'pred': preds_final})
df_energy.set_index('ts', inplace=True)
df_daily = df_energy.resample('D').sum() * (10.0 / 60.0) / 1000.0

# Quitamos días que estén a 0 o casi vacíos por seguridad
df_daily = df_daily[df_daily['real'] > 0.1] 

plt.figure(figsize=(12, 5))
x_labels = df_daily.index.strftime('%d-%b')
x_pos = np.arange(len(df_daily))
width = 0.35

plt.bar(x_pos - width/2, df_daily['real'], width, label='Real (Medidor)', color=COLOR_REAL, zorder=3)
plt.bar(x_pos + width/2, df_daily['pred'], width, label='Predicho (LSTM)', color=COLOR_PRED, zorder=3)

plt.xlabel('Día de evaluación', fontsize=FONT_LABEL, fontweight='bold')
plt.ylabel('Energía Total (kWh)', fontsize=FONT_LABEL, fontweight='bold')
plt.title('Comparativa de Producción de Energía Diaria Acumulada (2 Semanas)', fontsize=FONT_TITLE, fontweight='bold')
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend(fontsize=11)
plt.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'4_energia_diaria_HIL_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. HISTOGRAMA DE LATENCIA (Rendimiento Hardware)
if len(lats_validas) > 0:
    plt.figure(figsize=(9, 5))
    plt.hist(lats_validas / 1000, bins=50, color='#27AE60', edgecolor='black', lw=0.8, alpha=0.85, zorder=2)
    media_lat = lats_validas.mean() / 1000
    p95_lat = np.percentile(lats_validas, 95) / 1000
    
    plt.axvline(media_lat, color='black', ls='--', lw=2.5, label=f'Media: {media_lat:.2f} ms', zorder=3)
    plt.axvline(p95_lat, color='#C0392B', ls=':', lw=2.5, label=f'P95: {p95_lat:.2f} ms', zorder=3)
    
    plt.xlabel('Tiempo de Inferencia (ms)', fontsize=FONT_LABEL, fontweight='bold')
    plt.ylabel('Cantidad de Inferencias', fontsize=FONT_LABEL, fontweight='bold')
    plt.title('Perfil de Rendimiento HIL en ESP32-S3', fontsize=FONT_TITLE, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'5_latencia_HIL_{MODELO_ACTUAL}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f'\n[¡ÉXITO!] 5 Gráficas profesionales de validación exportadas a:\n{out_dir}')