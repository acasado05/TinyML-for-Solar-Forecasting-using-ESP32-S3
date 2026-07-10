import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CONFIGURACIÓN DE ESTILO PARA EL TFG
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,        
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# ==========================================
# 2. CARGA Y PREPROCESADO
# ==========================================
print("Cargando el dataset...")

df = pd.read_csv('grafana_prod_real/Conjunto.csv', sep=';', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
df = df.dropna(subset=['Timestamp'])

df['Potencia AC'] = pd.to_numeric(df['Potencia AC'].astype(str).str.replace(',', '.'), errors='coerce')
df['Predicción IA'] = pd.to_numeric(df['Predicción IA'].astype(str).str.replace(',', '.'), errors='coerce')

# --- PREPARACIÓN DE DATOS ---
df_real = df[['Timestamp', 'Potencia AC']].copy()

# Remuestreo de la predicción
df_pred = df[['Timestamp', 'Predicción IA']].set_index('Timestamp').resample('10min').max().reset_index()
df_pred.fillna(0, inplace=True)

# ==========================================
# 3. ALINEACIÓN TEMPORAL (SHIFT DE 1 HORA)
# ==========================================
# Como la predicción en el instante T es para T+1h, sumamos 1 hora a sus marcas de tiempo
df_pred['Timestamp'] = df_pred['Timestamp'] + pd.Timedelta(hours=1)

# ==========================================
# 4. CÁLCULO DE MÉTRICAS 
# ==========================================
# Hacemos un cruce (merge) para comparar estrictamente cada instante t con su homólogo
df_merged = pd.merge(df_pred, df_real, on='Timestamp', how='inner')

y_true = df_merged['Potencia AC']
y_pred = df_merged['Predicción IA']

# Métrica Completa
mae_comp = mean_absolute_error(y_true, y_pred)
rmse_comp = np.sqrt(mean_squared_error(y_true, y_pred))
r2_comp = r2_score(y_true, y_pred)

# Métrica Diurna (Potencia equivalente a G > 70 W/m2 aprox. > 3.5W para panel 50W)
df_diurno = df_merged[df_merged['Potencia AC'] > 3.5]
y_true_diur = df_diurno['Potencia AC']
y_pred_diur = df_diurno['Predicción IA']

mae_diur = mean_absolute_error(y_true_diur, y_pred_diur)
rmse_diur = np.sqrt(mean_squared_error(y_true_diur, y_pred_diur))
r2_diur = r2_score(y_true_diur, y_pred_diur)

print("\n--- RESULTADOS MÉTRICAS TFG ---")
print(f"[Contexto Completo 24h] -> MAE: {mae_comp:.2f} W | RMSE: {rmse_comp:.2f} W | R2: {r2_comp:.4f}")
print(f"[Contexto Diurno]       -> MAE: {mae_diur:.2f} W | RMSE: {rmse_diur:.2f} W | R2: {r2_diur:.4f}")
print("-------------------------------\n")

# ==========================================
# 5. DIVISIÓN PARA GRÁFICAS (2 TRAMOS DE 10 DÍAS)
# ==========================================
fecha_corte = pd.to_datetime('19/06/2026 00:00', dayfirst=True)

df_real_p1 = df_real[df_real['Timestamp'] < fecha_corte]
df_real_p2 = df_real[df_real['Timestamp'] >= fecha_corte]

df_pred_p1 = df_pred[df_pred['Timestamp'] < fecha_corte]
df_pred_p2 = df_pred[df_pred['Timestamp'] >= fecha_corte]

# ==========================================
# 6. CREACIÓN DE LA GRÁFICA
# ==========================================
print("Generando la gráfica alineada...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

def plot_tramo(ax, d_real, d_pred, title):
    ax.fill_between(d_real['Timestamp'], d_real['Potencia AC'], alpha=0.3, color='#1f77b4', label='Potencia Real AC (1 min)')
    ax.plot(d_real['Timestamp'], d_real['Potencia AC'], color='#1f77b4', linewidth=1.2)
    
    # La línea roja ahora está dibujada sobre el instante real para el que se predijo
    ax.plot(d_pred['Timestamp'], d_pred['Predicción IA'], color='#d62728', linewidth=1.2, linestyle='-', label='Predicción IA (+1h vista)')
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel('Potencia (W)', fontweight='bold')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('center')

    ax.legend(loc='upper left', ncol=2, framealpha=1.0, edgecolor='black')

plot_tramo(ax1, df_real_p1, df_pred_p1, 'Comparativa Potencia Real vs. Predicción IA (09 Jun - 18 Jun)')
plot_tramo(ax2, df_real_p2, df_pred_p2, 'Comparativa Potencia Real vs. Predicción IA (19 Jun - 28 Jun)')

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)

nombre_archivo = 'grafana_prod_real/Resultados_Alineados_TFG_v1.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"¡Éxito! Gráfica guardada como '{nombre_archivo}'.")