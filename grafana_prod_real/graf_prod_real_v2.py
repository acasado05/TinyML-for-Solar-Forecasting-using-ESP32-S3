import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
# 2. CARGA Y PREPROCESADO DE DATOS (ROBUSTO)
# ==========================================
print("Cargando el dataset...")

df = pd.read_csv('grafana_prod_real/Conjunto.csv', sep=';', encoding='utf-8-sig')
df.columns = df.columns.str.strip()

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
df = df.dropna(subset=['Timestamp'])

df['Potencia AC'] = pd.to_numeric(df['Potencia AC'].astype(str).str.replace(',', '.'), errors='coerce')
df['Predicción IA'] = pd.to_numeric(df['Predicción IA'].astype(str).str.replace(',', '.'), errors='coerce')

df.ffill(inplace=True)
df.fillna(0, inplace=True)

# ==========================================
# 3. DIVISIÓN DEL DATASET (2 TRAMOS DE 10 DÍAS)
# ==========================================
fecha_corte = pd.to_datetime('19/06/2026 00:00', dayfirst=True)

df_parte1 = df[df['Timestamp'] < fecha_corte]
df_parte2 = df[df['Timestamp'] >= fecha_corte]

# ==========================================
# 4. CREACIÓN DE LA GRÁFICA
# ==========================================
print("Generando la gráfica con marcadores y línea discontinua...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

def plot_tramo(ax, data, title):

    ax.fill_between(data['Timestamp'], data['Potencia AC'], 
                    alpha=0.3, color='#1f77b4', label='Potencia Real AC (1 min)')
    ax.plot(data['Timestamp'], data['Potencia AC'], 
            color='#1f77b4', linewidth=1.2)
    
    ax.plot(data['Timestamp'], data['Predicción IA'], 
            color='#d62728', linewidth=1.5, linestyle='--', 
            marker='o', markersize=3.5, markevery=10, 
            label='Predicción IA (10 min)')
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel('Potencia (W)', fontweight='bold')
    
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('center')

    ax.legend(loc='upper left', ncol=2, framealpha=1.0, edgecolor='black')

# Aplicar la función a las dos subgráficas
plot_tramo(ax1, df_parte1, 'Comparativa Potencia Real vs. Predicción HIL (09 Jun - 18 Jun)')
plot_tramo(ax2, df_parte2, 'Comparativa Potencia Real vs. Predicción HIL (19 Jun - 28 Jun)')

# Ajustar márgenes
plt.tight_layout()
plt.subplots_adjust(hspace=0.35)

# ==========================================
# 5. GUARDAR Y MOSTRAR RESULTADO
# ==========================================
nombre_archivo = 'grafana_prod_real/Resultados_Prediccion_TFG_v2.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"¡Éxito! Gráfica guardada como '{nombre_archivo}'.")