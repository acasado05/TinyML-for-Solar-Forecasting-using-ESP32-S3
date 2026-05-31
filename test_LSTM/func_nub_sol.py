import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

print("Cargando resultados de inferencia en hardware...")

# 1. Cargar el dataset de resultados
csv_path = 'test_LSTM/resultados_hardware.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"[ERROR] No se encuentra '{csv_path}'. Asegúrate de ejecutar este script en la misma carpeta.")
    exit()

# 2. Convertir la columna timestamp a tipo fecha y establecerla como índice
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ─── Configuración estética para imprenta (TFG) ───────────────────────
COLOR_REAL = '#2C3E50'   # Gris Oscuro/Azul marino
COLOR_PRED = '#E67E22'   # Naranja intenso (buen contraste)
FONT_TITLE = 14
FONT_LABEL = 12

# Función auxiliar para generar las gráficas con el mismo formato
def plot_dia_comparativa(df_dia, fecha_str, titulo, nombre_archivo):
    plt.figure(figsize=(10, 5))
    
    # Extraemos los datos del día
    real_slice = df_dia['real_W']
    pred_slice = df_dia['pred_W']
    
    # Calculamos el máximo para dar un poco de margen al techo de la gráfica
    max_y = max(real_slice.max(), pred_slice.max()) * 1.1
    
    # Dibujamos las líneas
    plt.plot(df_dia.index, real_slice, color=COLOR_REAL, lw=2.5, label='Potencia Real', zorder=3)
    plt.plot(df_dia.index, pred_slice, color=COLOR_PRED, lw=2.0, label='Predicción LSTM', alpha=0.9, zorder=4)

    # Formato del eje X (Mostramos solo la hora y el minuto)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Cuadriculamos el eje X cada 2 horas para que quede limpio
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=0)
    
    # Etiquetas y Leyenda
    plt.xlabel(f'Hora del día ({fecha_str})', fontsize=FONT_LABEL, fontweight='bold')
    plt.ylabel('Potencia (W)', fontsize=FONT_LABEL, fontweight='bold')
    plt.title(titulo, fontsize=FONT_TITLE, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    plt.ylim(0, max_y)
    plt.tight_layout()
    
    # Guardamos la gráfica
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Gráfica guardada: {nombre_archivo}")

# ─── Generación de las 2 gráficas de estudio ──────────────────────────

# Caso 1: Día Nublado (15 de mayo) - Alta volatilidad
df_nublado = df.loc['2026-05-15']
plot_dia_comparativa(
    df_dia = df_nublado, 
    fecha_str = '15 de Mayo de 2026', 
    titulo = 'Seguimiento Predictivo en Día Nublado (Alta Volatilidad)', 
    nombre_archivo = 'caso_estudio_nublado.png'
)

# Caso 2: Día Soleado (21 de mayo) - Perfil de cielo despejado ideal
df_soleado = df.loc['2026-05-21']
plot_dia_comparativa(
    df_dia = df_soleado, 
    fecha_str = '21 de Mayo de 2026', 
    titulo = 'Seguimiento Predictivo en Día Soleado (Perfil Ideal)', 
    nombre_archivo = 'caso_estudio_soleado.png'
)

print("\n¡Proceso completado! Ya puedes añadir las imágenes a tu memoria.")