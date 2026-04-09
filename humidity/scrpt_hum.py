import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# CONFIGURACIÓN DE ARCHIVOS Y COLUMNAS
# =========================================================
# Obtiene la carpeta donde está el script actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Une la carpeta con el nombre del archivo
archivo_entrada = os.path.join(directorio_actual, 'Hum_Rel_PVGIS.csv')
archivo_salida = os.path.join(directorio_actual, 'Humedad_10min_Final.csv')

columna_tiempo = 'time(UTC)'
columna_humedad = 'RH'

def procesar_humedad():
    try:
        print(f"Cargando datos desde {archivo_entrada}...")
        
        # 1. CARGA DE DATOS
        # sep=';' porque Excel en español usa punto y coma
        # decimal=',' porque en tu captura se veían comas (90,89)
        df = pd.read_csv(archivo_entrada, sep=';', decimal=',')

        # 2. PROCESAMIENTO DE FECHAS
        print("Transformando formato de tiempo YYYYMMDD:HHMM...")
        # Convertimos al formato que entiende Pandas
        df[columna_tiempo] = pd.to_datetime(df[columna_tiempo], format='%Y%m%d:%H%M')
        
        # Ajustamos el año a 2025 para que coincida con tus datos del inversor
        df[columna_tiempo] = df[columna_tiempo].apply(lambda x: x.replace(year=2025))
        
        # Establecemos la fecha como índice para poder interpolar
        df.set_index(columna_tiempo, inplace=True)

        # 3. INTERPOLACIÓN (Upsampling)
        print("Interpolando datos de 1 hora a 10 minutos...")
        # '10min' crea los huecos, .interpolate(method='linear') los rellena con una línea
        df_10min = df[[columna_humedad]].resample('10min').interpolate(method='linear')

        # 4. GUARDADO DEL NUEVO ARCHIVO
        # Guardamos con el mismo formato compatible con Excel (punto y coma)
        df_10min.to_csv(archivo_salida, sep=';', decimal=',', index=True)
        
        print("-" * 50)
        print(f"✅ ¡PROCESO EXITOSO!")
        print(f"Archivo guardado como: {archivo_salida}")
        print(f"Número de filas original: {len(df)}")
        print(f"Número de filas final:    {len(df_10min)}")
        print("-" * 50)

        # 5. VISUALIZACIÓN DE COMPROBACIÓN
        # Graficamos los dos primeros días (288 intervalos de 10 min)
        plt.figure(figsize=(12, 5))
        plt.plot(df_10min.index[:288], df_10min[columna_humedad][:288], color='dodgerblue', label='Humedad Interpolada (10m)')
        plt.scatter(df.index[:48], df[columna_humedad][:48], color='red', s=20, label='Puntos originales (1h)')
        
        plt.title('Validación de Interpolación de Humedad Relativa')
        plt.xlabel('Tiempo')
        plt.ylabel('Humedad (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        print("\nConsejos de depuración:")
        print("1. Verifica que el nombre del archivo .csv sea exacto.")
        print("2. Abre el CSV con el Bloc de Notas. Si ves comas en vez de puntos y comas, cambia sep=';' por sep=',' en el código.")

if __name__ == "__main__":
    procesar_humedad()