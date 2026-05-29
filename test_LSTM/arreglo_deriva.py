import pandas as pd

print("Cargando archivo original...")

nombre_entrada = 'test_LSTM/datos_2sem.csv' 
df = pd.read_csv(nombre_entrada, sep=';', decimal=',')

# 1. Cargamos tu dataset original 
# (Prueba primero con punto y coma, si falla, usa coma estándar)
try:
    df = pd.read_csv(nombre_entrada, sep=';', decimal=',')
except:
    df = pd.read_csv(nombre_entrada, sep=',', decimal='.')

# 2. Convertimos la columna a formato fecha de Pandas
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
except:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 3. Redondear cada tiempo a su bloque de 10 minutos más cercano
df['Timestamp'] = df['Timestamp'].dt.round('10min')

# 4. Volver a convertir al formato de texto que espera tu script ESP32 principal
df['Timestamp'] = df['Timestamp'].dt.strftime('%d/%m/%Y %H:%M')

# 5. Guardar el nuevo archivo limpio y listo para usar
nombre_salida = 'test_LSTM/datos_2sem_arreglado.csv'
df.to_csv(nombre_salida, sep=';', decimal=',', index=False)

print(f"[ÉXITO] Archivo corregido y guardado como: {nombre_salida}")