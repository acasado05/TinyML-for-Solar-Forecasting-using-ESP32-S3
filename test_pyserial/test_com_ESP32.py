import serial
import struct
import time
import random

# --- CONFIGURACIÓN ---
PORT = 'COM13'  
BAUD = 115200

# Formato Mensajes
FORMATO_TELEMETRIA = '<ffiifffff'  # Tipo 0x01 (36 bytes)
FORMATO_INPUTS = '<ffffffff'       # Tipo 0x02 (8 floats = 32 bytes)

SIZE_TELEMETRIA = struct.calcsize(FORMATO_TELEMETRIA)
SIZE_INPUTS = struct.calcsize(FORMATO_INPUTS)

SIZE_DEBUG_FILA = 33 # 1 byte índice + 32 bytes de los 8 floats

def enviar_inputs_simulados(ser):
    """Genera 8 valores aleatorios simulando los sensores y los envía al ESP32"""
    # Simulamos el set de entrada: ej. Irradiación, Temp, V_panel, I_panel, etc.
    inputs = [
        random.uniform(200.0, 1000.0), # Irradiación simulada (W/m2)
        random.uniform(15.0, 45.0),    # Temperatura celda (°C)
        random.uniform(300.0, 400.0),  # V_PV (V)
        random.uniform(0.0, 8.0),      # I_PV (A)
        random.uniform(20.0, 30.0),    # T_Ambiente (°C)
        random.uniform(30.0, 80.0),    # Humedad (%)
        random.uniform(0.04, 0.18),    # Precio KWh (€)
        random.uniform(0.0, 2000.0)    # Potencia Calculada anterior
    ]
    
    # Empaquetamos a binario Little Endian
    payload = struct.pack(FORMATO_INPUTS, *inputs)
    checksum = sum(payload) % 256
    
    # Construimos el frame completo
    frame = b'\xaa' + b'\x02' + bytes([SIZE_INPUTS]) + payload + bytes([checksum]) + b'\x55'
    
    print(f"\n[MASTER PC] Enviando 8 inputs al ESP32-S3 (Bytes totales frame: {len(frame)})...")
    ser.write(frame)

def procesar_telemetria(payload):
    """Desempaqueta los datos recibidos del ESP32-S3 (Tipo 0x01)"""
    try:
        data = struct.unpack(FORMATO_TELEMETRIA, payload)
        print("\n" + "="*40)
        print("   ECO RECIBIDO DESDE EL ESP32-S3 ")
        print("="*40)
        print(f"INVERSOR | PV: {data[0]:.1f}V / {data[1]:.2f}A")
        print(f"POTENCIA | DC: {data[2]}W  | AC: {data[3]}W")
        print(f"ENERGÍA  | Hoy: {data[4]:.2f} kWh")
        print(f"SHUNTS   | Irr: {data[5]:.6f}V | T_cel: {data[6]:.4f}V")
        print(f"AMBIENTE | T: {data[7]:.1f}°C | H: {data[8]:.1f}%")
        print("="*40)
    except Exception as e:
        print(f"Error en desempaquetado de telemetría: {e}")

def main():
    try:
        # Aumentamos el timeout a 0.5 para dar margen de lectura al USB nativo
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        print(f"Iniciando Master PC en el puerto {PORT}...")
        time.sleep(2) # Espera de cortesía para el S3

        # Limpiamos los buffers de Windows por si hay basura acumulada
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        last_envio_inputs = time.time()
        INTERVALO_ENVIO_INPUTS = 5.0 # Envío cada 5 segundos

        while True:
            # 1. Rutina de envío periódico desde el PC hacia el ESP32
            if time.time() - last_envio_inputs > INTERVALO_ENVIO_INPUTS:
                enviar_inputs_simulados(ser)
                last_envio_inputs = time.time()

            # 2. Rutina de lectura OPTIMIZADA en ráfaga
            # Comprobamos si hay bytes esperando en la cola del puerto serie
            if ser.in_waiting > 0:
                if ser.read() == b'\xaa': # Encontramos inicio de trama
                    tipo = ser.read()
                    len_recibida = ord(ser.read())
                    
                    # Leemos el payload completo + checksum (1 byte) + fin (1 byte) de golpe
                    bytes_a_leer = len_recibida + 2
                    bloque_restante = ser.read(bytes_a_leer)
                    
                    # Verificamos que hemos recibido el bloque completo sin caídas de timeout
                    if len(bloque_restante) == bytes_a_leer:
                        payload = bloque_restante[:-2]
                        checksum_recibido = bloque_restante[-2]
                        fin = bloque_restante[-1:]
                        
                        # Validamos Checksum e Integridad
                        if sum(payload) % 256 == checksum_recibido and fin == b'\x55':
                            
                            # --- TRATAMIENTO SEGÚN EL TIPO ---
                            if tipo == b'\x01':
                                procesar_telemetria(payload)
                                
                            elif tipo == b'\x03':
                                indice_fila = payload[0]
                                valores_fila = struct.unpack('<ffffffff', payload[1:])
                                print("\n" + "*"*54)
                                print(f"[OK] ¡NOTIFICACIÓN DE MEMORIA DESDE EL ESP32-S3!")
                                print(f"-> Fila [{indice_fila:02d}] escrita con éxito en el Buffer Circular.")
                                print(f"-> Inputs guardados: [" + ", ".join(f"{x:.2f}" for x in valores_fila) + "]")
                                print("*"*54)
                        else:
                            print("[ERROR INTERNO] Trama corrupta (Fallo de Checksum o Fin de trama)")
                    else:
                        print(f"[WARN] Fragmentación de paquete. Se esperaban {bytes_a_leer} bytes y llegaron {len(bloque_restante)}.")
                        
            time.sleep(0.001) # Alivio de CPU

    except serial.SerialException as e:
        print(f"Error de conexión serial: {e}")
    except KeyboardInterrupt:
        print("\nPrueba de simulacro terminada por el usuario.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()