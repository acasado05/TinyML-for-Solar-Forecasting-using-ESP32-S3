import serial
import struct
import time

# --- Configuración ---
PORT = 'COM9'
BAUD = 115200
# Formato: < (Little Endian), f (float), f (float), i (int32), i (int32), f...f (5 floats más)
# Total: ffiiifffff (9 campos)
FORMATO_TRAMA = '<ffiifffff' 
TAMANO_PAYLOAD = struct.calcsize(FORMATO_TRAMA)

def procesar_datos(payload):
    try:
        # Desempaquetamos los 36 bytes en las 9 variables
        data = struct.unpack(FORMATO_TRAMA, payload)
        
        print("\n" + "="*40)
        print("      DATOS RECIBIDOS (CHECKSUM OK) ")
        print("="*40)
        print(f"INVERSOR | PV: {data[0]:.1f}V / {data[1]:.2f}A")
        print(f"POTENCIA | DC: {data[2]}W  | AC: {data[3]}W")
        print(f"ENERGÍA  | Hoy: {data[4]:.2f} kWh")
        print(f"SHUNTS   | Irr: {data[5]:.6f}V | T_cel: {data[6]:.4f}V")
        print(f"AMBIENTE | T: {data[7]:.1f}°C | H: {data[8]:.1f}%")
        print("="*40)
        
    except Exception as e:
        print(f"Error en desempaquetado: {e}")

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Escuchando en {PORT}...")
        time.sleep(2) # Espera reset ESP32

        while True:
            # Buscamos el byte de inicio 0xAA
            if ser.read() == b'\xaa':
                tipo = ser.read()
                len_recibida = ord(ser.read())
                
                if len_recibida == TAMANO_PAYLOAD:
                    payload = ser.read(len_recibida)
                    checksum_recibido = ord(ser.read())
                    fin = ser.read()
                    
                    # Validación de integridad
                    if sum(payload) % 256 == checksum_recibido and fin == b'\x55':
                        procesar_datos(payload)
                    else:
                        print("Advertencia: Trama corrupta o Checksum inválido")
                else:
                    print(f"Error: Longitud inesperada ({len_recibida})")

    except serial.SerialException as e:
        print(f"Error de conexión: {e}")
    except KeyboardInterrupt:
        print("\nCerrando programa...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()