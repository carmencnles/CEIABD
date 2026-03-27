from kafka import KafkaConsumer
import mysql.connector
import json

# Conexión a MySQL
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="iot_db"
)
cursor = conexion.cursor()

# Configuración del consumidor Kafka
consumer = KafkaConsumer(
    'topico_sensores',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='grupo-bd',
    auto_offset_reset='earliest' 
)

print("Escuchando eventos en Kafka y guardando en MySQL...")

try:
    for mensaje in consumer:
        evento = mensaje.value
        
        # Insertamos en la BD
        sql = "INSERT INTO lecturas (sensor_id, temperatura, timestamp) VALUES (%s, %s, %s)"
        valores = (evento['sensor_id'], evento['temperatura'], evento['timestamp'])
        
        cursor.execute(sql, valores)
        conexion.commit()
        
        print(f"[Guardado en BD] Sensor {evento['sensor_id']} | Temp: {evento['temperatura']}ºC")
        
except KeyboardInterrupt:
    print("Consumo detenido.")
finally:
    cursor.close()
    conexion.close()
    consumer.close()


