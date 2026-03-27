# Instalar las librerías necesarias
"""
python -m venv _Entorno
source ./_Entorno/bin/activate 
pip install mysql-connector-python kafka-python mysql-replication
"""
import mysql.connector
import time
print("Conectando a MySQL y preparando la base de datos...")
try:
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="iot_db"
    )
except mysql.connector.Error as err:
    print(f"Error al conectar a MySQL (has esperado?):\n\n")
    exit(1)

cursor = conexion.cursor()
print("Creando la tabla 'lecturas' si no existe...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS lecturas (
        id INT AUTO_INCREMENT PRIMARY KEY,
        sensor_id INT,
        temperatura FLOAT,
        timestamp DATETIME
    )
""")
conexion.commit()
conexion.close()
print("Tabla 'lecturas' creada con éxito y control de cambios (Binlog) operativo en el servidor.")