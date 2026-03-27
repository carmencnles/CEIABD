from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)

# Datos de conexión apuntando a tu contenedor Docker expuesto en localhost
MYSQL_SETTINGS = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "root"
}

def analizar_binlog():
    # Iniciamos el lector del binlog
    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=100,  # Debe ser distinto al server-id de tu MySQL (que es 1)
        blocking=True, # False: lee el historial y termina. True: se queda escuchando en tiempo real.
        only_events=[DeleteRowsEvent, WriteRowsEvent, UpdateRowsEvent],
        only_schemas=['iot_db'],
        only_tables=['lecturas']
    )
    print("Analizando cambios en la tabla 'lecturas' (iot_db)...")
    print("-" * 60)
    # Recorremos cada evento capturado en el binlog
    for binlogevent in stream:
        for row in binlogevent.rows:
            if isinstance(binlogevent, WriteRowsEvent):
                print(f"🟢 [INSERCIÓN] Nuevos datos:")
                print(f"   {row['values']}")
                
            elif isinstance(binlogevent, UpdateRowsEvent):
                print(f"🟡 [ACTUALIZACIÓN]")
                print(f"   Antes:   {row['before_values']}")
                print(f"   Después: {row['after_values']}")
                
            elif isinstance(binlogevent, DeleteRowsEvent):
                print(f"🔴 [BORRADO] Fila eliminada:")
                print(f"   {row['values']}")
                
            print("-" * 60)
    stream.close()
if __name__ == "__main__":
    analizar_binlog()