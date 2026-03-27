from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Iniciando la emisión de eventos a Kafka...")

try:
    while True:
        evento = {
            'sensor_id': random.randint(1, 5),
            'temperatura': round(random.uniform(15.0, 40.0), 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        producer.send('topico_sensores', value=evento)
        print(f"[Enviado a Kafka] {evento}")
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Emisión detenida.")
finally:
    producer.close()
