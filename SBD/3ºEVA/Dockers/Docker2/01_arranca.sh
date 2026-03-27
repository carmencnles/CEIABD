
#Ejecuta este scrip como "source 01_arranca.sh" para arrancar el entorno de Docker y activar el entorno virtual de Python
#Activamos el entorno virtual de Python
source ./_Entorno/bin/activate


#Arranca el entorno de Docker con Kafka y MySQL
docker compose up -d


