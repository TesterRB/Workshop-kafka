from kafka import KafkaConsumer
from sqlalchemy import create_engine
import pandas as pd
import json
import time

KAFKA_TOPIC = "happiness_topic"
KAFKA_SERVER = "localhost:9092"

MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "happiness_dw"

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_SERVER],
    auto_offset_reset='earliest',  # comienza desde el primer mensaje disponible
    enable_auto_commit=True,
    group_id="happiness_group",
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

records = []
timeout = 5  # segundos sin mensajes tras el último recibido
first_message_received = False
last_message_time = None

print("📡 Esperando mensajes del topic...")

while True:
    msg_pack = consumer.poll(timeout_ms=1000)  # consulta cada segundo

    if msg_pack:
        for tp, messages in msg_pack.items():
            for message in messages:
                # Primer mensaje recibido → inicia el contador
                if not first_message_received:
                    first_message_received = True
                    last_message_time = time.time()

                records.append(message.value)
                last_message_time = time.time()

                # Guardar en MySQL cada 100 registros
                if len(records) % 100 == 0:
                    df = pd.DataFrame(records)
                    df.to_sql("happiness_predictions", engine,
                              if_exists="append", index=False)
                    print(f"✅ {len(records)} registros insertados en MySQL...")
                    records = []
    else:
        # Si ya empezó a recibir y pasa el timeout sin mensajes nuevos → cerrar
        if first_message_received and (time.time() - last_message_time > timeout):
            break

# Guardar últimos registros pendientes
if records:
    df = pd.DataFrame(records)
    df.to_sql("happiness_predictions", engine, if_exists="append", index=False)
    print(f"✅ Últimos {len(records)} registros insertados en MySQL.")

print("🎯 Proceso de consumo finalizado automáticamente.")
