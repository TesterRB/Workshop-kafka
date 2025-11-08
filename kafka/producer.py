import pandas as pd
import joblib
import json
import time
from kafka import KafkaProducer


model = joblib.load("../model/OLS_Model.pkl")
df = pd.read_csv("../data/processed/happiness_final.csv")

# Variables predictoras usadas en el modelo
features = ["GDP_per_Capita", "Social_Support", "Life_Expectancy",
            "Freedom", "Government_Corruption", "Generosity", "Year"]

# Hacer predicciones
df["Happiness_Score_Predicted"] = model.predict(df[features])


producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

topic = "happiness_topic"

for _, row in df.iterrows():
    message = row.to_dict()
    producer.send(topic, value=message)
    time.sleep(0.01)

producer.flush()
print(f"✅ Envío completo: {len(df)} registros enviados al topic '{topic}'.")
