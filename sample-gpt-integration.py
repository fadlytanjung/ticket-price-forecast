# Full RAG-GPT Orchestration with Model Invocation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os, json
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer

# === Load and prepare data ===
df = pd.read_csv("ticket_price_history.csv")
df['depart_date'] = pd.to_datetime(df['depart_date'])
df['extract_date'] = pd.to_datetime(df['extract_date'])
df['route'] = df['origin'] + "-" + df['destination']
df['days_until_departure'] = (df['depart_date'] - df['extract_date']).dt.days
df = df.dropna(subset=['best_price', 'distance_km', 'flight_time_hour'])
le_route = LabelEncoder()
df['route_encoded'] = le_route.fit_transform(df['route'])

# === RandomForest Model ===
features = ['distance_km', 'flight_time_hour', 'days_until_departure', 'route_encoded']
target = 'best_price'
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def predict_price_rf(origin, destination, depart_date_str):
    route = f"{origin}-{destination}"
    today = pd.Timestamp.today().normalize()
    depart_date = pd.to_datetime(depart_date_str)
    days_until_departure = (depart_date - today).days
    if route not in le_route.classes_:
        route_encoded = 0; distance_km = 1000.0; flight_time_hour = 2.0
    else:
        route_encoded = le_route.transform([route])[0]
        route_info = df[df['route'] == route].mean()
        distance_km, flight_time_hour = route_info['distance_km'], route_info['flight_time_hour']
    X_pred = pd.DataFrame([{ "distance_km": distance_km, "flight_time_hour": flight_time_hour, "days_until_departure": days_until_departure, "route_encoded": route_encoded }])
    return rf_model.predict(X_pred)[0]

# === TFT Setup ===
scaler_price = MinMaxScaler()
df['scaled_price'] = scaler_price.fit_transform(df[['best_price']])
max_encoder_length = 30
max_prediction_length = 7

tft_dataset = TimeSeriesDataSet(
    df.rename(columns={"depart_date": "time_idx"}),
    time_idx="time_idx",
    target="scaled_price",
    group_ids=["route"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["scaled_price"],
    time_varying_known_reals=["days_until_departure", "distance_km", "flight_time_hour"],
    static_categoricals=["route_encoded"]
)

train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64)
tft_model = TemporalFusionTransformer.from_dataset(
    tft_dataset, learning_rate=0.03, hidden_size=16, attention_head_size=1,
    dropout=0.1, loss=QuantileLoss(), log_interval=-1, reduce_on_plateau_patience=4)
trainer = Trainer(max_epochs=5, gradient_clip_val=0.1, accelerator="auto")
trainer.fit(tft_model, train_dataloader)

def predict_price_tft(route="CGK-KNO", start_days=1):
    today = pd.Timestamp.today().normalize()
    start_date = today + pd.Timedelta(days=start_days)
    df_input = pd.DataFrame({
        "route": [route] * max_prediction_length,
        "route_encoded": le_route.transform([route])[0],
        "time_idx": pd.date_range(start=start_date, periods=max_prediction_length).astype(int) // 10**9,
        "days_until_departure": list(range(start_days, start_days + max_prediction_length)),
        "distance_km": [1415.0] * max_prediction_length,
        "flight_time_hour": [2.3] * max_prediction_length,
    })
    prediction_data = tft_dataset.predict_dataset(df_input)
    raw_predictions, _ = tft_model.predict(prediction_data, mode="raw", return_x=True)
    predicted_scaled = raw_predictions.output[:, :, 0].detach().numpy()
    return scaler_price.inverse_transform(predicted_scaled)

# === GPT + LangChain Setup ===
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 1. Extract structured intent
extract_prompt = PromptTemplate.from_template("""
Extract the following from the user's flight price query:
- origin airport code (e.g. CGK)
- destination airport code (e.g. KNO)
- departure date (ISO format if possible)
- forecast type: "single" or "multi-day"
User Question: {question}
Return as JSON.
""")
extract_chain = LLMChain(prompt=extract_prompt, llm=llm)

# 2. Choose and call model
def handle_model_prediction(parsed_data):
    origin = parsed_data['origin']
    destination = parsed_data['destination']
    date = parsed_data['date']
    forecast_type = parsed_data.get('forecast_type', 'single')
    if forecast_type == "multi-day":
        route = f"{origin}-{destination}"
        prices = predict_price_tft(route=route)
        return {
            "type": "multi-day",
            "route": route,
            "prices": [round(p[0], -3) for p in prices]
        }
    else:
        price = predict_price_rf(origin, destination, date)
        return {
            "type": "single",
            "origin": origin,
            "destination": destination,
            "date": date,
            "price": round(price, -3)
        }

# 3. Generate final response
final_prompt = PromptTemplate.from_template("""
The user asked: {user_question}
You already have this model output: {model_output}
Write a friendly, helpful assistant response for the user.
""")
final_chain = LLMChain(prompt=final_prompt, llm=llm)

# Example full run
user_question = "How much is the ticket from CGK to KNO starting 17 June 2025 for 7 days?"
parsed = extract_chain.run({"question": user_question})
parsed_data = json.loads(parsed)
model_result = handle_model_prediction(parsed_data)
response = final_chain.run({"user_question": user_question, "model_output": json.dumps(model_result)})
print(response)
