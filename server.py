import os
from dotenv import load_dotenv
from pymongo import MongoClient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5500",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("matrimony_model.pkl")

client = MongoClient(MONGO_URI)
db = client["covenant_path"]
users_collection = db["users"]

@app.get("/")
def home():
    return {"message": "Matrimony ML API Running"}

@app.post("/predict")
def predict_match(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "match_prediction": int(prediction),
        "compatibility_score": round(float(probability * 100), 2)
    }

@app.post("/register")
def register_user(user: dict):
    users_collection.insert_one(user)
    return {"status": "success", "message": "User saved"}

# --- Notifications Collection ---
notifications_collection = db["notifications"]

@app.post("/notify")
def add_notification(data: dict):
    notifications_collection.insert_one(data)
    return {"status": "ok"}

@app.get("/notifications/{email}")
def get_notifications(email: str):

    notifications = list(
        notifications_collection.find(
            {"target_email": email}
        )
    )

    for n in notifications:
        n["_id"] = str(n["_id"])

    return {"notifications": notifications}

# --- Chat Messages Collection ---
messages_collection = db["messages"]

@app.post("/send-message")
def send_message(data: dict):
    messages_collection.insert_one(data)
    return {"status": "sent"}

@app.get("/messages/{user1}/{user2}")
def get_messages(user1: str, user2: str):

    messages = list(messages_collection.find({
        "$or": [
            {"from": user1, "to": user2},
            {"from": user2, "to": user1}
        ]
    }).sort("time", 1))

    for m in messages:
        m["_id"] = str(m["_id"])

    return {"messages": messages}

@app.get("/admin/users")
def admin_get_users():
    users = list(users_collection.find())

    for u in users:
        u["_id"] = str(u["_id"])
    return {"users": users}
    
from bson import ObjectId

@app.post("/admin/approve/{user_id}")
def approve_user(user_id: str):

    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"verified": True}}
    )

    return {"status": "approved"}

@app.delete("/admin/delete/{user_id}")
def delete_user(user_id: str):

    users_collection.delete_one(
        {"_id": ObjectId(user_id)}
    )

    return {"status": "deleted"}

@app.get("/user/{email}")
def get_user(email: str):

    user = users_collection.find_one({"email": email})

    if not user:
        return {"user": None}

    user["_id"] = str(user["_id"])

    return {"user": user}

def spiritual_score(user, person):

    score = 0

    # Same denomination = strong base
    if user.get("denomination") == person.get("denomination"):
        score += 30

    # Sabbath alignment
    sabbath_diff = abs(user.get("sabbath",2) - person.get("sabbath",2))
    score += max(0, 20 - sabbath_diff * 6)

    # Prayer alignment
    prayer_diff = abs(user.get("prayer",2) - person.get("prayer",2))
    score += max(0, 20 - prayer_diff * 6)

    # Mission + life purpose
    if user.get("mission") == person.get("mission"):
        score += 15

    if user.get("lifeGoal") == person.get("lifeGoal"):
        score += 15

    return min(score, 100)

def build_ml_features(user, person):

    return {
        "age_difference": abs(user.get("age",25) - person.get("age",25)),
        "denomination_match": 1 if user.get("denomination")==person.get("denomination") else 0.5,
        "sabbath_match": 1 - (abs(user.get("sabbath",2)-person.get("sabbath",2))/3),
        "prayer_match": 1 - (abs(user.get("prayer",2)-person.get("prayer",2))/3),
        "mission_match": 1 if user.get("mission")==person.get("mission") else 0.5,
        "music_match": 1 if user.get("music")==person.get("music") else 0.5,
        "life_goal_match": 1 if user.get("lifeGoal")==person.get("lifeGoal") else 0.5,
        "temperament_match": 1 if user.get("temperament")==person.get("temperament") else 0.5,
        "economic_match": 1 if user.get("economic")==person.get("economic") else 0.5,
        "lifestyle_match": 1 if user.get("lifestyle")==person.get("lifestyle") else 0.5,
    }   

@app.post("/matches")
def get_matches(user: dict):

    candidates = list(users_collection.find({
        "gender": {"$ne": user["gender"]}
    }))

    matches = []

    for person in candidates:

        data = build_ml_features(user, person)

        df = pd.DataFrame([data])

        ml_probability = float(
            model.predict_proba(df)[0][1]
        ) * 100

        faith_score = spiritual_score(user, person)

        # HYBRID SCORE (REAL MATCHING)
        final_score = (ml_probability * 0.7) + (faith_score * 0.3)

        person["_id"] = str(person["_id"])
        person["match_score"] = round(final_score, 2)
        person["ml_score"] = round(ml_probability, 2)
        person["faith_score"] = round(faith_score, 2)

        # Skip if extremely spiritually incompatible
        if faith_score < 30:
            continue

        # frontend compatibility fields
        person["denom"] = person.get("denomination", "")
        person["photo"] = person.get("photo") or "https://via.placeholder.com/400"
        person["location"] = person.get(
            "location",
            "Unknown"
        )

        matches.append(person)

    matches.sort(key=lambda x: x["match_score"], reverse=True)

    return {"matches": matches[:10]}
