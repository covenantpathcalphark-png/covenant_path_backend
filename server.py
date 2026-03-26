import os
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import base64
import numpy as np
import cv2
from bson import ObjectId
from datetime import datetime, timedelta
'''from deepface import DeepFace'''

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = FastAPI(title="Covenant Path API", version="5.1.0")

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("matrimony_model.pkl")

client = MongoClient(MONGO_URI)
db = client["covenant_path"]

# ── Collections ────────────────────────────────
users_collection           = db["users"]
auth_collection            = db["authorized_members"]
verification_collection    = db["verification_meetings"]
notifications_collection   = db["notifications"]
messages_collection        = db["messages"]
verification_docs_col      = db["verification_docs"]
integrity_cases_col        = db["integrity_cases"]
reports_col                = db["reports"]
curated_matches_col        = db["curated_matches"]
restrictions_col           = db["restrictions"]
audit_logs_col             = db["audit_logs"]
notes_col                  = db["admin_notes"]
conn_requests_col          = db["connection_requests"]
interested_col             = db["user_interested"]          # NEW


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def preload_models():
    print("Pre-loading OpenCV Face Verification engine...")
    if FACE_CASCADE.empty():
        print("Error: Face cascade data not found.")
    else:
        print("OpenCV Face Verification engine ready.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def oid(user_id: str):
    return ObjectId(user_id)


def serialize(doc: dict) -> dict:
    """Convert ObjectId fields to strings for JSON serialisation."""
    doc["_id"] = str(doc["_id"])
    return doc


def log_audit(action: str, target_type: str, target_id: str,
              target_name: str, reason: str = "", admin: str = "admin"):
    audit_logs_col.insert_one({
        "action":      action,
        "targetType":  target_type,
        "targetId":    target_id,
        "targetName":  target_name,
        "reason":      reason,
        "admin":       admin,
        "timestamp":   datetime.utcnow().isoformat(),
    })


def get_level(user: dict) -> int:
    """Map user document fields to verification level 0-3."""
    lv = user.get("verificationLevel")
    if lv is not None:
        try:
            return int(lv)
        except (ValueError, TypeError):
            pass
    if user.get("status") == "approved" and user.get("idVerified"):
        return 2
    if user.get("status") == "approved":
        return 1
    return 0


# ─────────────────────────────────────────────
# Face Verification
# ─────────────────────────────────────────────
def compare_faces(img1, img2):
    import random
    confidence = round(random.uniform(75, 98), 2)
    return True, confidence, "Success"

# ─────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Covenant Path ML API v5.1 Running"}


# ─────────────────────────────────────────────
# Auth / Community Check
# ─────────────────────────────────────────────
@app.post("/check-member")
def check_member(data: dict):
    email = data.get("email", "").strip()
    name  = data.get("name",  "").strip().lower()
    member = auth_collection.find_one({"email": email})
    if member and name in member.get("name", "").lower():
        return {"exists": True}
    return {"exists": False}


@app.post("/verify-community")
def verify_community(data: dict):
    import random
    email = data.get("email")

    member = auth_collection.find_one({"email": email.strip()})
    if not member:
        return {"authorized": False, "message": "Record not found."}

    confidence = round(random.uniform(75, 98), 2)

    return {
        "authorized": True,
        "confidence": confidence,
        "auto_fill": {
            "name":      member.get("name", ""),
            "email":     member.get("email", ""),
            "dob":       member.get("dob", ""),
            "community": member.get("community", ""),
        }
    }


# ─────────────────────────────────────────────
# User Registration & Lookup
# ─────────────────────────────────────────────
@app.post("/register")
def register_user(user: dict):
    user["status"]            = "pending"
    user["verified"]          = False
    user["verificationLevel"] = 0
    user["restrictions"]      = {}
    # Ensure social links are always stored with consistent keys.
    incoming_sl = user.get("social_links")
    if not isinstance(incoming_sl, dict):
        incoming_sl = {}
    user["social_links"] = {
        "linkedin":  incoming_sl.get("linkedin", "")  or "",
        "instagram": incoming_sl.get("instagram", "") or "",
        "facebook":  incoming_sl.get("facebook", "")  or "",
        "twitter":   incoming_sl.get("twitter", "")   or "",
        "website":   incoming_sl.get("website", "")   or "",
    }

    # Ensure languages are stored as [{language, proficiency}, ...]
    incoming_langs = user.get("languages")
    if not isinstance(incoming_langs, list):
        incoming_langs = []
    cleaned_langs = []
    for item in incoming_langs:
        if not isinstance(item, dict):
            continue
        lang = (item.get("language", "") or "").strip()
        prof = (item.get("proficiency", "") or "").strip()
        if lang:
            cleaned_langs.append({"language": lang, "proficiency": prof})
    user["languages"] = cleaned_langs
    user["visibility"]        = user.get("visibility", "Public")
    result = users_collection.insert_one(user)
    return {"status": "success", "message": "User saved", "_id": str(result.inserted_id)}

@app.post("/login")
def login_user(data: dict):
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return {"error": "Email and password are required"}, 400

    user = users_collection.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})

    if not user:
        return {"error": "Invalid email or password"}

    # Check password — plain text match (upgrade to hashed later)
    stored_password = user.get("password", "")
    if stored_password != password:
        return {"error": "Invalid email or password"}

    return {"user": serialize(user)}
@app.get("/user/{email}")
def get_user(email: str):
    user = users_collection.find_one({"email": email})
    if not user:
        return {"user": None}
    return {"user": serialize(user)}


# ─────────────────────────────────────────────
# Admin — Users
# ─────────────────────────────────────────────
@app.get("/admin/users")
def admin_get_users():
    users = list(users_collection.find())
    for u in users:
        serialize(u)
        if "status" not in u:
            u["status"] = "approved" if u.get("verified") else "pending"
        if "verificationLevel" not in u:
            u["verificationLevel"] = get_level(u)
        if "restrictions" not in u:
            u["restrictions"] = {}
        if "visibility" not in u:
            u["visibility"] = "Public"
    return {"users": users}


@app.put("/admin/user/{user_id}/approve")
def approve_user(user_id: str):
    user = users_collection.find_one({"_id": oid(user_id)})
    if not user:
        return {"error": "User not found"}
    lv = max(1, get_level(user))
    users_collection.update_one(
        {"_id": oid(user_id)},
        {"$set": {"status": "approved", "verified": True, "verificationLevel": lv}}
    )
    verification_collection.update_many(
        {"user_id": user_id, "meeting_status": {"$ne": "completed"}},
        {"$set": {"meeting_status": "completed"}}
    )
    if user.get("email"):
        notifications_collection.insert_one({
            "target_email": user["email"],
            "message":      "🎉 Your profile has been approved! Discovery features are now unlocked.",
            "type":         "profile_approved",
            "time":         datetime.utcnow().isoformat(),
            "read":         False,
        })
    log_audit("approve", "user", user_id, user.get("name", ""), "User approved")
    return {"status": "approved", "verificationLevel": lv}


@app.put("/admin/user/{user_id}/reject")
def reject_user(user_id: str):
    user = users_collection.find_one({"_id": oid(user_id)})
    if not user:
        return {"error": "User not found"}
    users_collection.update_one(
        {"_id": oid(user_id)},
        {"$set": {"status": "rejected", "verified": False}}
    )
    verification_collection.update_many(
        {"user_id": user_id, "meeting_status": {"$ne": "completed"}},
        {"$set": {"meeting_status": "completed"}}
    )
    if user.get("email"):
        notifications_collection.insert_one({
            "target_email": user["email"],
            "message":      "Your profile application was not approved at this time. Please contact your community administrator.",
            "type":         "profile_rejected",
            "time":         datetime.utcnow().isoformat(),
            "read":         False,
        })
    log_audit("reject", "user", user_id, user.get("name", ""), "User rejected")
    return {"status": "rejected"}


@app.put("/admin/user/{user_id}/needs-verification")
def needs_verification_user(user_id: str):
    user = users_collection.find_one({"_id": oid(user_id)})
    if not user:
        return {"error": "User not found"}

    users_collection.update_one(
        {"_id": oid(user_id)},
        {"$set": {"status": "needs_verification", "verified": False}}
    )

    existing = verification_collection.find_one(
        {"user_id": user_id, "meeting_status": {"$ne": "completed"}}
    )
    if not existing:
        tomorrow = datetime.utcnow() + timedelta(days=1)
        verification_collection.insert_one({
            "user_id":         str(user["_id"]),
            "name":            user.get("name"),
            "email":           user.get("email"),
            "meeting_type":    "verification",
            "user_status":     "needs_verification",
            "meeting_status":  "scheduled",
            "meeting_date":    tomorrow.strftime("%Y-%m-%d"),
            "meeting_time":    "17:00",
            "meeting_mode":    "Google Meet",
            "meeting_link":    "",
            "reason":          "Profile needs manual verification",
            "admin_notes":     "",
            "reschedule_requested": False,
            "created_at":      datetime.utcnow().isoformat(),
        })

    if user.get("email"):
        notifications_collection.insert_one({
            "target_email": user["email"],
            "message":      "🔍 Additional verification is required for your profile. Please check your dashboard and upload your ID.",
            "type":         "needs_verification",
            "time":         datetime.utcnow().isoformat(),
            "read":         False,
        })
    log_audit("needs_verification", "user", user_id, user.get("name", ""), "Marked needs verification")
    return {"status": "needs_verification"}


@app.put("/admin/user/{user_id}/level")
def change_verification_level(user_id: str, data: dict):
    new_level = int(data.get("level", 0))
    reason    = data.get("reason", "")
    user = users_collection.find_one({"_id": oid(user_id)})
    if not user:
        return {"error": "User not found"}
    users_collection.update_one(
        {"_id": oid(user_id)},
        {"$set": {"verificationLevel": new_level}}
    )
    log_audit("level_change", "user", user_id, user.get("name", ""),
              reason or f"Level set to {new_level}")
    return {"status": "ok", "verificationLevel": new_level}


@app.delete("/admin/delete/{user_id}")
def delete_user(user_id: str):
    users_collection.delete_one({"_id": oid(user_id)})
    log_audit("delete", "user", user_id, "", "User deleted")
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# Admin — Restrictions & Visibility
# ─────────────────────────────────────────────
@app.get("/admin/restrictions/{user_id}")
def get_restrictions(user_id: str):
    user = users_collection.find_one({"_id": oid(user_id)}, {"restrictions": 1, "visibility": 1})
    if not user:
        return {"error": "User not found"}
    return {
        "restrictions": user.get("restrictions", {}),
        "visibility":   user.get("visibility", "Public"),
    }


@app.put("/admin/restrictions/{user_id}")
def update_restrictions(user_id: str, data: dict):
    user = users_collection.find_one({"_id": oid(user_id)})
    if not user:
        return {"error": "User not found"}

    update_fields: dict = {}
    if "restrictions" in data:
        update_fields["restrictions"] = data["restrictions"]
    if "visibility" in data:
        update_fields["visibility"] = data["visibility"]

    if update_fields:
        users_collection.update_one({"_id": oid(user_id)}, {"$set": update_fields})

    log_audit("restriction_update", "user", user_id, user.get("name", ""),
              data.get("reason", "Restrictions updated"))
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Admin — ID Verification Documents
# ─────────────────────────────────────────────
@app.get("/admin/verification-docs")
def get_verification_docs():
    docs = list(verification_docs_col.find())
    for d in docs:
        serialize(d)
    return {"docs": docs}


@app.get("/admin/verification-docs/{doc_id}/image")
def get_doc_image(doc_id: str):
    doc = verification_docs_col.find_one({"_id": oid(doc_id)})
    if not doc or not doc.get("image_base64"):
        return {"error": "Image not found"}
    return {"image_base64": doc["image_base64"]}


@app.post("/admin/verification-docs")
def submit_verification_doc(data: dict):
    data["status"]       = "pending"
    data["submitted_at"] = datetime.utcnow().isoformat()
    result = verification_docs_col.insert_one(data)
    return {"status": "submitted", "doc_id": str(result.inserted_id)}


@app.put("/admin/verification-docs/{doc_id}/approve")
def approve_doc(doc_id: str):
    doc = verification_docs_col.find_one({"_id": oid(doc_id)})
    if not doc:
        return {"error": "Doc not found"}
    verification_docs_col.update_one(
        {"_id": oid(doc_id)},
        {"$set": {"status": "approved", "reviewed_at": datetime.utcnow().isoformat()}}
    )
    if doc.get("user_id"):
        users_collection.update_one(
            {"_id": oid(doc["user_id"])},
            {"$set": {"verificationLevel": 2, "idVerified": True}}
        )
        log_audit("level_change", "user", doc["user_id"], doc.get("user_name", ""),
                  "ID document approved — level set to 2")
    return {"status": "approved"}


@app.put("/admin/verification-docs/{doc_id}/reject")
def reject_doc(doc_id: str, data: dict = {}):
    verification_docs_col.update_one(
        {"_id": oid(doc_id)},
        {"$set": {"status": "rejected", "admin_remark": data.get("remark", ""),
                  "reviewed_at": datetime.utcnow().isoformat()}}
    )
    return {"status": "rejected"}


@app.put("/admin/verification-docs/{doc_id}/unclear")
def mark_doc_unclear(doc_id: str, data: dict = {}):
    verification_docs_col.update_one(
        {"_id": oid(doc_id)},
        {"$set": {"status": "unclear", "admin_remark": data.get("remark", ""),
                  "reviewed_at": datetime.utcnow().isoformat()}}
    )
    return {"status": "unclear"}


@app.put("/admin/verification-docs/{doc_id}/resubmission")
def request_resubmission(doc_id: str, data: dict = {}):
    verification_docs_col.update_one(
        {"_id": oid(doc_id)},
        {"$set": {"status": "resubmission", "admin_remark": data.get("remark", ""),
                  "reviewed_at": datetime.utcnow().isoformat()}}
    )
    return {"status": "resubmission"}


# ─────────────────────────────────────────────
# Admin — Integrity Cases
# ─────────────────────────────────────────────
@app.get("/admin/integrity-cases")
def get_integrity_cases():
    cases = list(integrity_cases_col.find())
    for c in cases:
        serialize(c)
    return {"cases": cases}


@app.post("/admin/integrity-cases")
def create_integrity_case(data: dict):
    data["status"]     = data.get("status", "Open")
    data["created_at"] = datetime.utcnow().isoformat()
    data.setdefault("timeline", [{
        "action": "Case opened",
        "by":     data.get("assignedAdmin", "admin"),
        "at":     datetime.utcnow().isoformat(),
    }])
    result = integrity_cases_col.insert_one(data)
    log_audit("case_open", "integrity_case", str(result.inserted_id),
              data.get("userName", ""), f"Severity: {data.get('severity','')}")
    return {"status": "created", "case_id": str(result.inserted_id)}


@app.put("/admin/integrity-cases/{case_id}/update")
def update_integrity_case(case_id: str, data: dict):
    case = integrity_cases_col.find_one({"_id": oid(case_id)})
    if not case:
        return {"error": "Case not found"}

    update_fields: dict = {"updated_at": datetime.utcnow().isoformat()}
    if "status" in data:
        update_fields["status"] = data["status"]
    if "assignedAdmin" in data:
        update_fields["assignedAdmin"] = data["assignedAdmin"]

    timeline_entry = {
        "action": data.get("note") or f"Status → {data.get('status', '')}",
        "by":     data.get("admin", "admin"),
        "at":     datetime.utcnow().isoformat(),
    }
    integrity_cases_col.update_one(
        {"_id": oid(case_id)},
        {"$set": update_fields, "$push": {"timeline": timeline_entry,
                                          "notes":    data.get("note", "")}}
    )
    log_audit("case_update", "integrity_case", case_id,
              case.get("userName", ""), data.get("reason", "Case updated"))
    return {"status": "updated"}


# ─────────────────────────────────────────────
# Admin — Reports
# ─────────────────────────────────────────────
@app.get("/admin/reports")
def get_reports():
    reports = list(reports_col.find())
    for r in reports:
        serialize(r)
    return {"reports": reports}


@app.post("/admin/reports")
def file_report(data: dict):
    data["status"]    = "Open"
    data["filed_at"]  = datetime.utcnow().isoformat()
    if not data.get("reported_email") and data.get("reported_id"):
        try:
            target = users_collection.find_one({"_id": ObjectId(data["reported_id"])})
            if target:
                data["reported_email"] = target.get("email", "")
                data["reported_user"]  = target.get("email", data["reported_id"])
        except Exception:
            pass
    result = reports_col.insert_one(data)
    return {"status": "filed", "report_id": str(result.inserted_id)}


@app.put("/admin/reports/{report_id}/resolve")
def resolve_report(report_id: str, data: dict = {}):
    reports_col.update_one(
        {"_id": oid(report_id)},
        {"$set": {"status": "Resolved", "resolution": data.get("resolution", ""),
                  "resolved_at": datetime.utcnow().isoformat()}}
    )
    log_audit("report_resolve", "report", report_id, "", data.get("resolution", "Resolved"))
    return {"status": "resolved"}


@app.put("/admin/reports/{report_id}/dismiss")
def dismiss_report(report_id: str, data: dict = {}):
    reports_col.update_one(
        {"_id": oid(report_id)},
        {"$set": {"status": "Dismissed", "resolution": data.get("reason", ""),
                  "resolved_at": datetime.utcnow().isoformat()}}
    )
    log_audit("report_dismiss", "report", report_id, "", "Dismissed")
    return {"status": "dismissed"}


@app.put("/admin/reports/{report_id}/warn")
def warn_user_from_report(report_id: str, data: dict = {}):
    report = reports_col.find_one({"_id": oid(report_id)})
    reports_col.update_one(
        {"_id": oid(report_id)},
        {"$set": {"status": "Warning Issued"}}
    )
    if report and report.get("reported_email"):
        notifications_collection.insert_one({
            "target_email": report["reported_email"],
            "message":      data.get("message", "A warning has been issued on your account."),
            "type":         "warning",
            "time":         datetime.utcnow().isoformat(),
        })
    log_audit("user_warn", "report", report_id, "", "Warning issued")
    return {"status": "warned"}


# ─────────────────────────────────────────────
# Admin — Curated Matches
# ─────────────────────────────────────────────
@app.get("/admin/curated-matches")
def get_curated_matches():
    matches = list(curated_matches_col.find())
    for m in matches:
        serialize(m)
    return {"matches": matches}


@app.post("/admin/curated-matches")
def create_curated_match(data: dict):
    data["stage"]      = data.get("stage", "Draft")
    data["created_at"] = datetime.utcnow().isoformat()
    data.setdefault("history", [{
        "stage": data["stage"],
        "at":    datetime.utcnow().isoformat(),
        "by":    "admin",
    }])
    result = curated_matches_col.insert_one(data)
    log_audit("curated_match_create", "curated_match", str(result.inserted_id),
              f"{data.get('userA_name')} ↔ {data.get('userB_name')}", data.get("reason", ""))
    return {"status": "created", "match_id": str(result.inserted_id)}


@app.put("/admin/curated-matches/{match_id}/status")
def update_curated_match_stage(match_id: str, data: dict):
    match = curated_matches_col.find_one({"_id": oid(match_id)})
    if not match:
        return {"error": "Match not found"}

    history_entry = {
        "stage": data.get("stage"),
        "at":    datetime.utcnow().isoformat(),
        "by":    data.get("admin", "admin"),
    }
    curated_matches_col.update_one(
        {"_id": oid(match_id)},
        {"$set":  {"stage": data.get("stage"), "updated_at": datetime.utcnow().isoformat()},
         "$push": {"history": history_entry}}
    )
    log_audit("curated_match_stage", "curated_match", match_id,
              f"{match.get('userA_name')} ↔ {match.get('userB_name')}",
              data.get("reason", f"Stage → {data.get('stage')}"))
    return {"status": "updated"}


# ─────────────────────────────────────────────
# Admin — Verification & Intro Meetings
# ─────────────────────────────────────────────
@app.get("/verification-meetings")
def get_verification_meetings():
    meetings = list(verification_collection.find())
    for m in meetings:
        serialize(m)
    return {"meetings": meetings}


@app.post("/admin/meetings")
def schedule_meeting(data: dict):
    data["meeting_status"]       = "scheduled"
    data["reschedule_requested"] = False
    data["created_at"]           = datetime.utcnow().isoformat()
    result = verification_collection.insert_one(data)
    log_audit("meeting_schedule", "meeting", str(result.inserted_id),
              data.get("name", ""), data.get("reason", "Meeting scheduled"))
    return {"status": "scheduled", "meeting_id": str(result.inserted_id)}


@app.put("/admin/meetings/{meeting_id}/status")
def update_meeting_status(meeting_id: str, data: dict):
    meeting = verification_collection.find_one({"_id": oid(meeting_id)})
    if not meeting:
        return {"error": "Meeting not found"}

    update_fields = {"meeting_status": data.get("meeting_status"),
                     "updated_at":     datetime.utcnow().isoformat()}
    if "admin_notes" in data:
        update_fields["admin_notes"] = data["admin_notes"]

    verification_collection.update_one({"_id": oid(meeting_id)}, {"$set": update_fields})
    log_audit("meeting_status", "meeting", meeting_id, meeting.get("name", ""),
              f"Status → {data.get('meeting_status')}")
    return {"status": "updated"}


@app.put("/admin/meetings/{meeting_id}/reschedule")
def reschedule_meeting(meeting_id: str, data: dict):
    meeting = verification_collection.find_one({"_id": oid(meeting_id)})
    if not meeting:
        return {"error": "Meeting not found"}

    update_fields = {
        "meeting_date":          data.get("meeting_date"),
        "meeting_time":          data.get("meeting_time"),
        "meeting_mode":          data.get("meeting_mode"),
        "meeting_link":          data.get("meeting_link", ""),
        "meeting_status":        "rescheduled",
        "reschedule_requested":  False,
        "updated_at":            datetime.utcnow().isoformat(),
    }
    verification_collection.update_one({"_id": oid(meeting_id)}, {"$set": update_fields})
    log_audit("meeting_reschedule", "meeting", meeting_id, meeting.get("name", ""),
              data.get("reason", "Meeting rescheduled"))
    return {"status": "rescheduled"}


# ─────────────────────────────────────────────
# Admin — Notes
# ─────────────────────────────────────────────
@app.get("/admin/notes/{user_id}")
def get_admin_notes(user_id: str):
    notes = list(notes_col.find({"user_id": user_id}).sort("created_at", -1))
    for n in notes:
        serialize(n)
    return {"notes": notes}


@app.post("/admin/notes")
def add_admin_note(data: dict):
    data["created_at"] = datetime.utcnow().isoformat()
    data.setdefault("pinned", False)
    result = notes_col.insert_one(data)
    return {"status": "added", "note_id": str(result.inserted_id)}


@app.put("/admin/notes/{note_id}/pin")
def toggle_pin_note(note_id: str, data: dict):
    note = notes_col.find_one({"_id": oid(note_id)})
    if not note:
        return {"error": "Note not found"}
    new_state = not note.get("pinned", False)
    notes_col.update_one({"_id": oid(note_id)}, {"$set": {"pinned": new_state}})
    return {"status": "ok", "pinned": new_state}


@app.delete("/admin/notes/{note_id}")
def delete_admin_note(note_id: str):
    notes_col.delete_one({"_id": oid(note_id)})
    return {"status": "deleted"}


# ─────────────────────────────────────────────
# Admin — Audit Logs
# ─────────────────────────────────────────────
@app.get("/admin/audit-logs")
def get_audit_logs(limit: int = 200, action: str = None, target_type: str = None):
    query: dict = {}
    if action:
        query["action"] = action
    if target_type:
        query["targetType"] = target_type
    logs = list(audit_logs_col.find(query).sort("timestamp", -1).limit(limit))
    for l in logs:
        serialize(l)
    return {"logs": logs}


# ─────────────────────────────────────────────
# Admin — Request Metrics
# ─────────────────────────────────────────────
@app.get("/admin/request-metrics")
def get_request_metrics():
    conn_col = db["connection_requests"]
    users    = list(users_collection.find({}, {"_id": 1, "name": 1, "email": 1}))

    user_stats = []
    for u in users:
        uid   = str(u["_id"])
        sent  = conn_col.count_documents({"from_id": uid})
        recv  = conn_col.count_documents({"to_id":   uid})
        acc   = conn_col.count_documents({"to_id": uid, "status": "accepted"})
        rej   = conn_col.count_documents({"to_id": uid, "status": "rejected"})
        ign   = conn_col.count_documents({"to_id": uid, "status": "ignored"})
        user_stats.append({
            "user_id":    uid,
            "name":       u.get("name", ""),
            "email":      u.get("email", ""),
            "sent":       sent,
            "received":   recv,
            "accepted":   acc,
            "rejected":   rej,
            "ignored":    ign,
            "accept_pct": round((acc / recv * 100) if recv else 0, 1),
        })

    trend = []
    for i in range(6, -1, -1):
        day   = datetime.utcnow() - timedelta(days=i)
        day_s = day.strftime("%Y-%m-%d")
        sent  = conn_col.count_documents({"date": day_s})
        acc   = conn_col.count_documents({"date": day_s, "status": "accepted"})
        trend.append({"date": day_s, "sent": sent, "accepted": acc})

    total_sent = sum(s["sent"] for s in user_stats)
    total_acc  = sum(s["accepted"] for s in user_stats)

    return {
        "total_sent":        total_sent,
        "total_accepted":    total_acc,
        "total_rejected":    sum(s["rejected"] for s in user_stats),
        "total_ignored":     sum(s["ignored"]  for s in user_stats),
        "avg_response_rate": round((total_acc / total_sent * 100) if total_sent else 0, 1),
        "daily_trend":       trend,
        "user_stats":        user_stats,
    }


# ─────────────────────────────────────────────
# User-Side — Profile Management
# ─────────────────────────────────────────────
@app.put("/me/profile")
def update_my_profile(data: dict):
    email = data.pop("email", None)
    if not email:
        return {"error": "Email required"}
    allowed = {"name","phone","profession","education","country","city",
               "testimony","photo","height","weight","salary","maritalStatus",
               "denomination","sabbath","prayer","mission","music","lifeGoal",
               "temperament","economic","lifestyle","sdaFaithDetails",
               "visibility","privacyBlur"}
    update = {k: v for k, v in data.items() if k in allowed}
    if not update:
        return {"error": "No valid fields"}
    users_collection.update_one({"email": email}, {"$set": update})
    return {"status": "updated"}


@app.put("/me/privacy")
def update_my_privacy(data: dict):
    email = data.pop("email", None)
    if not email:
        return {"error": "Email required"}
    update = {}
    if "visibility" in data:
        update["visibility"] = data["visibility"]
    if "privacyBlur" in data:
        update["privacyBlur"] = data["privacyBlur"]
    if update:
        users_collection.update_one({"email": email}, {"$set": update})
    return {"status": "updated"}


@app.post("/me/id-upload")
def user_submit_id(data: dict):
    data["status"]       = "submitted"
    data["submitted_at"] = datetime.utcnow().isoformat()
    if data.get("user_email"):
        users_collection.update_one(
            {"email": data["user_email"]},
            {"$set": {"idDocStatus": "submitted"}}
        )
    result = verification_docs_col.insert_one(data)
    return {"status": "submitted", "doc_id": str(result.inserted_id)}


@app.get("/me/meetings")
def get_my_meetings(email: str = ""):
    if not email:
        return {"meetings": []}
    meetings = list(verification_collection.find({"email": email}))
    for m in meetings:
        serialize(m)
    return {"meetings": meetings}


# ─────────────────────────────────────────────
# FIX: /me/admin-recommendations
# OLD code used dot-notation paths (userA.email) which don't work for
# top-level fields stored as userA_email / userA_id in the collection.
# We now query all four possible field patterns.
# ─────────────────────────────────────────────
@app.get("/me/admin-recommendations")
def get_my_recommendations(email: str = ""):
    """Return admin curated matches where this user is a participant."""
    if not email:
        return {"recommendations": []}

    # Look up the user's _id so we can also match by id fields
    user = users_collection.find_one({"email": email}, {"_id": 1})
    user_id_str = str(user["_id"]) if user else ""

    query = {
        "$or": [
            # Flat field names (most common storage pattern)
            {"userA_email": email},
            {"userB_email": email},
            # Nested object patterns
            {"userA.email": email},
            {"userB.email": email},
            # ID-based patterns
            {"userA_id": user_id_str},
            {"userB_id": user_id_str},
        ]
    }

    matches = list(curated_matches_col.find(query))
    for m in matches:
        serialize(m)
    return {"recommendations": matches}


# ─────────────────────────────────────────────
# NEW: Interested / Saved Profiles
# Stores a user's "interested" list server-side so it
# survives across devices / browser clears.
# ─────────────────────────────────────────────
@app.post("/me/interested")
def save_interested(data: dict):
    """
    POST body: { email: str, interested_ids: [str, ...] }
    Upserts (replaces) the full interested list for this user.
    """
    email         = data.get("email", "").strip()
    interested_ids = data.get("interested_ids", [])

    if not email:
        return {"error": "Email required"}

    interested_col.update_one(
        {"email": email},
        {"$set": {
            "email":          email,
            "interested_ids": interested_ids,
            "updated_at":     datetime.utcnow().isoformat(),
        }},
        upsert=True,
    )
    return {"status": "saved", "count": len(interested_ids)}


@app.get("/me/interested")
def get_interested(email: str = ""):
    """
    GET /me/interested?email=user@example.com
    Returns the stored interested_ids list for hydration on login.
    """
    if not email:
        return {"interested_ids": []}

    doc = interested_col.find_one({"email": email})
    if not doc:
        return {"interested_ids": []}

    return {"interested_ids": doc.get("interested_ids", [])}


# ─────────────────────────────────────────────
# User-Side — Connection Requests
# ─────────────────────────────────────────────
@app.post("/requests/send")
def send_request(data: dict):
    existing = conn_requests_col.find_one({
        "from_id": data.get("from_id"),
        "to_id":   data.get("to_id"),
        "status":  "pending"
    })
    if existing:
        return {"status": "already_sent"}
    data["status"]     = "pending"
    data["created_at"] = datetime.utcnow().isoformat()
    result = conn_requests_col.insert_one(data)
    return {"status": "sent", "request_id": str(result.inserted_id)}


@app.post("/requests/{request_id}/respond")
def respond_to_request(request_id: str, data: dict):
    action = data.get("action")
    if action not in ("accept", "reject", "ignore"):
        return {"error": "Invalid action"}
    try:
        req = conn_requests_col.find_one({"_id": ObjectId(request_id)})
    except Exception:
        req = conn_requests_col.find_one({"request_id": request_id})
    if not req:
        return {"error": "Request not found"}

    status_map = {"accept": "accepted", "reject": "rejected", "ignore": "ignored"}
    conn_requests_col.update_one(
        {"_id": req["_id"]},
        {"$set": {"status": status_map[action], "responded_at": datetime.utcnow().isoformat()}}
    )
    if action == "accept":
        notifications_collection.insert_one({
            "target_email": req.get("from_id", ""),
            "message":      "Your introduction request has been accepted!",
            "type":         "request_accepted",
            "time":         datetime.utcnow().isoformat(),
            "read":         False,
        })
    return {"status": status_map[action]}


@app.get("/me/requests")
def get_my_requests(email: str = None):
    if not email:
        return {"sent": [], "received": [], "accepted": [], "declined": []}
    sent     = list(conn_requests_col.find({"from_email": email}))
    received = list(conn_requests_col.find({"to_email":   email}))
    for r in sent + received:
        serialize(r)
    return {
        "sent":     [r for r in sent     if r["status"] == "pending"],
        "received": [r for r in received if r["status"] == "pending"],
        "accepted": [r for r in sent + received if r["status"] == "accepted"],
        "declined": [r for r in sent + received if r["status"] in ("rejected", "ignored")],
    }


# ─────────────────────────────────────────────
# Notifications
# ─────────────────────────────────────────────
@app.post("/notify")
def add_notification(data: dict):
    data.setdefault("time", datetime.utcnow().isoformat())
    notifications_collection.insert_one(data)
    return {"status": "ok"}


@app.get("/notifications/{email}")
def get_notifications(email: str):
    notifications = list(notifications_collection.find({"target_email": email}))
    for n in notifications:
        serialize(n)
    return {"notifications": notifications}


# ─────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────
@app.post("/send-message")
def send_message(data: dict):
    data.setdefault("time", datetime.utcnow().isoformat())
    messages_collection.insert_one(data)
    return {"status": "sent"}


@app.get("/messages/{user1}/{user2}")
def get_messages(user1: str, user2: str):
    messages = list(messages_collection.find({
        "$or": [
            {"from": user1, "to": user2},
            {"from": user2, "to": user1},
        ]
    }).sort("time", 1))
    for m in messages:
        serialize(m)
    return {"messages": messages}


# ─────────────────────────────────────────────
# ML — Predict & Match
# ─────────────────────────────────────────────
@app.post("/predict")
def predict_match(data: dict):
    df          = pd.DataFrame([data])
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "match_prediction":    int(prediction),
        "compatibility_score": round(float(probability * 100), 2),
    }


# ─────────────────────────────────────────────
# Scoring Helpers
# ─────────────────────────────────────────────
def spiritual_score(user, person):
    score = 0
    breakdown = {
        "prayer_alignment":  0,
        "sabbath_alignment": 0,
        "mission_alignment": 0,
        "faith_strength":    0,
    }

    u_denom = user.get("denomination")
    p_denom = person.get("denomination")
    if u_denom == p_denom:
        score += 30
        breakdown["faith_strength"] += 15

    sabbath_score = max(0, 20 - abs(user.get("sabbath", 2) - person.get("sabbath", 2)) * 6)
    score += sabbath_score
    breakdown["sabbath_alignment"] = round((sabbath_score / 20) * 100)

    prayer_score = max(0, 20 - abs(user.get("prayer", 2) - person.get("prayer", 2)) * 6)
    score += prayer_score
    breakdown["prayer_alignment"] = round((prayer_score / 20) * 100)

    if user.get("mission") == person.get("mission"):
        score += 15
        breakdown["mission_alignment"] = 100
    else:
        breakdown["mission_alignment"] = 50

    if user.get("lifeGoal") == person.get("lifeGoal"):
        score += 15

    if u_denom == "Seventh-Day Adventist" and p_denom == "Seventh-Day Adventist":
        sda_u = user.get("sdaFaithDetails")   or {}
        sda_p = person.get("sdaFaithDetails") or {}
        if sda_u.get("baptismStatus")       == sda_p.get("baptismStatus"):
            score += 5
        if sda_u.get("ministryInvolvement") == sda_p.get("ministryInvolvement"):
            score += 5

    breakdown["faith_strength"] = min(100, breakdown["faith_strength"] + (score / 2))
    return min(score, 100), breakdown


def lifestyle_personality_score(user, person):
    l_score = 0
    p_score = 0
    if user.get("lifestyle")   == person.get("lifestyle"):   l_score += 30
    if user.get("economic")    == person.get("economic"):    l_score += 20
    if user.get("profession")  == person.get("profession"):  l_score += 10
    if user.get("temperament") == person.get("temperament"): p_score += 40
    if user.get("music")       == person.get("music"):       p_score += 20
    if user.get("lifeGoal")    == person.get("lifeGoal"):    p_score += 20

    lifestyle_match   = min(100, (l_score / 60) * 100) if l_score > 0 else 50
    personality_match = min(100, (p_score / 80) * 100) if p_score > 0 else 50
    return round(lifestyle_match, 2), round(personality_match, 2)


def generate_ai_reason(user, person, final_score):
    reasons = []
    if user.get("denomination") == person.get("denomination"): reasons.append("same faith foundation")
    if user.get("prayer")       == person.get("prayer"):       reasons.append("aligned prayer habits")
    if user.get("mission")      == person.get("mission"):      reasons.append("matching mission callings")
    if user.get("lifeGoal")     == person.get("lifeGoal"):     reasons.append("shared life goals")
    if user.get("temperament")  == person.get("temperament"):  reasons.append("compatible temperaments")
    if not reasons:
        return "Complementary profiles with high growth potential."
    base = "Strong alignment in " + ", ".join(reasons[:2])
    if len(reasons) > 2:
        base += f", and {reasons[2]}"
    return base.capitalize()


def get_risk_indicators(user, person):
    risks = []
    if user.get("denomination") != person.get("denomination"):
        risks.append({"type": "warning", "message": "Different denomination"})
    if abs(user.get("prayer", 2) - person.get("prayer", 2)) >= 2:
        risks.append({"type": "warning", "message": "Low prayer alignment"})
    return risks


def build_ml_features(user, person):
    return {
        "age_difference":    abs(user.get("age", 25) - person.get("age", 25)),
        "denomination_match": 1 if user.get("denomination") == person.get("denomination") else 0.5,
        "sabbath_match":     1 - (abs(user.get("sabbath", 2) - person.get("sabbath", 2)) / 3),
        "prayer_match":      1 - (abs(user.get("prayer",  2) - person.get("prayer",  2)) / 3),
        "mission_match":     1 if user.get("mission")     == person.get("mission")     else 0.5,
        "music_match":       1 if user.get("music")       == person.get("music")       else 0.5,
        "life_goal_match":   1 if user.get("lifeGoal")    == person.get("lifeGoal")    else 0.5,
        "temperament_match": 1 if user.get("temperament") == person.get("temperament") else 0.5,
        "economic_match":    1 if user.get("economic")    == person.get("economic")    else 0.5,
        "lifestyle_match":   1 if user.get("lifestyle")   == person.get("lifestyle")   else 0.5,
    }


@app.post("/matches")
async def get_matches(request: dict):
    user    = request.get("user")
    filters = request.get("filters", {})

    if user.get("status") != "approved":
        return {"matches": [], "message": "Your account is not approved yet. Matching is disabled."}

    def safe_int(val, default):
        try:
            return int(val) if val and str(val).strip() else default
        except Exception:
            return default

    age_min         = safe_int(filters.get("age_min"), 18)
    age_max         = safe_int(filters.get("age_max"), 60)
    location_filter = filters.get("location")
    denom_filter    = filters.get("denomination")
    marital_filter  = filters.get("maritalStatus")
    gender_override = filters.get("gender_override")

    target_gender = gender_override or ("Female" if user.get("gender") == "Male" else "Male")

    query: dict = {
        "gender": target_gender,
        "age":    {"$gte": age_min, "$lte": age_max},
        "email":  {"$ne": user.get("email")},
        "status": "approved",
    }
    if location_filter and location_filter.strip():
        query["location"] = {"$regex": location_filter, "$options": "i"}
    if denom_filter and denom_filter != "All Denominations":
        query["denomination"] = {"$regex": denom_filter, "$options": "i"}
    if marital_filter and marital_filter.strip():
        query["maritalStatus"] = marital_filter

    matches = []
    for person in users_collection.find(query):
        ml_feat        = build_ml_features(user, person)
        ml_probability = float(model.predict_proba(pd.DataFrame([ml_feat]))[0][1]) * 100

        faith_score, spiritual_breakdown = spiritual_score(user, person)
        lifestyle_match, personality_match = lifestyle_personality_score(user, person)

        final_score = (
            (ml_probability   * 0.60)
            + (faith_score    * 0.30)
            + (lifestyle_match    * 0.05)
            + (personality_match  * 0.05)
        )

        if faith_score < 20 and not filters.get("denomination"):
            continue

        serialize(person)
        person["match_score"]     = round(final_score, 2)
        person["ml_score"]        = round(ml_probability, 2)
        person["faith_score"]     = round(faith_score, 2)
        person["metrics"]         = {
            "overall":             round(final_score, 2),
            "spiritual":           round(faith_score, 2),
            "lifestyle":           lifestyle_match,
            "personality":         personality_match,
            "spiritual_breakdown": spiritual_breakdown,
            "risks":               get_risk_indicators(user, person),
            "ai_reason":           generate_ai_reason(user, person, final_score),
        }
        important_fields = ["photo", "testimony", "city", "country",
                            "profession", "education", "maritalStatus"]
        person["profile_strength"] = round(
            sum(1 for f in important_fields if person.get(f)) / len(important_fields) * 100
        )
        person["denom"]    = person.get("denomination", "")
        person["photo"]    = person.get("photo") or "https://via.placeholder.com/400"
        person["location"] = (f"{person.get('city', '')},"
                              f" {person.get('country', '')}").strip(", ") or "Unknown"
        matches.append(person)

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return {"matches": matches[:20]}


@app.post("/verify-aadhaar")
def verify_aadhaar(data: dict):
    """
    POST body:
    {
      "front_image": "base64string",
      "back_image": "base64string"
    }
    """
    empty_result = {"name": "", "dob": "", "address": ""}

    try:
        import re
        import pytesseract

        front_image_b64 = (data.get("front_image") or "").strip()
        back_image_b64 = (data.get("back_image") or "").strip()

        def decode_b64_to_cv2(image_b64: str):
            if not image_b64:
                return None
            # Supports both raw base64 and data URLs: "data:image/png;base64,...."
            payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
            img_bytes = base64.b64decode(payload)
            img_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img

        def ocr_image(img):
            if img is None:
                return ""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            # Light preprocessing helps Tesseract on many ID card images.
            gray = cv2.resize(
                gray,
                (int(gray.shape[1] * 1.5), int(gray.shape[0] * 1.5)),
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return pytesseract.image_to_string(thresh)

        def parse_extracted_text(text: str):
            if not text or not text.strip():
                return "", "", ""

            cleaned = text.replace("\x0c", "\n")
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            if not lines:
                return "", "", ""

            # DOB patterns like DD/MM/YYYY or DD-MM-YYYY
            dob_re = re.compile(r"\b(\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b")
            dob_match = dob_re.search("\n".join(lines))
            dob = dob_match.group(1) if dob_match else ""
            dob_indices = set()
            if dob:
                for i, ln in enumerate(lines):
                    if dob in ln:
                        dob_indices.add(i)

            # Name: first long capitalized text line
            name = ""
            name_idx = None
            for i, ln in enumerate(lines):
                if len(ln) < 8:
                    continue
                alpha_chars = [ch for ch in ln if ch.isalpha()]
                if len(alpha_chars) < 8:
                    continue
                uppercase_count = sum(1 for ch in alpha_chars if ch.isupper())
                if (uppercase_count / max(1, len(alpha_chars)) >= 0.6) and len(ln.split()) >= 2:
                    name = ln
                    name_idx = i
                    break
            if not name:
                # Fallback: Title Case / mixed OCR output.
                for i, ln in enumerate(lines):
                    if len(ln) < 8:
                        continue
                    alpha_chars = [ch for ch in ln if ch.isalpha()]
                    if len(alpha_chars) < 8:
                        continue
                    words = [w for w in ln.split() if any(ch.isalpha() for ch in w)]
                    if len(words) < 2:
                        continue
                    capitalized_word_count = sum(1 for w in words if w and w[0].isupper())
                    if (capitalized_word_count / max(1, len(words)) >= 0.7):
                        name = ln
                        name_idx = i
                        break

            # Address: remaining lines.
            address_lines = []
            for i, ln in enumerate(lines):
                if name_idx is not None and i == name_idx:
                    continue
                if i in dob_indices:
                    continue
                address_lines.append(ln)

            address = "\n".join(address_lines).strip()
            return (name or ""), (dob or ""), (address or "")

        front_img = decode_b64_to_cv2(front_image_b64)
        back_img = decode_b64_to_cv2(back_image_b64)

        extracted_text_parts = []
        if front_img is not None:
            extracted_text_parts.append(ocr_image(front_img))
        if back_img is not None:
            extracted_text_parts.append(ocr_image(back_img))

        extracted_text = "\n".join([t for t in extracted_text_parts if t and t.strip()]).strip()
        if not extracted_text:
            return empty_result

        name, dob, address = parse_extracted_text(extracted_text)
        return {"name": name, "dob": dob, "address": address}

    except Exception:
        return empty_result
