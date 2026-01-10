# streamlit_app.py
# Streamlit app for an Adaptive Explanation Study prototype
# - SQLite + SQLAlchemy storage for users & sessions
# - bcrypt password hashing
# - pseudonymization (HMAC with SECRET_KEY saved to secret.key)
# - consent UI (checkbox that proceeds reliably)
# - randomized/counterbalanced policy assignment
# - attention check
# - admin researcher view (download anonymized CSV, delete data)
# - environment/package versions logged per session
# - mirrored CSV export: collected_sessions.csv
#
# Save as streamlit_app.py and run with:
#   streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import uuid
import bcrypt
import hashlib
import hmac
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    insert,
    select,
    desc,
)
import importlib.metadata as importlib_metadata
import json

# -----------------------------
# Helper: safe rerun (handles Streamlit versions that lack experimental_rerun)
# -----------------------------
def safe_rerun():
    """
    Try to call streamlit.experimental_rerun(). If not available,
    set a small session-state toggle and stop execution. The page
    will reload on next interaction or manual refresh.
    """
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state["_needs_rerun"] = not st.session_state.get("_needs_rerun", False)
        st.stop()

# -----------------------------
# Config & paths
# -----------------------------
APP_VERSION = "1.2.1"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
CSV_PATH = BASE_DIR / "collected_sessions.csv"
SECRET_KEY_PATH = BASE_DIR / "secret.key"
ADMIN_HASH_PATH = BASE_DIR / "admin_pass.hash"
PREREG_LINK = "https://osf.io/your-preregistration-placeholder"  # put your OSF prereg link here

st.set_page_config(page_title="Interactive AI Study", layout="centered")

# -----------------------------
# Ensure secret key exists (HMAC key for pseudonymization)
# -----------------------------
def get_or_create_secret_key(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    key = os.urandom(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return key

SECRET_KEY = get_or_create_secret_key(SECRET_KEY_PATH)

# -----------------------------
# Database initialization (SQLite + SQLAlchemy)
# -----------------------------
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("username", String, primary_key=True),
    Column("password_hash", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("pseudonym", String, nullable=False, index=True),
    Column("username_hmac", String, nullable=False),
    Column("participant_id", String, nullable=False),
    Column("trial", Integer),
    Column("policy", String),
    Column("difficulty", String),
    Column("agreement", String),
    Column("confidence", Float),
    Column("accuracy", Integer),
    Column("reaction_time_ms", Integer),
    Column("attention_check_failed", Boolean, default=False),
    Column("timestamp_utc", String),
    Column("app_version", String),
    Column("env_info", String),  # JSON string for reproducibility
)

metadata.create_all(engine)

# -----------------------------
# Admin password initialization (default: 'changeme' - change it!)
# -----------------------------
def get_or_create_admin_hash(path: Path) -> str:
    if path.exists():
        return path.read_text()
    default = "changeme"
    h = bcrypt.hashpw(default.encode(), bcrypt.gensalt()).decode()
    path.write_text(h)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return h

ADMIN_PASSWORD_HASH = get_or_create_admin_hash(ADMIN_HASH_PATH)

# -----------------------------
# CSV mirror (flat-file export)
# -----------------------------
def append_to_csv(row: dict, path: Path):
    """
    Append a single trial row to collected_sessions.csv.
    Creates the file with header if it does not exist.
    """
    df = pd.DataFrame([row])
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)

# -----------------------------
# Utility functions
# -----------------------------
def hash_password_bcrypt(raw_password: str) -> str:
    return bcrypt.hashpw(raw_password.encode(), bcrypt.gensalt()).decode()

def check_password_bcrypt(raw_password: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(raw_password.encode(), pw_hash.encode())
    except Exception:
        return False

def username_hmac(username: str) -> str:
    # deterministic HMAC for server-side mapping (keeps username non-reversible without key)
    return hmac.new(SECRET_KEY, username.encode(), hashlib.sha256).hexdigest()

def pseudonym_from_username(username: str) -> str:
    # short pseudonym derived from HMAC for display/download
    full = username_hmac(username)
    return full[:12]

def get_env_info():
    pkgs = ["python", "streamlit", "pandas", "numpy", "sqlalchemy", "bcrypt"]
    info = {}
    info["python_version"] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    for p in pkgs[1:]:
        try:
            info[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            info[p] = None
    return info

# -----------------------------
# DB helpers
# -----------------------------
def create_user(username: str, raw_password: str) -> bool:
    with engine.begin() as conn:
        # check exists
        res = conn.execute(select(users.c.username).where(users.c.username == username)).fetchone()
        if res:
            return False
        pw_hash = hash_password_bcrypt(raw_password)
        conn.execute(insert(users).values(username=username, password_hash=pw_hash, created_at=datetime.utcnow()))
    return True

def authenticate(username: str, raw_password: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(select(users.c.password_hash).where(users.c.username == username)).fetchone()
        if not row:
            return False
        stored_hash = row[0]
        return check_password_bcrypt(raw_password, stored_hash)

def append_session_record(row: dict):
    with engine.begin() as conn:
        conn.execute(insert(sessions).values(**row))

def fetch_recent_accuracy(participant_id: str, n: int = 3):
    # returns mean accuracy of last n trials for this participant (or None)
    with engine.connect() as conn:
        df = pd.read_sql(
            select(sessions).where(sessions.c.participant_id == participant_id).order_by(desc(sessions.c.id)).limit(n),
            conn,
        )
    if df.empty:
        return None
    return df["accuracy"].astype(float).mean()

def fetch_all_sessions_df():
    with engine.connect() as conn:
        df = pd.read_sql(select(sessions), conn)
    return df

def delete_all_data():
    with engine.begin() as conn:
        conn.execute(sessions.delete())
        conn.execute(users.delete())
    # also remove CSV mirror if exists
    try:
        if CSV_PATH.exists():
            CSV_PATH.unlink()
    except Exception:
        pass

# -----------------------------
# Session-state initialization
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "consent" not in st.session_state:
    st.session_state.consent = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = str(uuid.uuid4())
if "trial" not in st.session_state:
    st.session_state.trial = 1
if "history" not in st.session_state:
    st.session_state.history = []
if "attention_index" not in st.session_state:
    st.session_state.attention_index = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "assigned_policy" not in st.session_state:
    st.session_state.assigned_policy = None

# -----------------------------
# Authentication UI
# -----------------------------
st.title("Interactive AI Study (Prototype)")

if not st.session_state.logged_in:
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate(login_user, login_pass):
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success("Login successful")
                safe_rerun()
            else:
                st.error("Invalid username or password")
    with tab_register:
        st.subheader("Register")
        new_user = st.text_input("New username", key="reg_user")
        new_pass = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Register"):
            if not new_user or not new_pass:
                st.error("Username and password required")
            elif create_user(new_user, new_pass):
                st.success("Registration successful. You may now log in.")
            else:
                st.error("Username already exists")
    st.stop()

# -----------------------------
# Consent UI (detailed) -- using checkbox (Option A)
# -----------------------------
if not st.session_state.consent:
    st.header("Consent to participate in research")
    st.markdown(
        f"""
This study is a research prototype hosted by the investigator. Please read the following carefully before consenting.

**Purpose:** To study how adaptive explanations affect users' decisions, confidence, and reaction times.  
**Data collected:** pseudonymized trial-level data (trial-level responses, reaction times, confidence, difficulty, agreement), and minimal account username pseudonym (HMAC). No raw personal identifiers beyond username are stored in plaintext. Environment info (package versions) is saved for reproducibility.  
**Storage & retention:** Data are stored locally in `app.db` and mirrored to `collected_sessions.csv`. Data will be retained for research use for up to 5 years or as required by ethics. You can request deletion using the contact below.  
**Withdrawal:** You may withdraw at any time; contact the researcher to request deletion of your data. Withdrawal does not affect your rights.  


By consenting you acknowledge that you are over 18, participation is voluntary, and you allow pseudonymized storage of your trial data for research purposes.
        """
    )
    # checkbox uses a session key so its state persists across reruns
    consent_checked = st.checkbox("I have read the information above and consent to participate", key="consent_checkbox")
    if consent_checked:
        st.session_state.consent = True

    # if still not consented, stop; once checked, Streamlit reruns and code continues
    if not st.session_state.consent:
        st.stop()

# -----------------------------
# Sidebar (settings + admin)
# -----------------------------
st.sidebar.header("Study settings")
policy_override = st.sidebar.selectbox("Default explanation policy (researcher override)", ["random", "static", "adaptive"])
n_trials = st.sidebar.slider("Trials", 3, 15, 6)

st.sidebar.markdown("### Files / Info")
st.sidebar.code(str(DB_PATH))
st.sidebar.markdown(f"CSV mirror: {CSV_PATH}")
st.sidebar.markdown(f"Logged in as: **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    # remove only relevant keys
    for k in ["logged_in", "username", "participant_id", "trial", "history", "consent", "assigned_policy"]:
        st.session_state.pop(k, None)
    safe_rerun()

# Admin login
st.sidebar.markdown("---")
st.sidebar.subheader("Researcher admin")
admin_pw = st.sidebar.text_input("Admin password", type="password", key="admin_pw")
is_admin = False
if admin_pw:
    if check_password_bcrypt(admin_pw, ADMIN_PASSWORD_HASH):
        is_admin = True
    else:
        st.sidebar.warning("Admin auth failed (change admin password file!)")

# -----------------------------
# Helper: explanation logic
# -----------------------------
def explanation(policy: str, recent_acc, confidence: float) -> str:
    if policy == "static":
        return "The model uses features A, B, and C to generate its prediction."
    # adaptive: if recent_acc low, give more detail; if low confidence, expand
    if recent_acc is not None and recent_acc < 0.6:
        return (
            "Because recent accuracy is low, this explanation is more detailed: "
            "Feature A contributes positively, Feature B moderates uncertainty, "
            "and Feature C adjusts the final score."
        )
    if confidence < 0.6:
        return (
            "Since your confidence is low, here is additional detail: "
            "Feature A dominates the decision, Feature B reduces noise, "
            "and Feature C balances the outcome."
        )
    return "The model uses features A, B, and C to generate its prediction."

# -----------------------------
# Assign policy for participant (counterbalanced)
# -----------------------------
if st.session_state.assigned_policy is None:
    if policy_override != "random":
        st.session_state.assigned_policy = policy_override
    else:
        # assign by pseudonym deterministic random to maintain counterbalance across sessions
        phash = username_hmac(st.session_state.username)
        st.session_state.assigned_policy = ["static", "adaptive"][int(phash, 16) % 2]

# -----------------------------
# Attention check: set index if not already
# -----------------------------
if st.session_state.attention_index is None:
    # choose a trial index for attention check (not first)
    if n_trials >= 3:
        st.session_state.attention_index = np.random.randint(2, n_trials)  # 1-indexed trials will be compared
    else:
        st.session_state.attention_index = None

# -----------------------------
# Main experiment UI (using st.form for cleaner RT measures)
# -----------------------------
st.header("Adaptive Explanation Study")
st.write(f"Participant ID: `{st.session_state.participant_id}`")
st.write(f"Trial {st.session_state.trial} of {n_trials}")

# compute recent accuracy from local history or DB
recent_acc_local = (
    np.mean([h["accuracy"] for h in st.session_state.history[-3:]])
    if len(st.session_state.history) >= 3 else None
)
# fallback to DB query
if recent_acc_local is None:
    recent_acc_local = fetch_recent_accuracy(st.session_state.participant_id, n=3)

# show assigned policy
st.info(f"Assigned explanation policy for this participant: **{st.session_state.assigned_policy}**")

if st.session_state.trial <= n_trials:
    # Start timer when form appears
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    with st.form(key=f"trial_form_{st.session_state.trial}"):
        difficulty = st.selectbox("Trial difficulty", ["easy", "hard"], key=f"difficulty_{st.session_state.trial}")
        st.markdown("**Model recommendation:** Option A")
        confidence = st.slider("Your confidence", 0.0, 1.0, 0.7, 0.05, key=f"confidence_{st.session_state.trial}")
        st.write(explanation(st.session_state.assigned_policy, recent_acc_local, confidence))

        agree = st.radio("Do you agree with the model?", ["Agree", "Disagree"], key=f"agree_{st.session_state.trial}")

        # optionally insert attention check
        attention_failed = False
        if st.session_state.attention_index is not None and st.session_state.trial == int(st.session_state.attention_index):
            st.markdown("**Attention check:** To show you are paying attention, please select 'Disagree' for this question.")
            att_choice = st.radio("Attention check: Do you read the instructions?", ["Agree", "Disagree"], key=f"att_{st.session_state.trial}")
            if att_choice != "Disagree":
                attention_failed = True

        submitted = st.form_submit_button("Submit trial")
        if submitted:
            rt_ms = int((time.time() - st.session_state.start_time) * 1000)
            # simulate model correctness probabilistically
            rng = np.random.RandomState(seed=int(hashlib.sha256(st.session_state.participant_id.encode()).hexdigest(), 16) % (2**32) + st.session_state.trial)
            model_correct = rng.rand() < (0.75 if difficulty == "easy" else 0.55)
            user_correct = (agree == "Agree" and model_correct) or (agree == "Disagree" and not model_correct)
            row = {
                "pseudonym": pseudonym_from_username(st.session_state.username),
                "username_hmac": username_hmac(st.session_state.username),
                "participant_id": st.session_state.participant_id,
                "trial": st.session_state.trial,
                "policy": st.session_state.assigned_policy,
                "difficulty": difficulty,
                "agreement": agree,
                "confidence": float(confidence),
                "accuracy": int(user_correct),
                "reaction_time_ms": int(rt_ms),
                "attention_check_failed": bool(attention_failed),
                "timestamp_utc": datetime.utcnow().isoformat(),
                "app_version": APP_VERSION,
                "env_info": json.dumps(get_env_info()),
            }
            # write to SQLite (primary storage)
            append_session_record(row)
            # mirror to CSV (secondary flat-file export)
            try:
                append_to_csv(row, CSV_PATH)
            except Exception as e:
                st.warning(f"Could not append to CSV mirror: {e}")
            # keep local session copy
            st.session_state.history.append(row)
            # update counters
            st.session_state.trial += 1
            st.session_state.start_time = None
            safe_rerun()

else:
    st.success("Session complete. Thank you for participating.")
    df = pd.DataFrame(st.session_state.history)
    if not df.empty:
        st.dataframe(df)
        st.write(f"Accuracy: {df['accuracy'].mean():.2f}")
        st.write(f"Mean confidence: {df['confidence'].mean():.2f}")
        st.write(f"Median RT (ms): {int(df['reaction_time_ms'].median())}")
    if st.button("Start new session (new participant id)"):
        st.session_state.participant_id = str(uuid.uuid4())
        for k in ["trial", "history", "start_time", "attention_index", "assigned_policy"]:
            st.session_state.pop(k, None)
        safe_rerun()

# -----------------------------
# Admin / Researcher view
# -----------------------------
if is_admin:
    st.markdown("---")
    st.header("Researcher admin panel")
    st.write("You are authenticated as admin. Use these controls to inspect and export data.")
    df_all = fetch_all_sessions_df()
    st.write(f"Total records: {len(df_all)}")
    st.dataframe(df_all.head(200))

    # raw CSV download (mirror)
    if CSV_PATH.exists():
        try:
            csv_bytes = CSV_PATH.read_bytes()
            st.download_button("Download CSV mirror (collected_sessions.csv)", csv_bytes, "collected_sessions.csv")
        except Exception:
            st.warning("Unable to prepare CSV download.")

    # anonymized download: drop username_hmac and env_info
    if not df_all.empty:
        csv_bytes_raw = df_all.to_csv(index=False).encode()
        st.download_button("Download raw DB CSV (current)", csv_bytes_raw, "sessions_raw.csv")
        df_anon = df_all.copy()
        df_anon = df_anon.drop(columns=["username_hmac", "env_info"], errors="ignore")
        st.download_button("Download anonymized CSV (for sharing)", df_anon.to_csv(index=False).encode(), "sessions_anonymized.csv")

    # safe two-step deletion confirmation
    if st.button("Delete all data (users + sessions)"):
        st.session_state["_confirm_delete"] = True

    if st.session_state.get("_confirm_delete", False):
        st.warning("Deleting all data is irreversible. Click 'Confirm delete' to proceed or 'Cancel' to abort.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm delete"):
                delete_all_data()
                st.success("All data deleted. Restarting...")
                st.session_state["_confirm_delete"] = False
                safe_rerun()
        with col2:
            if st.button("Cancel delete"):
                st.session_state["_confirm_delete"] = False
                safe_rerun()

# -----------------------------
# Footer / GitHub link & reproducibility
# -----------------------------
st.markdown("---")
st.markdown(
    """
**Reproducibility & GitHub**  
Repository (example): https://github.com/shinejose0007/interactive-ai-prototype  
The repo should contain: `streamlit_app.py`, `requirements.txt`, `README.md` with run instructions, `analysis.ipynb` demonstrating mixed-effects modeling (e.g., `statsmodels` or `lme4` in R), and the preregistration link.
"""
)
st.caption("App version: " + APP_VERSION)
