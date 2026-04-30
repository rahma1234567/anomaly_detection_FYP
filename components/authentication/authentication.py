import os
import sys
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logger import log_info, log_warning  # noqa: E402

CRED_PATH = Path(__file__).resolve().parent / "credentials.yaml"


def authenticate():
    if not CRED_PATH.exists():
        st.error(f"Missing credentials file: {CRED_PATH}")
        st.stop()

    with open(CRED_PATH) as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],)

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    name        = st.session_state.get("name")
    username    = st.session_state.get("username")

    if auth_status is False:
        st.error("Username or password is incorrect.")
        log_warning(f"Failed login attempt: username={username!r}")
        st.stop()
    if auth_status is None:
        st.warning("Please log in to continue.")
        st.stop()

    role = config["credentials"]["usernames"][username]["role"]
    session_key = f"_login_logged_{username}"
    if session_key not in st.session_state:
        log_info(f"User '{username}' logged in as {role}")
        st.session_state[session_key] = True

    return authenticator, name, username, role