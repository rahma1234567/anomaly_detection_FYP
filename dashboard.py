import sys
import warnings
from pathlib import Path
import streamlit as st
from components.authentication.authentication import authenticate                        
from components import upload_page, investigation_page, logs_page, retrain_page  
from logger import log_info

#Allow import of folders
sys.path.append(str(Path(__file__).resolve().parent))

#page layout & authentication
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",)
authenticator, name, username, role = authenticate()
 
#Sidebar - user info and page selection
st.sidebar.markdown(f"**{name}**  \n_{role}_")
authenticator.logout("Log out", "sidebar")
st.sidebar.markdown("---")
 
pages = ["Upload & detect", "Investigate"]  # Page selection
if role == "Admin":
    pages += ["Audit logs", "Retrain models"]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = pages[0]
page = st.sidebar.radio("Page", pages, key="current_page")
 
# Model picker - isolation forest and autoencoder
st.sidebar.markdown("---")
st.sidebar.subheader("Model")
model_choice = st.sidebar.selectbox(
    "Detector",
    ["Isolation Forest", "Autoencoder"],
    help="Choose which model you want to use for anomaly detection",
    key="model_choice",)
 
st.title("Anomaly Detection Dashboard")
 
if page == "Upload & detect":
    upload_page.render(model_choice, username)
elif page == "Investigate":
    investigation_page.render()
elif page == "Audit logs" and role == "Admin":
    logs_page.render()
elif page == "Retrain models" and role == "Admin":
    retrain_page.render(username)
else:
    st.error("You don't have permission to view this page.")