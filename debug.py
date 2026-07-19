import streamlit as st

st.title("Debug Test")

try:
    import config
    st.success("✓ config.py")
except Exception as e:
    st.error(f"✗ config.py: {e}")

try:
    import vcds_engine
    st.success("✓ vcds_engine.py")
except Exception as e:
    st.error(f"✗ vcds_engine.py: {e}")

try:
    import audio_engine_diagnosis
    st.success("✓ audio_engine_diagnosis.py")
except Exception as e:
    st.error(f"✗ audio_engine_diagnosis.py: {e}")

try:
    st.write("Secrets:", list(st.secrets.keys()) if st.secrets else "No secrets")
except Exception as e:
    st.error(f"Secrets error: {e}")
