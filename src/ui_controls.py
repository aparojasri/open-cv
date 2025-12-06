import streamlit as st


def get_ui_config():
    """
    All sidebar controls in one place.
    Returns a dict consumed by the VideoProcessor in app.py.
    """
    st.sidebar.title("⚙️ Vision Debug Controls")

    # Debug / developer options
    debug = st.sidebar.checkbox("Enable Debug Mode", value=True)

    # Performance
    st.sidebar.subheader("Performance")
    process_every = st.sidebar.slider("Process every N frames", 1, 8, 2)
    downscale_width = st.sidebar.slider("Downscale width", 320, 1280, 640)

    # Drawing toggles
    st.sidebar.subheader("Drawing Options")
    draw_face = st.sidebar.checkbox("Draw Face Mesh", value=True)
    draw_hands = st.sidebar.checkbox("Draw Hand Landmarks", value=True)
    draw_gestures = st.sidebar.checkbox("Show Gesture Labels", value=True)

    # Face ID controls (future)
    st.sidebar.subheader("Face ID")
    enable_face_id = st.sidebar.checkbox("Enable Face Identification", value=False)
    draw_face_id = st.sidebar.checkbox(
        "Show Face ID Overlay",
        value=True,
        disabled=not enable_face_id,
    )

    register_new_user = None
    if enable_face_id:
        with st.sidebar.expander("Register New User"):
            name = st.text_input("Enter user name", key="face_reg_name")
            if st.button("Register current face", key="face_reg_button"):
                if name.strip():
                    register_new_user = name.strip()

    # Logging / telemetry
    st.sidebar.subheader("Logging")
    enable_logging = st.sidebar.checkbox("Enable Telemetry Logging", value=True)

    # Profiler HUD
    st.sidebar.subheader("Performance Metrics Overlay")
    show_profiler = st.sidebar.checkbox("Show FPS / Latency HUD", value=True)

    config = {
        "debug": debug,
        "process_every": process_every,
        "downscale_width": downscale_width,
        "draw_face": draw_face,
        "draw_hands": draw_hands,
        "draw_gestures": draw_gestures,
        "enable_face_id": enable_face_id,
        "draw_face_id": draw_face_id and enable_face_id,
        "register_new_user": register_new_user,
        "enable_logging": enable_logging,
        "show_profiler": show_profiler,
    }

    return config
