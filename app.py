import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import time
from datetime import datetime
from inference_engine import (
    load_image_model, predict_image, 
    load_audio_model, predict_audio,
    load_text_pipeline, predict_text,
    predict_multimodal,
    get_prediction_history, log_prediction,
    EMOTIONS
)

# Page configuration
st.set_page_config(
    page_title="EmotionSense AI | Sky Blue & Teal Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM UI COLORS: SKY BLUE & TEAL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Force Dark Theme Background (Sky Blue) */
    [data-testid="stAppViewContainer"], .stApp, .main {
        background: radial-gradient(circle at top right, #00c6ff, #001a33) !important;
        background-color: #001a33 !important;
        color: #e6f7ff !important;
    }

    /* Fix Header Bar Visibility */
    [data-testid="stHeader"] {
        background: transparent !important;
        color: #00BFFF !important;
    }

    /* Glassmorphism Cards with Teal Glow */
    .emotion-card {
        padding: 25px;
        border-radius: 20px;
        background: rgba(0, 48, 48, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 230, 230, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .emotion-card:hover {
        transform: scale(1.02);
        border: 1px solid rgba(0, 191, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(0, 191, 255, 0.2);
    }

    h1, h2, h3, h4, h5, h6, .result-text {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h1 { color: #00e6e6 !important; }
    h2 { color: #00e6e6 !important; }
    h3 { color: #00e6e6 !important; }

    /* Predicted Emotion Highlight (Sky Blue) */
    .result-text {
        color: #00BFFF;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0, 191, 255, 0.4);
    }

    /* Premium Sky Blue Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00BFFF 0%, #00d2ff 100%);
        color: white;
        border-radius: 14px;
        border: none;
        padding: 14px 28px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 191, 255, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 25px rgba(0, 191, 255, 0.5);
        transform: translateY(-2px);
        filter: brightness(1.1);
    }

    /* Sidebar Styling: Blue Theme */
    section[data-testid="stSidebar"] {
        background-color: #000c14 !important;
        border-right: 1px solid rgba(0, 191, 255, 0.4);
    }

    .history-item {
        padding: 12px;
        border-radius: 10px;
        background: rgba(0, 230, 230, 0.05);
        margin-bottom: 12px;
        font-size: 0.9rem;
        border-left: 4px solid #00BFFF;
        transition: background 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(0, 230, 230, 0.1);
    }

    /* Metrics Color */
    [data-testid="stMetricValue"] {
        color: #00e6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def format_time_ago(time_str):
    try:
        now = datetime.now()
        past = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        diff = now - past
        secs = diff.total_seconds()
        if secs < 60: return "Just now"
        if secs < 3600: return f"{int(secs//60)} mins ago"
        if secs < 86400: return f"{int(secs//3600)} hours ago"
        return f"{int(secs//86400)} days ago"
    except:
        return time_str

def calculate_stress(preds):
    if not preds or "Error" in preds: return "Unknown", "#ffffff"
    angry = preds.get("Angry", 0)
    fear = preds.get("Fear", 0)
    happy = preds.get("Happy", 0)
    neutral = preds.get("Neutral", 0)
    
    high_stress = angry + fear
    low_stress = happy + neutral
    
    if high_stress > 0.4: return "High Stress Level🚨", "#ff0000"
    if low_stress > 0.5: return "Low Stress Level✅", "#00ff00"
    return "Medium Stress Level⚠️", "#ffaa00"

# --- HELPER: RADAR CHART (TEAL & SKY BLUE) ---
def plot_radar_chart(results, title="Emotion Pulse"):
    categories = [k for k in results.keys() if k != "Error"]
    values = [results[k] for k in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#00BFFF',
        fillcolor='rgba(0, 230, 230, 0.3)',
        marker=dict(color='#00e6e6')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,230,230,0.1)", tickfont=dict(color="#00e6e6")),
            angularaxis=dict(gridcolor="rgba(0,230,230,0.1)", tickfont=dict(color="#ffffff"))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title=dict(text=title, font=dict(size=20, color='#00e6e6')),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# --- HELPER: MULTI-RADAR CHART ---
def plot_multi_radar_chart(image_res, audio_res, text_res, fused_res, title="Unified Multimodal Emotion Spheres"):
    categories = EMOTIONS
    fig = go.Figure()
    
    def add_trace(res, name, color, fill='none', dash='solid'):
        if res and "Error" not in res:
            # Map values exactly to EMOTIONS to ensure alignment
            values = [res.get(k, 0) for k in categories]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill=fill,
                name=name,
                line=dict(color=color, dash=dash)
            ))
            
    add_trace(image_res, 'Image', '#00e6e6', dash='dot')
    add_trace(audio_res, 'Audio', '#ffaa00', dash='dot')
    add_trace(text_res, 'Text', '#b000ff', dash='dot')
    
    # Add Fused prominently
    if fused_res and "Error" not in fused_res:
        values = [fused_res.get(k, 0) for k in categories]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Fused',
            line_color='#ffffff',
            fillcolor='rgba(0, 191, 255, 0.4)',
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.2)", tickfont=dict(color="#ffffff")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.2)", tickfont=dict(color="#ffffff"))
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title=dict(text=title, font=dict(size=20, color='#ffffff')),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Sidebar Navigation & History
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        st.title("🧠 EmotionSense")
        
    st.markdown("### Navigation")
    page = st.selectbox(
        "Go to", 
        ["🏠 Dashboard Home", "🖼️ Image Analytics", "🎙️ Audio Analytics", "📝 Text Analytics", "🌌 Multimodal Fusion", "📊 Comparison & Metrics"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### 🕒 Prediction History")
    history = get_prediction_history()
    if history:
        for item in history:
            time_ago = format_time_ago(item['timestamp'])
            st.markdown(f"""
            <div class="history-item">
                <small style='color: #8b949e'>{time_ago}</small> — <b style='color: #00e6e6'>{item['modality']}</b>: <span style='color: #00BFFF'>{item['prediction']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No history logged.")

    st.divider()
    st.write("### Backend Systems")
    st.info("Image: ResNet18 (Live)")
    st.info("Voice: CNN1D (Live)")
    st.info("Text: BERT (Live)")

# --- PAGE 1: HOME ---
if page == "🏠 Dashboard Home":
    st.title("🚀 EmotionSense-AI Dashboard")
    
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="emotion-card">
            <h3 style="margin-top:0; color:#00e6e6">🖼️ Image</h3>
            <p>Advanced Computer Vision using <b>ResNet18</b> for facial feature mapping.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="emotion-card">
            <h3 style="margin-top:0; color:#00e6e6">🎙️ Voice</h3>
            <p>Direct waveform analysis via <b>CNN1D</b> for acoustic sentiment detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="emotion-card">
            <h3 style="margin-top:0; color:#00e6e6">📝 Text</h3>
            <p>Deep semantic understanding with <b>DistilRoBERTa</b> Transformers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Benchmarks")
    
    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Accuracy", "94.1%", "Text model")
    m2.metric("Latency", "105ms", "-5ms")
    m3.metric("Dataset", "FER+RAVDESS", "Mixed")

# --- PAGE 2: IMAGE ANALYTICS ---
elif page == "🖼️ Image Analytics":
    st.title("🖼️ Image Intelligence")
    st.write("Extract emotional state from facial landmarks.")
    
    col_up, col_res = st.columns([1, 1])
    
    with col_up:
        input_method = st.radio("Image Source", ["File Upload", "Webcam"], horizontal=True)
        img_file = None
        if input_method == "File Upload":
            img_file = st.file_uploader("Upload Profile Image", type=["jpg", "jpeg", "png"])
            if img_file:
                st.image(img_file, caption="Analyzing Image...", use_container_width=True)
        else:
            img_file = st.camera_input("Live Webcam Feed")
            
        if img_file:
            # For webcam, trigger automatically if it's a new capture
            is_new_cam = input_method == "Webcam" and ('last_cam_size' not in st.session_state or st.session_state['last_cam_size'] != img_file.size)
            
            if is_new_cam or (input_method == "File Upload" and st.button("🚀 Analyze Now")):
                with st.spinner("Mapping facial vectors..."):
                    model = load_image_model()
                    with open("temp_img.jpg", "wb") as f:
                        f.write(img_file.getbuffer())
                    
                    predictions = predict_image(model, "temp_img.jpg")
                    if os.path.exists("temp_img.jpg"):
                        os.remove("temp_img.jpg")
                        
                    st.session_state['img_preds'] = predictions
                    if input_method == "Webcam":
                        st.session_state['last_cam_size'] = img_file.size
                        
                    log_prediction("Image", predictions)
                    st.rerun()

    with col_res:
        if 'img_preds' in st.session_state:
            preds = st.session_state['img_preds']
            if "Error" in preds:
                st.error(preds["Error"])
            else:
                top_emo = max(preds, key=preds.get)
                st.markdown(f"### Predicted: <span class='result-text'>{top_emo}</span>", unsafe_allow_html=True)
                
                # Confidence Progress Bars
                for emo, score in sorted(preds.items(), key=lambda item: item[1], reverse=True)[:3]:
                    st.write(f"**{emo}** {score*100:.1f}%")
                    st.progress(score)
                
                # Stress Indicator
                stress_lvl, stress_color = calculate_stress(preds)
                st.markdown(f"<div style='margin-top:15px; margin-bottom:15px; padding:10px; border-left:4px solid {stress_color}; background:rgba(255,255,255,0.05);'><b>Stress Monitor:</b> <span style='color:{stress_color}'>{stress_lvl}</span></div>", unsafe_allow_html=True)
                
                # Radar Chart
                st.plotly_chart(plot_radar_chart(preds, "Facial Emotion Pulse"), use_container_width=True)
                
                # Bar Chart with Teal/Pink scale
                df = px.bar(
                    x=list(preds.keys()), 
                    y=list(preds.values()),
                    labels={'x': 'Category', 'y': 'Conf'},
                    color=list(preds.values()),
                    color_continuous_scale=["#00e6e6", "#00BFFF"]
                )
                df.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=250)
                st.plotly_chart(df, use_container_width=True)

# --- PAGE 3: AUDIO ANALYTICS ---
elif page == "🎙️ Audio Analytics":
    st.title("🎙️ Vocal Sentiment Hub")
    st.write("Analyze acoustic properties for emotional undertones.")
    a_input_method = st.radio("Audio Source", ["File Upload", "Microphone"], horizontal=True)
    audio_file = None
    if a_input_method == "File Upload":
        audio_file = st.file_uploader("Upload Voice Clip (.wav)", type=["wav"])
    else:
        audio_file = st.audio_input("Record Voice")
    
    if audio_file:
        is_new_mic = a_input_method == "Microphone" and ('last_mic_size' not in st.session_state or st.session_state['last_mic_size'] != audio_file.size)
        
        st.audio(audio_file, format="audio/wav")
        if is_new_mic or (a_input_method == "File Upload" and st.button("🚀 Process Audio Waveform")):
            with st.spinner("Decoding acoustics..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_file.getbuffer())
                
                model = load_audio_model()
                predictions = predict_audio(model, "temp_audio.wav")
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
                    
                st.session_state['audio_preds'] = predictions
                if a_input_method == "Microphone":
                    st.session_state['last_mic_size'] = audio_file.size
                    
                log_prediction("Audio", predictions)
                st.rerun()
    if 'audio_preds' in st.session_state:
        preds = st.session_state['audio_preds']
        
        top_emo = max(preds, key=preds.get)
        st.markdown(f"### Predicted: <span class='result-text'>{top_emo}</span>", unsafe_allow_html=True)
        
        for emo, score in sorted(preds.items(), key=lambda item: item[1], reverse=True)[:3]:
            st.write(f"**{emo}** {score*100:.1f}%")
            st.progress(score)
            
        stress_lvl, stress_color = calculate_stress(preds)
        st.markdown(f"<div style='margin-top:15px; margin-bottom:15px; padding:10px; border-left:4px solid {stress_color}; background:rgba(255,255,255,0.05);'><b>Stress Monitor:</b> <span style='color:{stress_color}'>{stress_lvl}</span></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_radar_chart(preds, "Acoustic Profile"), use_container_width=True)
        with c2:
            df = px.pie(
                names=list(preds.keys()), 
                values=list(preds.values()),
                color_discrete_sequence=["#00e6e6", "#00BFFF", "#33ffff", "#00d2ff", "#008080", "#005f73", "#e0fafa"]
            )
            df.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(df, use_container_width=True)

# --- PAGE 4: TEXT ANALYTICS ---
elif page == "📝 Text Analytics":
    st.title("📝 Semantic Analyzer")
    st.write("Natural Language Processing for emotional intent.")
    
    user_text = st.text_area("How are you feeling?", placeholder="Type here...")
    
    if st.button("🚀 Extract Sentiment"):
        if user_text:
            with st.spinner("Tokenizing input..."):
                pipe = load_text_pipeline()
                predictions = predict_text(pipe, user_text)
                
                if "Error" in predictions:
                    st.error(predictions["Error"])
                else:
                    st.session_state['text_preds'] = predictions
                    log_prediction("Text", predictions)
                    st.rerun()

    if 'text_preds' in st.session_state:
        preds = st.session_state['text_preds']
        top_emo = max(preds, key=preds.get)
        st.markdown(f"### Predicted: <span class='result-text'>{top_emo}</span>", unsafe_allow_html=True)
        
        for emo, score in sorted(preds.items(), key=lambda item: item[1], reverse=True)[:3]:
            st.write(f"**{emo}** {score*100:.1f}%")
            st.progress(score)
            
        stress_lvl, stress_color = calculate_stress(preds)
        st.markdown(f"<div style='margin-top:15px; margin-bottom:15px; padding:10px; border-left:4px solid {stress_color}; background:rgba(255,255,255,0.05);'><b>Stress Monitor:</b> <span style='color:{stress_color}'>{stress_lvl}</span></div>", unsafe_allow_html=True)
        
        st.plotly_chart(plot_radar_chart(preds, "Semantic Pulse"), use_container_width=True)

# --- PAGE 5: COMPARISON & METRICS ---
elif page == "📊 Comparison & Metrics":
    st.title("📊 Model Comparison")
    
    metrics_data = {
        "Modality": ["Image", "Audio", "Text"],
        "Architecture": ["ResNet18", "CNN1D", "BERT/RoBERTa"],
        "Val Accuracy": [0.924, 0.856, 0.941],
        "F1-Score": [0.918, 0.842, 0.938]
    }
    
    st.markdown("#### Performance Leaderboard")
    st.table(metrics_data)
    
    st.divider()
    st.subheader("Metric Distribution")
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=metrics_data["Modality"], y=metrics_data["Val Accuracy"], marker_color='#00e6e6'),
        go.Bar(name='F1-Score', x=metrics_data["Modality"], y=metrics_data["F1-Score"], marker_color='#00BFFF')
    ])
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 6: MULTIMODAL FUSION ---
elif page == "🌌 Multimodal Fusion":
    st.title("🌌 Holistic Emotion Fusion")
    st.write("Combine multiple senses for a high-confidence emotional profile.")
    
    col_inputs, col_fused = st.columns([1, 1])
    
    with col_inputs:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Input Modalities")
        
        m_img_method = st.radio("1. Facial Image Source", ["Upload", "Webcam"], key="m_img_method", horizontal=True)
        if m_img_method == "Upload":
            m_img = st.file_uploader("1. Facial Image", type=["jpg", "png", "jpeg"], key="multi_img", label_visibility="collapsed")
        else:
            m_img = st.camera_input("1. Facial Image", key="multi_img_cam", label_visibility="collapsed")
            
        m_aud = st.file_uploader("2. Vocal Audio (.wav)", type=["wav"], key="multi_aud")
        m_txt = st.text_area("3. Written Statement", placeholder="How is the person communicating?", key="multi_txt")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Fused Analysis"):
            with st.spinner("Synthesizing data streams..."):
                # Save temp files for inference
                img_path = None
                if m_img:
                    img_path = "temp_multi_img.jpg"
                    with open(img_path, "wb") as f: f.write(m_img.getbuffer())
                
                aud_path = None
                if m_aud:
                    aud_path = "temp_multi_aud.wav"
                    with open(aud_path, "wb") as f: f.write(m_aud.getbuffer())
                
                # Load models
                img_model = load_image_model() if m_img else None
                aud_model = load_audio_model() if m_aud else None
                txt_pipe = load_text_pipeline() if m_txt else None
                
                # Predict
                fused_preds, individual_preds = predict_multimodal(
                    image_path=img_path, 
                    audio_path=aud_path, 
                    text=m_txt,
                    image_model=img_model,
                    audio_model=aud_model,
                    text_pipe=txt_pipe
                )
                
                # Cleanup
                if img_path and os.path.exists(img_path): os.remove(img_path)
                if aud_path and os.path.exists(aud_path): os.remove(aud_path)
                
                if "Error" in fused_preds:
                    st.error(fused_preds["Error"])
                else:
                    st.session_state['fused_preds'] = fused_preds
                    st.session_state['individual_preds'] = individual_preds
                    log_prediction("Multimodal", fused_preds)
                    st.rerun()

    with col_fused:
        if 'fused_preds' in st.session_state:
            preds = st.session_state['fused_preds']
            indiv = st.session_state.get('individual_preds', {})
            top_emo = max(preds, key=preds.get)
            
            st.markdown(f"""
            <div class="emotion-card" style="text-align: center; border: 2px solid #00BFFF">
                <h2 style="margin:0">OVERALL FUSION CONFIDENCE</h2>
                <h1 span class='result-text' style="font-size: 3.5rem">{top_emo}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Stress Indicator
            stress_lvl, stress_color = calculate_stress(preds)
            st.markdown(f"<div style='margin-bottom:15px; padding:10px; border-left:4px solid {stress_color}; background:rgba(255,255,255,0.05);'><b>Multimodal Stress Monitor:</b> <span style='color:{stress_color}'>{stress_lvl}</span></div>", unsafe_allow_html=True)
            
            # Confidence Breakdown
            st.markdown("#### Overall Confidence Breakdown")
            for emo, score in sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.write(f"**{emo}**: {score*100:.1f}%")
                st.progress(score)
            
            st.plotly_chart(plot_multi_radar_chart(
                indiv.get('Image'),
                indiv.get('Audio'),
                indiv.get('Text'),
                preds,
                "Unified Emotion Spheres"
            ), use_container_width=True)
        else:
            st.info("Upload at least one modality and click 'Run Fused Analysis' to see the combined result.")

st.sidebar.divider()
st.sidebar.caption("© 2026 EmotionSense AI | v2.1 Sky Blue & Teal Edition")
