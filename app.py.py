"""
========================================================================
AI COLOR PSYCHOLOGY ANALYZER
========================================================================
A beginner-friendly Streamlit application that analyzes the psychology
of a user-selected color, predicts a likely audience persona, suggests
design use-cases, classifies the color into Warm/Cool/Neutral using a
simple Machine Learning model (KMeans / Decision Tree), and lets the
user export a text report.

Tech Stack: Python, Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn

Author: (Your Name) - Internship Project
========================================================================
"""

# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import io
import os

# ------------------------------------------------------------------
# 2. PAGE CONFIGURATION (must be the first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Color Psychology Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 3. SIMPLE CUSTOM CSS FOR A MODERN LOOK
# ------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #f7f8fa;
    }
    /* Card-style containers */
    .custom-card {
        background-color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    /* Color preview swatch */
    .color-swatch {
        height: 130px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(0,0,0,0.08);
    }
    /* Headings */
    h1, h2, h3 {
        font-family: 'Trebuchet MS', sans-serif;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1f2937;
    }
    section[data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4. LOAD SAMPLE DATASET
# ------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """
    Loads the sample color-psychology dataset (CSV).
    The CSV contains: color name, hex code, RGB values, three emotions,
    a target persona, a recommended use-case, and a temperature label
    (Warm / Cool / Neutral) used as ground-truth for the ML model.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "color_psychology_data.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_dataset()

# ------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ------------------------------------------------------------------

def hex_to_rgb(hex_color: str):
    """Convert a hex color string (#RRGGBB) into an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def find_closest_color(rgb, dataset):
    """
    Find the closest matching color in our dataset using simple
    Euclidean distance on the RGB values. This lets us "look up"
    psychological meaning even for colors not exactly in the table.
    """
    r, g, b = rgb
    dataset = dataset.copy()
    dataset["distance"] = np.sqrt(
        (dataset["r"] - r) ** 2 +
        (dataset["g"] - g) ** 2 +
        (dataset["b"] - b) ** 2
    )
    closest_row = dataset.sort_values("distance").iloc[0]
    return closest_row


@st.cache_resource
def train_decision_tree(dataset):
    """
    Train a simple Decision Tree Classifier that predicts the color
    'temperature' category (Warm / Cool / Neutral) from RGB values.
    This demonstrates a basic supervised ML workflow.
    """
    X = dataset[["r", "g", "b"]]
    y = dataset["temperature"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y_encoded)
    return model, le


@st.cache_resource
def train_kmeans(dataset):
    """
    Train a KMeans clustering model (unsupervised ML) on RGB values
    to group colors into 3 clusters. We then map each cluster to the
    majority temperature label present in that cluster, so the result
    is still human-readable (Warm / Cool / Neutral).
    """
    X = dataset[["r", "g", "b"]].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    dataset = dataset.copy()
    dataset["cluster"] = clusters

    # Map each cluster number -> most common temperature label in that cluster
    cluster_to_label = (
        dataset.groupby("cluster")["temperature"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return kmeans, cluster_to_label


dt_model, label_encoder = train_decision_tree(df)
kmeans_model, cluster_label_map = train_kmeans(df)


def classify_color_ai(rgb, method="Decision Tree"):
    """
    Classify a given RGB color into Warm / Cool / Neutral using the
    chosen ML method.
    """
    X_new = np.array(rgb).reshape(1, -1)

    if method == "Decision Tree":
        pred_encoded = dt_model.predict(X_new)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
    else:  # KMeans
        cluster = kmeans_model.predict(X_new)[0]
        prediction = cluster_label_map[cluster]

    return prediction


def generate_report(color_hex, rgb, closest, ai_category, persona, use_case):
    """
    Build a plain-text analysis report that the user can download.
    """
    report = io.StringIO()
    report.write("=" * 60 + "\n")
    report.write("        AI COLOR PSYCHOLOGY ANALYZER - REPORT\n")
    report.write("=" * 60 + "\n")
    report.write(f"Generated on : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.write(f"Selected Color (HEX) : {color_hex}\n")
    report.write(f"RGB Value             : {rgb}\n")
    report.write(f"Closest Known Color   : {closest['color_name']}\n\n")

    report.write("-- Emotional Associations --\n")
    report.write(f"  • {closest['emotion_1']}\n")
    report.write(f"  • {closest['emotion_2']}\n")
    report.write(f"  • {closest['emotion_3']}\n\n")

    report.write("-- AI Classification --\n")
    report.write(f"  Color Temperature Group : {ai_category}\n\n")

    report.write("-- Suggested Target Audience --\n")
    report.write(f"  {persona}\n\n")

    report.write("-- Recommended Design Use-Case --\n")
    report.write(f"  {use_case}\n\n")

    report.write("=" * 60 + "\n")
    report.write("Report generated by AI Color Psychology Analyzer\n")
    return report.getvalue()


# ------------------------------------------------------------------
# 6. SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("🎨 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "🧠 Color Psychology",
        "👥 Persona & Design",
        "🤖 AI Classification",
        "📊 Visual Dashboard",
        "📁 Sample Dataset",
        "📥 Export Report",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖌️ Pick Your Color")
selected_color = st.sidebar.color_picker("Choose a color", "#3399FF")

st.sidebar.markdown("---")
ml_method = st.sidebar.selectbox(
    "AI Classification Method",
    ["Decision Tree", "KMeans Clustering"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses simple Machine Learning models "
    "(Decision Tree & KMeans) trained on a sample color-psychology "
    "dataset to generate insights."
)

# ------------------------------------------------------------------
# 7. CORE CALCULATIONS (used across multiple pages)
# ------------------------------------------------------------------
rgb_value = hex_to_rgb(selected_color)
closest_color_row = find_closest_color(rgb_value, df)
ai_temperature = classify_color_ai(rgb_value, method=ml_method)

emotions = [
    closest_color_row["emotion_1"],
    closest_color_row["emotion_2"],
    closest_color_row["emotion_3"],
]
persona_suggestion = closest_color_row["persona"]
use_case_suggestion = closest_color_row["best_use_case"]

# ------------------------------------------------------------------
# 8. PAGE: HOME
# ------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🎨 AI Color Psychology Analyzer")
    st.markdown(
        "Welcome! This tool helps **designers, students, and businesses** "
        "understand the **psychological impact** of colors using basic "
        "AI/Machine Learning concepts."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Selected Color Preview")
        st.markdown(
            f"<div class='color-swatch' style='background-color:{selected_color};'></div>",
            unsafe_allow_html=True
        )
        st.write(f"**HEX:** `{selected_color}`")
        st.write(f"**RGB:** `{rgb_value}`")

    with col2:
        st.markdown("#### Quick Summary")
        st.markdown(f"""
        <div class='custom-card'>
        <b>Closest Known Color:</b> {closest_color_row['color_name']} <br>
        <b>Top Emotion:</b> {emotions[0]} <br>
        <b>AI Temperature Group:</b> {ai_temperature} <br>
        <b>Likely Audience:</b> {persona_suggestion} <br>
        <b>Best Used In:</b> {use_case_suggestion}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🧭 How to use this app
    1. Pick a color from the **sidebar color picker**.
    2. Explore the **Color Psychology** page to see associated emotions.
    3. Check **Persona & Design** for audience and use-case suggestions.
    4. View **AI Classification** to see Warm/Cool/Neutral grouping.
    5. Visit the **Visual Dashboard** for charts.
    6. Download your personalized report from **Export Report**.
    """)

# ------------------------------------------------------------------
# 9. PAGE: COLOR PSYCHOLOGY
# ------------------------------------------------------------------
elif page == "🧠 Color Psychology":
    st.title("🧠 Color Psychology Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div class='color-swatch' style='background-color:{selected_color};'></div>",
            unsafe_allow_html=True
        )
        st.write(f"**HEX:** `{selected_color}`  |  **RGB:** `{rgb_value}`")

    with col2:
        st.markdown(f"""
        <div class='custom-card'>
        <h4>Closest Match: {closest_color_row['color_name']}</h4>
        This color is psychologically associated with:
        <ul>
            <li><b>{emotions[0]}</b></li>
            <li><b>{emotions[1]}</b></li>
            <li><b>{emotions[2]}</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📚 Common Color Meanings (Reference Table)")
    st.dataframe(
        df[["color_name", "hex_code", "emotion_1", "emotion_2", "emotion_3"]],
        use_container_width=True
    )

# ------------------------------------------------------------------
# 10. PAGE: PERSONA & DESIGN RECOMMENDATIONS
# ------------------------------------------------------------------
elif page == "👥 Persona & Design":
    st.title("👥 User Persona & Design Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='custom-card'>
        <h4>🎯 Likely Target Audience</h4>
        <h2 style='color:{selected_color};'>{persona_suggestion}</h2>
        <p>Based on the emotional profile of this color, the audience most
        likely to respond positively is <b>{persona_suggestion}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='custom-card'>
        <h4>🏗️ Recommended Use-Case</h4>
        <h2 style='color:{selected_color};'>{use_case_suggestion}</h2>
        <p>This color works particularly well for a <b>{use_case_suggestion}</b>
        due to the emotions it evokes.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🗂️ All Persona & Use-Case Mappings")
    st.dataframe(
        df[["color_name", "persona", "best_use_case"]],
        use_container_width=True
    )

    st.markdown("""
    #### 💡 General Design Guidelines
    - **Healthcare Apps** → Calming colors like Green, Teal, Blue, Mint, Beige.
    - **Banking Apps** → Trust-building colors like Navy, Blue, Gold, Black, Silver.
    - **E-commerce Websites** → Energetic colors like Red, Orange, Pink, Purple.
    - **Educational Platforms** → Friendly, optimistic colors like Yellow, Sky Blue, Orange.
    """)

# ------------------------------------------------------------------
# 11. PAGE: AI CLASSIFICATION
# ------------------------------------------------------------------
elif page == "🤖 AI Classification":
    st.title("🤖 AI-Based Color Classification")

    st.markdown(f"""
    <div class='custom-card'>
    <h4>Selected Model: {ml_method}</h4>
    <p>Predicted Temperature Group for <code>{selected_color}</code>:</p>
    <h2 style='color:{selected_color};'>{ai_temperature}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### ⚙️ How the AI Models Work
    - **Decision Tree Classifier**: A *supervised* ML model trained on
      labeled RGB → Temperature (Warm/Cool/Neutral) examples from our
      dataset. It learns simple rules (e.g. "if Red value is high and
      Blue is low → Warm") to classify new colors.
    - **KMeans Clustering**: An *unsupervised* ML model that groups
      colors into 3 clusters purely based on RGB similarity. Each
      cluster is then labeled using the majority temperature found
      inside it.
    """)

    st.markdown("### 🌡️ Temperature Distribution in Sample Dataset")
    temp_counts = df["temperature"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    colors_map = {"Warm": "#FF6F61", "Cool": "#4FA8DA", "Neutral": "#A9A9A9"}
    bar_colors = [colors_map.get(t, "#999999") for t in temp_counts.index]
    ax1.bar(temp_counts.index, temp_counts.values, color=bar_colors)
    ax1.set_ylabel("Number of Colors")
    ax1.set_title("Warm vs Cool vs Neutral Colors in Dataset")
    st.pyplot(fig1)

# ------------------------------------------------------------------
# 12. PAGE: VISUAL DASHBOARD
# ------------------------------------------------------------------
elif page == "📊 Visual Dashboard":
    st.title("📊 Visual Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Closest Color", closest_color_row["color_name"])
    col2.metric("AI Category", ai_temperature)
    col3.metric("Target Audience", persona_suggestion)

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    # --- Chart 1: Emotion Intensity Bar Chart (mock intensity scores) ---
    with chart_col1:
        st.markdown("#### 💭 Emotion Association Strength")
        # Simple mock "intensity" scores for visualization purposes
        intensity_scores = [90, 75, 60]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(emotions, intensity_scores, color=selected_color, edgecolor="black")
        ax2.set_xlabel("Relative Strength (%)")
        ax2.set_xlim(0, 100)
        ax2.set_title(f"Emotions Linked to {closest_color_row['color_name']}")
        st.pyplot(fig2)

    # --- Chart 2: RGB Composition Pie Chart ---
    with chart_col2:
        st.markdown("#### 🎨 RGB Composition")
        labels = ["Red", "Green", "Blue"]
        sizes = [rgb_value[0] + 1, rgb_value[1] + 1, rgb_value[2] + 1]  # +1 avoids all-zero
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.pie(
            sizes,
            labels=labels,
            colors=["#FF4C4C", "#4CFF6B", "#4C7CFF"],
            autopct="%1.0f%%",
            startangle=90
        )
        ax3.set_title("RGB Value Breakdown")
        st.pyplot(fig3)

    st.markdown("---")
    st.markdown("#### 🗺️ Color Recommendation Map")
    rec_df = df.groupby("best_use_case")["color_name"].apply(list).reset_index()
    st.dataframe(rec_df, use_container_width=True)

# ------------------------------------------------------------------
# 13. PAGE: SAMPLE DATASET
# ------------------------------------------------------------------
elif page == "📁 Sample Dataset":
    st.title("📁 Sample Dataset Used by the App")
    st.markdown(
        "This dataset powers the lookup logic and trains the AI "
        "classification models used throughout the app."
    )
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Sample Dataset (CSV)",
        data=csv_bytes,
        file_name="color_psychology_data.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------
# 14. PAGE: EXPORT REPORT
# ------------------------------------------------------------------
elif page == "📥 Export Report":
    st.title("📥 Export Your Color Analysis Report")

    st.markdown(
        f"<div class='color-swatch' style='background-color:{selected_color};'></div>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class='custom-card'>
    <b>Color:</b> {selected_color} ({rgb_value}) <br>
    <b>Closest Match:</b> {closest_color_row['color_name']} <br>
    <b>Emotions:</b> {', '.join(emotions)} <br>
    <b>AI Category:</b> {ai_temperature} <br>
    <b>Audience:</b> {persona_suggestion} <br>
    <b>Best Use-Case:</b> {use_case_suggestion}
    </div>
    """, unsafe_allow_html=True)

    report_text = generate_report(
        selected_color, rgb_value, closest_color_row,
        ai_temperature, persona_suggestion, use_case_suggestion
    )

    st.text_area("Preview Report", report_text, height=300)

    st.download_button(
        label="⬇️ Download Report (.txt)",
        data=report_text,
        file_name=f"color_report_{selected_color.lstrip('#')}.txt",
        mime="text/plain"
    )

# ------------------------------------------------------------------
# 15. FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit | AI Color Psychology Analyzer (Internship Project)</center>",
    unsafe_allow_html=True
)
