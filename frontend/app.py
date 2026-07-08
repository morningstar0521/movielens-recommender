"""
Streamlit frontend - polished card-based UI.

WHY Streamlit over a React SPA:
- Single Python file, no build step, no node_modules.
- Hugging Face Spaces and Render both host Streamlit out of the box.
- For a portfolio that focuses on ML, the frontend should not eat
  more than 200 lines of code.

Run:
    streamlit run frontend/app.py
Environment:
    API_URL=http://localhost:8000  (default)
"""

import hashlib
import os
from collections import Counter

import requests
import streamlit as st

# streamlit-keyup reruns on every key release (with a debounce) which
# gives real search-as-you-type behaviour. If the package is missing,
# fall back gracefully to the built-in text_input (fires on Enter).
try:
    from st_keyup import st_keyup  # type: ignore
    _KEYUP_AVAILABLE = True
except ImportError:
    _KEYUP_AVAILABLE = False


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Righteous&family=Share+Tech+Mono&display=swap');

    /* --- Global --- */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Righteous', sans-serif !important;
        letter-spacing: 0.5px;
    }

    .stApp {
        background: radial-gradient(circle at 50% 50%, #161233 0%, #080714 100%) !important;
        color: #F8FAFC;
    }

    /* CRT Scanline Overlay */
    .stApp::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.12) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.02), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.02));
        z-index: 99999;
        background-size: 100% 3px, 3px 100%;
        pointer-events: none;
        opacity: 0.85;
    }

    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* --- Hero --- */
    .hero {
        background: linear-gradient(135deg, #110A24 0%, #1F103A 100%) !important;
        padding: 2rem 2.5rem;
        border-radius: 10px;
        border: 2px solid #FF006E;
        color: white;
        margin-bottom: 2.5rem;
        position: relative;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.25), inset 0 0 15px rgba(255, 0, 110, 0.15);
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: #00FFFF;
        box-shadow: 0 0 8px #00FFFF;
        animation: scan 4s linear infinite;
    }

    @keyframes scan {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }

    .hero h1 {
        font-size: 2.6rem;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        color: #00FFFF;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.6), 0 0 20px rgba(0, 255, 255, 0.3);
        display: flex;
        align-items: center;
    }

    .hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05rem;
        font-family: 'Poppins', sans-serif;
        color: #e2e8f0;
    }

    /* --- Section headings --- */
    .section-title {
        color: #00FFFF;
        font-family: 'Righteous', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.8rem 0 1.2rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- Retro Monospace Console Screen --- */
    .console-screen {
        background-color: #05050C !important;
        border: 1px solid #39FF14 !important;
        border-radius: 6px;
        padding: 1.2rem;
        font-family: 'Share Tech Mono', monospace !important;
        color: #39FF14 !important;
        text-shadow: 0 0 4px rgba(57, 255, 20, 0.5);
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.15) !important;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .console-title {
        font-weight: bold;
        border-bottom: 1px dashed #39FF14;
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .console-cursor {
        display: inline-block;
        width: 8px;
        height: 15px;
        background-color: #39FF14;
        margin-left: 4px;
        animation: cursor-blink 1s infinite;
        vertical-align: middle;
    }

    @keyframes cursor-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* --- Cards --- */
    .movie-card {
        background: rgba(16, 15, 30, 0.7);
        border: 1.5px solid #0080FF;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        cursor: pointer;
    }

    .movie-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 3px; height: 100%;
        background: #0080FF;
        transition: all 0.25s;
    }

    .movie-card:hover {
        background: rgba(22, 20, 42, 0.85);
        border-color: #FF006E;
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 0 20px rgba(255, 0, 110, 0.35), 0 5px 15px rgba(0, 0, 0, 0.4);
    }

    .movie-card:hover::before {
        background: #FF006E;
        box-shadow: 0 0 8px #FF006E;
        width: 5px;
    }

    .movie-title {
        color: #ffffff;
        font-family: 'Righteous', sans-serif !important;
        font-size: 1.15rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        line-height: 1.3;
        display: flex;
        align-items: center;
    }

    .movie-rank {
        display: inline-block;
        background: linear-gradient(135deg, #FF006E 0%, #9E0054 100%);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        text-align: center;
        line-height: 24px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.8rem;
        box-shadow: 0 0 5px rgba(255, 0, 110, 0.5);
        font-family: 'Righteous', sans-serif;
    }

    /* --- Genre pills --- */
    .genre-pill {
        display: inline-block;
        background: rgba(0, 255, 255, 0.08) !important;
        color: #00FFFF !important;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 4px 6px 4px 0;
        border: 1px solid rgba(0, 255, 255, 0.35) !important;
        font-family: 'Share Tech Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }

    .movie-card:hover .genre-pill {
        border-color: rgba(255, 0, 110, 0.5) !important;
        color: #FF85B3 !important;
        background: rgba(255, 0, 110, 0.08) !important;
    }

    /* --- Basket chips --- */
    .basket-chip {
        display: inline-block;
        background: rgba(255, 0, 110, 0.08) !important;
        color: #FF85B3 !important;
        padding: 6px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 4px 6px 4px 0;
        border: 1px dashed rgba(255, 0, 110, 0.6) !important;
        font-family: 'Share Tech Mono', monospace !important;
        transition: all 0.2s;
    }

    .basket-chip:hover {
        background: rgba(255, 0, 110, 0.15) !important;
        border-color: #FF006E !important;
        box-shadow: 0 0 8px rgba(255, 0, 110, 0.3);
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #070712 !important;
        border-right: 1px solid rgba(0, 128, 255, 0.2) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {color: #d0d0e0;}

    /* --- Inputs & Custom Overrides --- */
    div[data-baseweb="input"] {
        background-color: rgba(10, 10, 25, 0.6) !important;
        border: 1px solid rgba(0, 128, 255, 0.3) !important;
        border-radius: 6px !important;
        transition: all 0.2s;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #FF006E !important;
        box-shadow: 0 0 10px rgba(255, 0, 110, 0.2) !important;
    }

    input {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
    }

    .stMultiSelect > div > div {
        background-color: rgba(10, 10, 25, 0.6) !important;
        border-radius: 6px !important;
        border: 1px solid rgba(0, 128, 255, 0.3) !important;
    }

    /* Buttons styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0080FF 0%, #0059B3 100%) !important;
        color: #ffffff !important;
        border: 1px solid #00FFFF !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Righteous', sans-serif !important;
        text-transform: uppercase;
        font-size: 0.9rem !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
        width: 100%;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #FF006E 0%, #B3004D 100%) !important;
        border-color: #FF006E !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.4) !important;
    }

    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Primary Button Modifier (Get Recommendations) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF006E 0%, #9E0054 100%) !important;
        border-color: #FF006E !important;
        font-size: 1rem !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.3) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00FFCC 0%, #009977 100%) !important;
        border-color: #00FFCC !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6) !important;
    }

    /* Streamlit Radio & Slider labels */
    div[data-testid="stWidgetLabel"] p {
        color: #00FFFF !important;
        font-family: 'Righteous', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem !important;
        text-shadow: 0 0 4px rgba(0, 255, 255, 0.2);
    }

    /* Slider Track & Thumb */
    div[data-baseweb="slider"] {
        margin-bottom: 1.5rem;
    }

    /* Spinner */
    div[data-testid="stSpinner"] {
        border-color: #FF006E !important;
    }

    /* --- Empty state --- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #4b4b7a;
        background: rgba(10, 10, 25, 0.3);
        border: 1px dashed rgba(0, 128, 255, 0.2);
        border-radius: 8px;
    }

    /* --- Live suggestions --- */
    .suggestion-item {
        background: rgba(10, 10, 25, 0.6);
        border: 1px solid rgba(0, 128, 255, 0.25);
        border-left: 3px solid #0080FF;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.4rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .suggestion-item:hover {
        border-color: #FF006E;
        border-left-color: #FF006E;
        background: rgba(22, 20, 42, 0.85);
        transform: translateX(4px);
        box-shadow: 0 0 10px rgba(255, 0, 110, 0.2);
    }
    .suggestion-title-txt {
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .suggestion-empty {
        text-align: center;
        padding: 1.2rem;
        color: #4b4b7a;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        border: 1px dashed rgba(0, 128, 255, 0.15);
        border-radius: 6px;
    }

    /* --- How-to dialog --- */
    .help-step {
        display: flex;
        gap: 0.9rem;
        margin: 0.7rem 0;
        align-items: flex-start;
    }
    .help-step-num {
        background: linear-gradient(135deg, #FF006E 0%, #9E0054 100%);
        color: white;
        min-width: 28px;
        height: 28px;
        border-radius: 4px;
        text-align: center;
        line-height: 28px;
        font-family: 'Righteous', sans-serif;
        font-weight: 700;
        flex-shrink: 0;
        box-shadow: 0 0 6px rgba(255, 0, 110, 0.5);
    }
    .help-step-text {
        color: #cbd5e1;
        font-family: 'Poppins', sans-serif;
        line-height: 1.55;
        padding-top: 2px;
    }
    .help-step-key {
        color: #00FFFF;
        font-family: 'Share Tech Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        text-shadow: 0 0 4px rgba(0, 255, 255, 0.3);
    }
    .help-tip {
        color: #FF85B3;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        background: rgba(255, 0, 110, 0.06);
        border: 1px dashed rgba(255, 0, 110, 0.3);
        border-radius: 4px;
        padding: 0.6rem 0.9rem;
        margin-top: 1rem;
    }

    /* --- Taste profile bars --- */
    .taste-panel {
        background: rgba(10, 10, 25, 0.55);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .taste-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.55rem 0;
    }
    .taste-label {
        width: 110px;
        color: #00FFFF;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: right;
        flex-shrink: 0;
    }
    .taste-bar-container {
        flex: 1;
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 128, 255, 0.15);
        border-radius: 3px;
        height: 18px;
        position: relative;
        overflow: hidden;
    }
    .taste-bar {
        height: 100%;
        background: linear-gradient(90deg, #FF006E 0%, #9E0054 50%, #00FFCC 100%);
        box-shadow: 0 0 8px rgba(255, 0, 110, 0.35);
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .taste-pct {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        color: white;
        font-weight: 700;
        text-shadow: 0 0 4px rgba(0, 0, 0, 0.9);
    }
    .taste-summary {
        color: #cbd5e1;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        margin-top: 0.9rem;
        padding-top: 0.7rem;
        border-top: 1px dashed rgba(0, 255, 255, 0.15);
    }
    .taste-summary b {
        color: #FF85B3;
    }

    /* --- Genre match highlighting on recommendations --- */
    .genre-pill.genre-matched {
        background: rgba(255, 0, 110, 0.22) !important;
        color: #FF85B3 !important;
        border-color: rgba(255, 0, 110, 0.6) !important;
        box-shadow: 0 0 6px rgba(255, 0, 110, 0.3);
    }
    .match-badge {
        display: inline-block;
        margin-left: auto;
        background: linear-gradient(135deg, #00FFCC 0%, #00997A 100%);
        color: #001a12;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-family: 'Share Tech Mono', monospace;
        font-weight: 700;
        box-shadow: 0 0 6px rgba(0, 255, 204, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .match-badge.low {
        background: linear-gradient(135deg, #4a4a6a 0%, #2a2a4a 100%);
        color: #a5b4fc;
        box-shadow: none;
    }
    .movie-title {
        justify-content: space-between !important;
    }
    .movie-title-inner {
        display: flex;
        align-items: center;
    }

    /* --- Results view: prominent "back" button --- */
    .results-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    /* --- "Because you liked X" reason line --- */
    .rec-reason {
        color: #a5b4fc;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px dashed rgba(165, 180, 252, 0.2);
        line-height: 1.4;
    }
    .rec-reason b {
        color: #FF85B3;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _genre_color(genre: str) -> str:
    """Deterministic color per genre so the same tag always looks the same."""
    h = int(hashlib.md5(genre.encode()).hexdigest()[:6], 16)
    r = 80 + (h & 0xFF) % 100
    g = 80 + ((h >> 8) & 0xFF) % 100
    b = 120 + ((h >> 16) & 0xFF) % 100
    return f"rgba({r}, {g}, {b}, 0.25)"


def render_movie_card(rec: dict, rank: int, user_top_genres: set | None = None) -> None:
    """
    Render one recommendation card.

    If `user_top_genres` is passed, genres that overlap with the user's
    taste are highlighted in pink and a MATCH% badge is shown on the
    title bar. WHY: recommender output is often a black box - showing
    "these 3 of 5 genres match what you like" makes the model's choice
    legible without exposing raw ALS scores that mean nothing to the user.
    """
    genres = rec["genres"]
    user_set = user_top_genres or set()

    genre_html_parts = []
    matched = 0
    for g in genres:
        cls = "genre-pill"
        if g in user_set:
            cls += " genre-matched"
            matched += 1
        genre_html_parts.append(f'<span class="{cls}">{g}</span>')
    genre_html = "".join(genre_html_parts)

    if user_set and genres:
        match_pct = matched / len(genres) * 100
        badge_cls = "match-badge" if match_pct >= 40 else "match-badge low"
        badge_html = f'<span class="{badge_cls}">{match_pct:.0f}% MATCH</span>'
    else:
        badge_html = ""

    # "Because you liked X" reason line, when the API attached one.
    because = rec.get("because_of")
    reason_html = (
        f'<div class="rec-reason">&raquo; Because you liked <b>{because}</b></div>'
        if because and because != rec["title"]
        else ""
    )

    html = (
        f'<div class="movie-card">'
        f'<div class="movie-title">'
        f'<span class="movie-title-inner"><span class="movie-rank">{rank}</span>{rec["title"]}</span>'
        f'{badge_html}'
        f'</div>'
        f'<div>{genre_html}</div>'
        f'{reason_html}'
        f'</div>'
    )
    st.html(html)


def render_suggestion(opt: dict) -> None:
    """Card for the live-search dropdown. Kept HTML-only so the row
    stays visually unified with the '+' button rendered by Streamlit
    in the adjacent column."""
    genre_html = "".join(
        f'<span class="genre-pill" style="background:{_genre_color(g)}">{g}</span>'
        for g in opt["genres"]
    )
    st.markdown(
        f"""
        <div class="suggestion-item">
            <span class="suggestion-title-txt">{opt['title']}</span>
            <div>{genre_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_taste_profile(basket: list) -> None:
    """
    Aggregate genres across the basket into a normalised bar chart.

    WHY show this: giving the user a live view of the profile they are
    building serves two purposes:
      1. UX - they see how their choices are being interpreted before
         the model runs, so they can steer their picks.
      2. Storytelling for interviews - "the frontend visualises the
         user embedding in a human-readable way" is a strong line.
    """
    if not basket:
        return

    counts: Counter = Counter()
    for item in basket:
        for g in item.get("genres", []):
            if g and g != "(no genres listed)":
                counts[g] += 1
    if not counts:
        return

    n_movies = len(basket)
    top = counts.most_common(6)
    top_genre = top[0][0]
    diversity = len(counts)

    # Bar width = fraction of the user's movies that touch this genre.
    # 100% = every picked movie has it (strong signal). 25% = weak.
    # WHY not "share of total genre-hits": with 5 picks × 5 genres each,
    # every genre lands somewhere and shares squash toward uniformity.
    rows_html = ""
    for genre, count in top:
        pct = min(100.0, count / n_movies * 100)
        rows_html += (
            f'<div class="taste-row">'
            f'<div class="taste-label">{genre}</div>'
            f'<div class="taste-bar-container">'
            f'<div class="taste-bar" style="width: {pct:.1f}%;"></div>'
            f'<span class="taste-pct">{count}/{n_movies}</span>'
            f'</div>'
            f'</div>'
        )

    html = (
        f'<div class="taste-panel">'
        f'{rows_html}'
        f'<div class="taste-summary">'
        f'PROFILE &raquo; <b>{top_genre}</b>-leaning &nbsp;&middot;&nbsp; '
        f'{n_movies} movie{"s" if n_movies != 1 else ""} &nbsp;&middot;&nbsp; '
        f'{diversity} distinct genres'
        f'</div>'
        f'</div>'
    )
    # st.html bypasses markdown parsing entirely, avoiding blockquote /
    # code-block confusion when the HTML contains characters like `>`
    # or has multi-line indentation.
    st.html(html)


# ---------------------------------------------------------------------------
# How-to-use dialog
# WHY st.dialog: modal overlay is the right pattern for onboarding - it
# blocks the UI so the user reads the flow before doing anything, then
# it disappears. Requires Streamlit >= 1.34.
# ---------------------------------------------------------------------------
@st.dialog("SYSTEM // USER MANUAL", width="large")
def show_how_to() -> None:
    st.markdown(
        """
        <div class="console-screen">
            <div class="console-title">> INITIALISING TUTORIAL SEQUENCE</div>
            Follow the steps below to get personalised movie recommendations
            powered by ALS collaborative filtering.
            <span class="console-cursor"></span>
        </div>

        <div class="help-step">
            <div class="help-step-num">1</div>
            <div class="help-step-text">
                <span class="help-step-key">SEARCH</span> — Start typing a movie name
                in the search box. Live suggestions appear below as you type
                (min 2 characters).
            </div>
        </div>
        <div class="help-step">
            <div class="help-step-num">2</div>
            <div class="help-step-text">
                <span class="help-step-key">ADD</span> — Click the
                <b style="color:#00FFCC;">+</b> button next to any suggestion
                to add it to your list.
            </div>
        </div>
        <div class="help-step">
            <div class="help-step-num">3</div>
            <div class="help-step-text">
                <span class="help-step-key">REPEAT</span> — Add 3-5 movies you have
                genuinely enjoyed. More diverse choices = more accurate recommendations.
            </div>
        </div>
        <div class="help-step">
            <div class="help-step-num">4</div>
            <div class="help-step-text">
                <span class="help-step-key">EXECUTE</span> — Hit
                <b style="color:#00FFCC;">GET RECOMMENDATIONS</b>. The ALS engine
                scores 40K+ movies against your taste vector in under 10ms.
            </div>
        </div>
        <div class="help-step">
            <div class="help-step-num">5</div>
            <div class="help-step-text">
                <span class="help-step-key">EXPLORE</span> — Hover over any card
                to see it light up. Adjust the recommendation count from the
                sidebar. Switch to <b>userId</b> mode to test the model on a
                real MovieLens user.
            </div>
        </div>

        <div class="help-tip">
            > TIP: The model was trained on ratings up to 2018.
            Newer movies won't appear in results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("LET'S GO", type="primary", use_container_width=True, key="help_close"):
        st.rerun()


# Initialise session state.
# basket:      list of {"title": str, "genres": [str]} for taste profile
# recs:        None | dict returned by /recommend (marks results view)
# help_shown:  whether the how-to dialog has been auto-opened this session
if "help_shown" not in st.session_state:
    st.session_state["help_shown"] = False
if "basket" not in st.session_state:
    st.session_state["basket"] = []
if "recs" not in st.session_state:
    st.session_state["recs"] = None


def _basket_titles() -> list:
    return [item["title"] for item in st.session_state["basket"]]


def _user_top_genres(basket: list, min_share: float = 0.25) -> set:
    """
    Genres that appear in at least `min_share` of the basket.
    Used to highlight matching genres and compute MATCH % on
    recommendation cards.
    """
    if not basket:
        return set()
    counts: Counter = Counter()
    for item in basket:
        for g in item.get("genres", []):
            if g and g != "(no genres listed)":
                counts[g] += 1
    threshold = max(1, int(len(basket) * min_share))
    return {g for g, c in counts.items() if c >= threshold}


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 12px; color: #00FFFF;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M17 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/></svg>
            MOVIELENS RECOMMENDER
        </h1>
        <p>ALS collaborative filtering trained on 25M ratings. Pick a few movies you love and get personalised recommendations in under 10ms.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Trigger the how-to dialog on first load. Session-state flag prevents
# it from popping up every rerun.
if not st.session_state["help_shown"]:
    st.session_state["help_shown"] = True
    show_how_to()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFFF;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
            <span style="font-family: 'Righteous', sans-serif; color: #00FFFF; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Settings</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    n_recs = st.slider("Recommendations", min_value=5, max_value=30, value=10)
    mode = st.radio(
        "Input mode",
        ["Pick movies you like", "Use a MovieLens userId"],
        label_visibility="visible",
    )
    if st.button("HOW TO USE", use_container_width=True, key="help_sidebar"):
        show_how_to()
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFFF;"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>
            <span style="font-family: 'Righteous', sans-serif; color: #00FFFF; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">System</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        api_status = f"ONLINE\n  USERS  : {h['n_users']:,}\n  ITEMS  : {h['n_items']:,}\n  FACTORS: {h['model_factors']}"
    except Exception:
        api_status = "OFFLINE"

    st.markdown(
        f"""
        <div class="console-screen">
            <div class="console-title">> CORE INTERFACE</div>
            MODEL  : ALS Filter<br>
            FACTORS: 64<br>
            NDCG@10: 0.151<br>
            API    : {api_status}
            <span class="console-cursor"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if mode == "Pick movies you like" and st.session_state["recs"] is None:
    st.markdown(
        """
        <div class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFFF;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            Search Titles
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ---- Live search-as-you-type ----------------------------------------
    # st_keyup fires a Streamlit rerun on every key release (debounced
    # by 250ms) so suggestions update without the user hitting Enter.
    # Falls back to text_input if the optional package is not installed.
    if _KEYUP_AVAILABLE:
        query = st_keyup(
            "Search",
            placeholder="Try: matrix, godfather, inception, dark knight...",
            debounce=250,
            key="search_query",
            label_visibility="collapsed",
        )
    else:
        query = st.text_input(
            "Search",
            placeholder="Try: matrix, godfather, inception, dark knight...",
            label_visibility="collapsed",
            key="search_query",
        )
        st.caption("Install `streamlit-keyup` for live search: `pip install streamlit-keyup`")

    # Each keystroke triggers a full Streamlit rerun, which re-fetches
    # from the API. That is fine because /movies/search is a linear scan
    # in memory (~60K titles) and returns in < 5 ms.
    options = []
    if query and len(query) >= 2:
        try:
            r = requests.get(
                f"{API_URL}/movies/search",
                params={"q": query, "limit": 15},
                timeout=10,
            )
            r.raise_for_status()
            options = r.json()
        except requests.RequestException as e:
            st.error(f"Search failed: {e}")

    if query and len(query) >= 2:
        basket_titles = set(_basket_titles())
        fresh = [o for o in options if o["title"] not in basket_titles]
        if fresh:
            for opt in fresh:
                sug_col, btn_col = st.columns([9, 1])
                with sug_col:
                    render_suggestion(opt)
                with btn_col:
                    if st.button("+", key=f"add_{opt['movie_id']}", help="Add to your list"):
                        st.session_state["basket"].append(
                            {"title": opt["title"], "genres": opt.get("genres", [])}
                        )
                        st.rerun()
        else:
            st.markdown(
                '<div class="suggestion-empty">> NO NEW MATCHES · try another query</div>',
                unsafe_allow_html=True,
            )

    # ---- Clear all -------------------------------------------------------
    if st.session_state["basket"]:
        if st.button("CLEAR ALL", use_container_width=False, key="clear_basket"):
            st.session_state["basket"] = []
            st.rerun()

    if st.session_state["basket"]:
        st.markdown(
            """
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #FF006E;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                Your Liked Movies
            </div>
            """,
            unsafe_allow_html=True,
        )
        chips = "".join(
            f'<span class="basket-chip">{item["title"]}</span>'
            for item in st.session_state["basket"]
        )
        st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ---- CTA + Clear side-by-side ------------------------------------
        # Taste profile is intentionally NOT shown in the pick view -
        # it appears only after recommendations are generated, so the
        # reveal has more impact.
        cta_col, clear_col = st.columns([4, 1])
        with cta_col:
            if st.button(
                f"FIND MY RECOMMENDATIONS ({len(st.session_state['basket'])} MOVIES)",
                type="primary",
                use_container_width=True,
                key="cta_recommend",
            ):
                with st.spinner("Scoring 40K+ movies against your taste..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/recommend",
                            json={"liked_titles": _basket_titles(), "n": n_recs},
                            timeout=15,
                        )
                        r.raise_for_status()
                        st.session_state["recs"] = r.json()
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Recommendation failed: {e}")
        with clear_col:
            if st.button("CLEAR", use_container_width=True, key="clear_all"):
                st.session_state["basket"] = []
                st.rerun()
    else:
        st.markdown(
            """
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color: #4b4b7a; margin-bottom: 1rem;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M17 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/></svg>
                <br>
                Search for movies above and add a few you love to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )

elif mode == "Pick movies you like" and st.session_state["recs"] is not None:
    # ============================================================
    # RESULTS VIEW - shown after Recommend was pressed.
    # Hides the search / suggestions. Shows: taste profile + recs
    # with per-card genre-match highlighting.
    # ============================================================
    payload = st.session_state["recs"]
    user_top = _user_top_genres(st.session_state["basket"])

    # Back button
    if st.button("&laquo; PICK MORE MOVIES / REFINE", key="back_to_pick", use_container_width=False):
        st.session_state["recs"] = None
        st.rerun()

    # Taste profile
    st.markdown(
        """
        <div class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFFF;"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
            Your Taste Profile
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_taste_profile(st.session_state["basket"])

    # Recommendations with match highlighting
    st.markdown(
        f"""
        <div class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFCC;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
            Top {len(payload["recommendations"])} for you
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Pink pills = genre you like · MATCH % = fraction of the movie's genres that overlap your taste")

    left, right = st.columns(2)
    for i, rec in enumerate(payload["recommendations"], 1):
        with (left if i % 2 else right):
            render_movie_card(rec, i, user_top_genres=user_top)

else:
    st.markdown(
        """
        <div class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFFF;"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            MovieLens userId
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_id = st.number_input("userId", min_value=1, step=1, value=1, label_visibility="collapsed")

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Computing recommendations..."):
            try:
                r = requests.post(
                    f"{API_URL}/recommend",
                    json={"user_id": int(user_id), "n": n_recs},
                    timeout=15,
                )
                r.raise_for_status()
                payload = r.json()
            except requests.RequestException as e:
                st.error(f"Recommendation failed: {e}")
                payload = None

        if payload:
            st.markdown(
                f"""
                <div class="section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #00FFCC;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                    Top {len(payload["recommendations"])} for user {user_id}
                </div>
                """,
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            for i, rec in enumerate(payload["recommendations"], 1):
                with (left if i % 2 else right):
                    render_movie_card(rec, i)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#5a5a7a; font-family: "Share Tech Mono", monospace; font-size:0.8rem; padding:2rem 1rem;'>
        Built with FastAPI · implicit ALS · Streamlit &nbsp;·&nbsp;
        Data by <a href='https://grouplens.org/datasets/movielens/25m/' style='color:#FF006E; text-decoration: none; text-shadow: 0 0 4px rgba(255, 0, 110, 0.4);'>GroupLens</a>
    </div>
    """,
    unsafe_allow_html=True,
)
