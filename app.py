import os
import re
import json
import html
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="WorldCup 2026 Observatory", layout="wide")

DATA_PATH = "data/location_all_df.csv"
CITY_COORDS_PATH = "data/city_coords.json"
MAP_PANEL_HEIGHT = 650
CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
WORLD_GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

HOST_CITIES_2026 = {
    "toronto", "vancouver", "mexico city", "guadalajara", "monterrey",
    "atlanta", "boston", "dallas", "houston", "kansas city", "los angeles",
    "miami", "new york", "new york city", "new jersey", "philadelphia",
    "san francisco", "seattle",
}

HOST_COUNTRIES_2026 = {
    "United States", "Canada", "Mexico",
}

HOST_CITY_ALIASES = {
    "arlington": "dallas",
    "inglewood": "los angeles",
    "foxborough": "boston",
    "miami gardens": "miami",
    "kansas city, missouri": "kansas city",
    "east rutherford": "new jersey",
    "bay area": "san francisco",
}

st.markdown("""
<style>
.side-panel-shell {
    height: 650px;
    border: 1px solid #e8e8e8;
    border-radius: 14px;
    padding: 14px;
    background: white;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.side-panel-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111;
    margin-bottom: 12px;
    flex-shrink: 0;
}
.side-panel-top { flex-shrink: 0; }
.side-panel-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding-right: 6px;
    margin-top: 10px;
}
.side-meta-card {
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 8px 10px;
    background: #fafafa;
    margin-bottom: 8px;
}
.side-meta-row { margin-bottom: 8px; }
.side-meta-row:last-child { margin-bottom: 0; }
.side-meta-label {
    font-size: 0.72rem;
    color: #666;
    font-weight: 600;
    margin-bottom: 1px;
    line-height: 1.1;
}
.side-meta-value {
    font-size: 0.9rem;
    color: #111;
    font-weight: 600;
    line-height: 1.2;
    word-break: break-word;
}
.side-scrollbox { max-height: none; overflow: visible; padding-right: 0; }
.side-article-card {
    border: 1px solid #ececec;
    border-radius: 12px;
    padding: 12px;
    background: white;
    margin-bottom: 10px;
}
.side-article-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #111;
    margin-bottom: 8px;
    line-height: 1.35;
}
.side-article-meta { font-size: 0.84rem; color: #666; margin-bottom: 4px; }
.side-article-link a { font-size: 0.9rem; font-weight: 600; text-decoration: none; }
div.stButton > button {
    font-size: 0.82rem;
    padding: 0.20rem 0.70rem;
    min-height: 32px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Basic helpers
# ----------------------------
def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def pick_datetime_source(df: pd.DataFrame):
    if "published" in df.columns:
        return "published"
    if "fetched_at" in df.columns:
        return "fetched_at"
    return None


@st.cache_data(show_spinner="Loading data…")
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    dt_col = pick_datetime_source(df)
    if dt_col is None:
        df["published_dt"] = pd.NaT
    else:
        df["published_dt"] = safe_to_datetime(df[dt_col])
    for c in ["topic", "content", "title", "snippet", "city", "country", "real_link", "link"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


@st.cache_data(show_spinner=False)
def load_city_coords() -> dict:
    with open(CITY_COORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def split_places(x) -> list:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        p = p.strip(" .;:-_()[]{}'\"")
        if len(p) >= 2:
            out.append(p)
    return out


def normalize_country(name: str) -> str:
    n = (name or "").strip()
    lower = n.lower()
    mapping = {
        "u.s.": "United States",
        "u.s": "United States",
        "us": "United States",
        "usa": "United States",
        "the united states": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "russian federation": "Russia",
        "viet nam": "Vietnam",
        "iran, islamic republic of": "Iran",
        "korea, republic of": "South Korea",
        "republic of korea": "South Korea",
        "côte d'ivoire": "Cote d'Ivoire",
    }
    return mapping.get(lower, n)


def normalize_city_name(name: str) -> str:
    s = "" if name is None else str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return HOST_CITY_ALIASES.get(s, s)


def is_host_city_2026(name: str) -> bool:
    return normalize_city_name(name) in HOST_CITIES_2026


def is_host_country_2026(name: str) -> bool:
    return normalize_country(name) in HOST_COUNTRIES_2026


def clean_text_for_wordcloud(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def same_city(candidate: str, selected: str) -> bool:
    return normalize_city_name(candidate) == normalize_city_name(selected)


def strip_html_tags(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Place stats
# ----------------------------
@st.cache_data(show_spinner=False)
def build_place_stats(df_articles: pd.DataFrame) -> dict:
    stats = {}

    def add(place_type, place, topic):
        key = f"{place_type}:{place}"
        if key not in stats:
            stats[key] = {"count": 0, "topic_counts": {}}
        stats[key]["count"] += 1
        if topic:
            stats[key]["topic_counts"][topic] = stats[key]["topic_counts"].get(topic, 0) + 1

    for _, r in df_articles.iterrows():
        topic = str(r.get("topic", "") or "").strip()
        cities = split_places(r.get("city", None))
        countries = [normalize_country(x) for x in split_places(r.get("country", None))]
        for c in cities:
            add("city", c, topic)
        for c in countries:
            add("country", c, topic)

    final = {}
    for key, d in stats.items():
        topic_sorted = sorted(d["topic_counts"].items(), key=lambda x: x[1], reverse=True)
        most_talked_topics = (
            ", ".join([f"{topic} ({count})" for topic, count in topic_sorted])
            if topic_sorted else "—"
        )
        final[key] = {
            "count": int(d["count"]),
            "most_talked_topics": most_talked_topics,
        }
    return final


# ----------------------------
# City coords — reads from city_coords.json, NO Nominatim/geopy at all
# ----------------------------
def explode_city_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for c in split_places(r.get("city", None)):
            rows.append({"place": c})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = (
        out.groupby("place", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    return out


@st.cache_data(show_spinner=False)
def geocode_places_cached(city_df: pd.DataFrame, max_places: int = 150) -> pd.DataFrame:
    if city_df.empty:
        return city_df.assign(lat=None, lon=None)
    coords = load_city_coords()
    target = city_df.head(max_places).copy()
    target["lat"] = target["place"].map(lambda x: coords.get(x, {}).get("lat"))
    target["lon"] = target["place"].map(lambda x: coords.get(x, {}).get("lon"))
    return target.dropna(subset=["lat", "lon"]).copy()


# ----------------------------
# Country GeoJSON
# ----------------------------
@st.cache_data(show_spinner=False)
def load_world_geojson() -> dict:
    r = requests.get(WORLD_GEOJSON_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def get_feature_country_name(feat: dict) -> str:
    props = feat.get("properties", {}) or {}
    return str(props.get("ADMIN") or props.get("name") or props.get("NAME") or "")


def color_scale(count: int, max_count: int) -> list:
    if count <= 0 or max_count <= 0:
        return [220, 220, 220, 90]
    x = count / max_count
    if x < 0.15:
        return [255, 235, 140, 170]
    if x < 0.35:
        return [255, 200, 90, 180]
    if x < 0.65:
        return [255, 130, 60, 190]
    return [220, 40, 40, 200]


@st.cache_data(show_spinner=False)
def build_country_geojson(df_f: pd.DataFrame, place_stats_json: str) -> dict:
    place_stats = json.loads(place_stats_json)
    world = load_world_geojson()

    mentions = []
    for _, r in df_f.iterrows():
        for c in [normalize_country(x) for x in split_places(r.get("country", None))]:
            mentions.append(c)

    counts = pd.Series(mentions).value_counts().to_dict() if mentions else {}
    max_count = max(counts.values()) if counts else 0

    for feat in world.get("features", []):
        raw_name = get_feature_country_name(feat)
        name = normalize_country(raw_name)
        key = f"country:{name}"
        s = place_stats.get(key, None)
        cnt = int(counts.get(name, 0))
        most_talked_topics = s["most_talked_topics"] if s else "—"
        host_label = "Official 2026 World Cup host country" if is_host_country_2026(name) else "Non-host country"

        feat.setdefault("properties", {})
        feat["properties"]["fill_color"] = color_scale(cnt, max_count)
        feat["properties"]["kind"] = "Country"
        feat["properties"]["label"] = name
        feat["properties"]["count"] = cnt
        feat["properties"]["host_label"] = host_label
        feat["properties"]["most_talked_topics"] = most_talked_topics
        feat["kind"] = "Country"
        feat["label"] = name
        feat["count"] = cnt
        feat["host_label"] = host_label
        feat["most_talked_topics"] = most_talked_topics

    return world


# ----------------------------
# Article filters
# ----------------------------
def filter_articles_for_country(df_articles: pd.DataFrame, country_name: str) -> pd.DataFrame:
    if not country_name:
        return df_articles.iloc[0:0].copy()
    mask = df_articles["country"].fillna("").apply(
        lambda cell: any(
            normalize_country(c) == normalize_country(country_name)
            for c in split_places(cell)
        )
    )
    out = df_articles[mask].copy()
    if "published_dt" in out.columns:
        out = out.sort_values("published_dt", ascending=False, na_position="last")
    return out


def filter_articles_for_city(df_articles: pd.DataFrame, city_name: str) -> pd.DataFrame:
    if not city_name:
        return df_articles.iloc[0:0].copy()
    mask = df_articles["city"].fillna("").apply(
        lambda cell: any(same_city(c, city_name) for c in split_places(cell))
    )
    out = df_articles[mask].copy()
    if "published_dt" in out.columns:
        out = out.sort_values("published_dt", ascending=False, na_position="last")
    return out


def get_country_for_selected_place(df_articles: pd.DataFrame, selected_place_type: str, selected_place_name: str) -> str:
    if not selected_place_type or not selected_place_name:
        return "—"
    if selected_place_type == "country":
        return selected_place_name
    if selected_place_type == "city":
        city_articles = filter_articles_for_city(df_articles, selected_place_name)
        countries = []
        for cell in city_articles["country"].fillna(""):
            for c in split_places(cell):
                c_norm = normalize_country(c)
                if c_norm and c_norm not in countries:
                    countries.append(c_norm)
        return ", ".join(countries) if countries else "—"
    return "—"


def build_articles_html(df_articles: pd.DataFrame) -> str:
    if df_articles.empty:
        return """
        <div class="side-scrollbox">
            <div class="side-article-card">
                <div class="side-article-title">No articles found in the current filters.</div>
            </div>
        </div>
        """
    cards = []
    for i, (_, row) in enumerate(df_articles.iterrows(), start=1):
        title = html.escape(strip_html_tags(row.get("title", "Untitled")))
        topic = html.escape(strip_html_tags(row.get("topic", "")))
        published = html.escape(str(row.get("published", "") or ""))
        url = row.get("real_link", None)

        parts = [f'<div class="side-article-title">{i}. {title}</div>']
        if topic:
            parts.append(f'<div class="side-article-meta"><b>Topic:</b> {topic}</div>')
        if published.strip():
            parts.append(f'<div class="side-article-meta"><b>Published:</b> {published}</div>')
        if pd.notna(url) and str(url).strip():
            safe_url = html.escape(str(url))
            parts.append(f'<div class="side-article-link"><a href="{safe_url}" target="_blank">Open article</a></div>')

        cards.append(f'<div class="side-article-card">{"".join(parts)}</div>')
    return f'<div class="side-scrollbox">{"".join(cards)}</div>'


# ----------------------------
# Selection helpers
# ----------------------------
def ensure_selection_state():
    if "selected_place_type" not in st.session_state:
        st.session_state.selected_place_type = None
    if "selected_place_name" not in st.session_state:
        st.session_state.selected_place_name = None


def clear_selection():
    st.session_state.selected_place_type = None
    st.session_state.selected_place_name = None


def get_event_objects(event):
    if event is None:
        return {}
    if isinstance(event, dict):
        return event.get("selection", {}).get("objects", {})
    try:
        return event.selection.get("objects", {})
    except Exception:
        pass
    try:
        return event.selection.objects
    except Exception:
        pass
    return {}


def get_clicked_place(event):
    objects = get_event_objects(event)
    city_items = objects.get("cities", [])
    if city_items:
        obj = city_items[0]
        place_name = obj.get("label") or obj.get("place")
        if place_name:
            return "city", place_name
    country_items = objects.get("countries", [])
    if country_items:
        obj = country_items[0]
        place_name = (
            obj.get("label")
            or obj.get("properties", {}).get("label")
            or obj.get("NAME")
            or obj.get("ADMIN")
        )
        if place_name:
            return "country", place_name
    return None, None


# ----------------------------
# UI
# ----------------------------
ensure_selection_state()

title_col, spacer_col, btn1_col, btn2_col = st.columns([7.2, 2.2, 1.2, 1.4])

with title_col:
    st.title("WorldCup 2026 Observatory")

with btn1_col:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

with btn2_col:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("Clear selected"):
        clear_selection()
        st.rerun()

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Cannot find {DATA_PATH}. Update DATA_PATH at top.")
    st.stop()

if not os.path.exists(CITY_COORDS_PATH):
    st.error(f"Cannot find {CITY_COORDS_PATH}. Make sure city_coords.json is committed to your repo.")
    st.stop()

df = load_data(DATA_PATH)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Filters")

topics = sorted([t for t in df["topic"].dropna().unique().tolist()])
topic_choice = st.sidebar.selectbox("Select topic", ["All"] + topics, index=0)

min_dt = df["published_dt"].dropna().min()
max_dt = df["published_dt"].dropna().max()

if pd.isna(min_dt) or pd.isna(max_dt):
    st.sidebar.warning("No valid timestamps found. Date slider disabled.")
    date_range = None
else:
    date_range = st.sidebar.slider(
        "Published date range",
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        value=(min_dt.date(), max_dt.date()),
    )

map_mode = st.sidebar.selectbox("Map mode", ["Country (filled)", "City bubbles", "Both"], index=2)
topN_city = st.sidebar.slider("Max cities to show (top by frequency)", 25, 300, 150, step=25)
months_back = st.sidebar.slider("Topic popularity window (last N months)", 1, 36, 6, step=1)
top_k_topics = st.sidebar.slider("Top topics to show", 5, 30, 12, step=1)
max_rows_wordcloud = st.sidebar.slider("Max rows for wordcloud (speed)", 50, 5000, 2000, step=50)

# ----------------------------
# Filter data
# ----------------------------
df_f = df.copy()

if topic_choice != "All":
    df_f = df_f[df_f["topic"] == topic_choice]

if date_range is not None:
    start_d, end_d = date_range
    df_f = df_f[
        (df_f["published_dt"].dt.date >= start_d) &
        (df_f["published_dt"].dt.date <= end_d)
    ]

place_stats = build_place_stats(df_f)

# ----------------------------
# Map
# ----------------------------
tooltip_html = """
<div style="
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  width: 320px;
  line-height: 1.5;
">
  <div style="font-size: 20px; font-weight: 800; margin-bottom: 8px;">{label}</div>
  <div style="font-size: 14px; color: #444; margin-bottom: 4px;">{kind}</div>
  <div style="font-size: 14px; color: #9b1c1c; font-weight: 700; margin-bottom: 8px;">{host_label}</div>
  <div style="font-size: 14px; margin-bottom: 6px;"><b>Total articles talking about it:</b> {count}</div>
  <div style="font-size: 14px; color: #222;"><b>Most talked topics:</b> {most_talked_topics}</div>
</div>
"""

layers = []
geo_df = pd.DataFrame()

if map_mode in ["Country (filled)", "Both"]:
    world_geo = build_country_geojson(df_f, json.dumps(place_stats))
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            id="countries",
            data=world_geo,
            opacity=0.95,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[120, 120, 120, 140],
            line_width_min_pixels=0.6,
            pickable=True,
        )
    )

if map_mode in ["City bubbles", "Both"]:
    city_counts = explode_city_counts(df_f)
    if not city_counts.empty:
        geo_df = geocode_places_cached(city_counts, topN_city)

        if not geo_df.empty:
            def stat(place: str, field: str):
                s = place_stats.get(f"city:{place}")
                if not s:
                    return "—" if field == "most_talked_topics" else 0
                return s[field]

            geo_df["most_talked_topics"] = geo_df["place"].apply(lambda x: stat(x, "most_talked_topics"))
            geo_df["is_host_city"] = geo_df["place"].apply(is_host_city_2026)
            geo_df["host_label"] = geo_df["is_host_city"].apply(
                lambda x: "Official 2026 World Cup host city" if x else "Non-host city"
            )
            geo_df["kind"] = "City"
            geo_df["label"] = geo_df["place"]
            geo_df["radius"] = np.where(
                geo_df["is_host_city"],
                np.clip(np.sqrt(geo_df["count"].astype(float)) * 5000, 12000, 50000),
                np.clip(np.sqrt(geo_df["count"].astype(float)) * 3500, 8000, 35000),
            )
            geo_df["fill_color"] = geo_df["is_host_city"].apply(
                lambda x: [220, 50, 50, 210] if x else [30, 110, 190, 170]
            )
            geo_df["line_color"] = geo_df["is_host_city"].apply(
                lambda x: [120, 0, 0, 255] if x else [10, 50, 120, 220]
            )

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    id="cities",
                    data=geo_df,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="fill_color",
                    get_line_color="line_color",
                    stroked=True,
                    filled=True,
                    line_width_min_pixels=1.5,
                    pickable=True,
                    opacity=0.9,
                    radius_min_pixels=10,
                    radius_max_pixels=60,
                )
            )

view = pdk.ViewState(latitude=20, longitude=0, zoom=1.35, pitch=0)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style=CARTO_POSITRON,
    tooltip={
        "html": tooltip_html,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "maxWidth": "340px",
            "padding": "12px",
            "border": "1px solid rgba(0,0,0,0.12)",
            "borderRadius": "10px",
            "boxShadow": "0 8px 24px rgba(0,0,0,0.15)",
        },
    },
)

col_map, col_side = st.columns([2.1, 1], gap="large")

with col_map:
    event = st.pydeck_chart(
        deck,
        height=MAP_PANEL_HEIGHT,
        on_select="ignore",
        selection_mode="single-object",
        key="main_map",
    )

    clicked_type, clicked_name = get_clicked_place(event)
    if clicked_type and clicked_name:
        st.session_state.selected_place_type = clicked_type
        st.session_state.selected_place_name = clicked_name

with col_side:
    with st.container(height=MAP_PANEL_HEIGHT, border=True):
        st.markdown("Selected location")

        selected_place_type = st.session_state.selected_place_type
        selected_place_name = st.session_state.selected_place_name

        if not selected_place_type or not selected_place_name:
            st.info("Click a city or country on the map to show its articles here.")
        else:
            country_text = get_country_for_selected_place(df_f, selected_place_type, selected_place_name)
            display_city = selected_place_name if selected_place_type == "city" else "—"

            st.markdown(
                f"""
                <div class="side-meta-card">
                    <div class="side-meta-value">
                        <b>Type:</b> {html.escape(selected_place_type.title())},
                        <b>City:</b> {html.escape(display_city)},
                        <b>Country:</b> {html.escape(country_text)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if selected_place_type == "city":
                city_articles = filter_articles_for_city(df_f, selected_place_name)
                st.markdown(build_articles_html(city_articles), unsafe_allow_html=True)
            elif selected_place_type == "country":
                country_articles = filter_articles_for_country(df_f, selected_place_name)
                st.markdown(build_articles_html(country_articles), unsafe_allow_html=True)


# ----------------------------
# Topic popularity + WordCloud
# ----------------------------
st.divider()

col_pop, col_wc = st.columns([1, 1], gap="large")

df_pop = df_f.copy()

if date_range is not None:
    _, end_d = date_range
    window_end = pd.Timestamp(end_d)
else:
    window_end = df_pop["published_dt"].dropna().max()

with col_pop:
    st.subheader("Topic popularity")

    if pd.isna(window_end):
        st.info("No timestamps to compute popularity.")
        topic_counts = pd.DataFrame(columns=["topic", "count"])
        ts_counts = pd.DataFrame(columns=["month", "topic", "count"])
    else:
        window_start = (window_end - pd.DateOffset(months=months_back)).normalize()
        df_pop = df_pop[
            (df_pop["published_dt"].notna()) &
            (df_pop["published_dt"] >= window_start) &
            (df_pop["published_dt"] <= (window_end + pd.Timedelta(days=1)))
        ].copy()

        topic_counts = (
            df_pop["topic"].dropna().value_counts().head(top_k_topics)
            .rename_axis("topic").reset_index(name="count")
        )

        fig_bar = px.bar(
            topic_counts.sort_values("count", ascending=True),
            x="count", y="topic", orientation="h",
            labels={"count": "Articles", "topic": "Topic"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        df_ts = df_pop[["published_dt", "topic"]].dropna().copy()
        df_ts["month"] = df_ts["published_dt"].dt.to_period("M").dt.to_timestamp()

        if topic_choice != "All":
            df_ts = df_ts[df_ts["topic"] == topic_choice]
        else:
            top_topics = topic_counts["topic"].tolist()
            df_ts = df_ts[df_ts["topic"].isin(top_topics)]

        ts_counts = (
            df_ts.groupby(["month", "topic"])
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )

with col_wc:
    st.subheader("WordCloud")
    wc_df = df_f.head(max_rows_wordcloud)
    text = " ".join([clean_text_for_wordcloud(x) for x in wc_df["content"].dropna().tolist()])

    if len(text.strip()) < 50:
        st.info("Not enough content for a wordcloud.")
    else:
        wc = WordCloud(
            width=900, height=450,
            background_color="white",
            stopwords=set(STOPWORDS),
            collocations=False,
        ).generate(text)

        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

st.subheader("Popularity over time")

if "ts_counts" not in locals() or ts_counts.empty:
    st.info("Not enough data to plot popularity over time.")
else:
    fig_ts = px.line(
        ts_counts, x="month", y="count", color="topic", markers=True,
        labels={"month": "Month", "count": "Articles", "topic": "Topic"},
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------------
# Current view table
# ----------------------------
st.divider()
st.subheader("Articles in current view")

show_cols = [c for c in ["published", "topic", "title", "real_link", "link", "country", "city"] if c in df_f.columns]
st.dataframe(df_f[show_cols], use_container_width=True, hide_index=True)