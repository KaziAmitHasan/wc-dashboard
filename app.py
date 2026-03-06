import os
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

# Updated to use your new dataset file name
DATA_PATH = "data/location_all_df.csv"
GEOCACHE_PATH = "geocode_cache.json"

CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
WORLD_GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"


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

def load_data_no_cache(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    dt_col = pick_datetime_source(df)
    
    if dt_col is None:
        df["published_dt"] = pd.NaT
    else:
        df["published_dt"] = safe_to_datetime(df[dt_col])

    # Updated extracted_cities -> city, extracted_countries -> country
    for c in ["topic", "content", "title", "snippet", "city", "country", "real_link", "link"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df

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
        "côte d’ivoire": "Cote d'Ivoire",
        "côte d'ivoire": "Cote d'Ivoire",
    }
    return mapping.get(lower, n)

def load_geocache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_geocache(cache: dict, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def truncate(s: str, n: int = 220) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"

def clean_text_for_wordcloud(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Place stats for rich tooltip
# ----------------------------
def build_place_stats(df_articles: pd.DataFrame, max_examples: int = 3) -> dict:
    stats = {}

    def add(place_type, place, topic, title, preview):
        key = f"{place_type}:{place}"
        if key not in stats:
            stats[key] = {"count": 0, "topic_counts": {}, "examples": []}

        stats[key]["count"] += 1
        if topic:
            stats[key]["topic_counts"][topic] = stats[key]["topic_counts"].get(topic, 0) + 1

        if len(stats[key]["examples"]) < max_examples and (title or preview):
            stats[key]["examples"].append((title, preview))

    for _, r in df_articles.iterrows():
        topic = str(r.get("topic", "") or "")
        title = truncate(r.get("title", ""), 90)
        snippet = truncate(r.get("snippet", ""), 160)
        content = truncate(r.get("content", ""), 200)
        preview = snippet if snippet else content

        # Updated to new columns
        cities = split_places(r.get("city", None))
        countries = [normalize_country(x) for x in split_places(r.get("country", None))]

        for c in cities:
            add("city", c, topic, title, preview)
        for c in countries:
            add("country", c, topic, title, preview)

    final = {}
    for key, d in stats.items():
        top_sorted = sorted(d["topic_counts"].items(), key=lambda x: x[1], reverse=True)
        t1 = top_sorted[0] if len(top_sorted) >= 1 else ("—", 0)
        t2 = top_sorted[1] if len(top_sorted) >= 2 else ("—", 0)

        lines = []
        for i, (t, p) in enumerate(d["examples"], start=1):
            t = t or "Untitled"
            p = p or ""
            lines.append(f"<div style='margin-top:6px'><b>{i}. {t}</b><br/>{p}</div>")

        final[key] = {
            "count": int(d["count"]),
            "top_topic": t1[0],
            "top_topic_count": int(t1[1]),
            "top_topic2": t2[0],
            "top_topic2_count": int(t2[1]),
            "examples_html": "".join(lines) if lines else "<div style='color:#666'>No examples</div>",
        }
    return final


# ----------------------------
# Geocoding for cities
# ----------------------------
def explode_city_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        # Updated to new column 'city'
        for c in split_places(r.get("city", None)):
            rows.append({"place": c})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby("place", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
    return out

def geocode_places(city_df: pd.DataFrame, cache: dict, max_places: int = 150) -> pd.DataFrame:
    if city_df.empty:
        return city_df.assign(lat=np.nan, lon=np.nan)

    geolocator = Nominatim(user_agent="topic-geo-dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    target = city_df.head(max_places).copy()
    lats, lons = [], []

    with st.spinner("Geocoding cities (cached; first run may take a while)…"):
        prog = st.progress(0)
        total = len(target)
        for i, row in enumerate(target.itertuples(index=False), start=1):
            key = f"city:{row.place}"
            if key in cache:
                lat, lon = cache[key]
            else:
                try:
                    loc = geocode(row.place)
                except Exception:
                    loc = None
                if loc is None:
                    lat, lon = (None, None)
                else:
                    lat, lon = (loc.latitude, loc.longitude)
                cache[key] = [lat, lon]

            lats.append(lat)
            lons.append(lon)
            prog.progress(int(i / total * 100))

    target["lat"] = pd.to_numeric(lats, errors="coerce")
    target["lon"] = pd.to_numeric(lons, errors="coerce")
    target = target.dropna(subset=["lat", "lon"]).copy()
    return target


# ----------------------------
# Country GeoJSON choropleth
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

def build_country_geojson(df_f: pd.DataFrame, place_stats: dict) -> dict:
    world = load_world_geojson()

    mentions = []
    for _, r in df_f.iterrows():
        # Updated to new column 'country'
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
        top_topic = s["top_topic"] if s else "—"
        top_topic_count = int(s["top_topic_count"]) if s else 0
        top_topic2 = s["top_topic2"] if s else "—"
        top_topic2_count = int(s["top_topic2_count"]) if s else 0
        examples_html = s["examples_html"] if s else "<div style='color:#666'>No examples</div>"

        feat.setdefault("properties", {})
        feat["properties"]["fill_color"] = color_scale(cnt, max_count)

        feat["properties"]["kind"] = "Country"
        feat["properties"]["label"] = name
        feat["properties"]["count"] = cnt
        feat["properties"]["top_topic"] = top_topic
        feat["properties"]["top_topic_count"] = top_topic_count
        feat["properties"]["top_topic2"] = top_topic2
        feat["properties"]["top_topic2_count"] = top_topic2_count
        feat["properties"]["examples_html"] = examples_html

        # flatten for tooltip
        feat["kind"] = "Country"
        feat["label"] = name
        feat["count"] = cnt
        feat["top_topic"] = top_topic
        feat["top_topic_count"] = top_topic_count
        feat["top_topic2"] = top_topic2
        feat["top_topic2_count"] = top_topic2_count
        feat["examples_html"] = examples_html

    return world


# ----------------------------
# City GeoJSON points (flattened for tooltips)
# ----------------------------
def build_city_points_geojson(geo_df: pd.DataFrame) -> dict:
    feats = []
    for _, r in geo_df.iterrows():
        place = str(r["place"])
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": {
                "kind": "City",
                "label": place,
                "count": int(r["count"]),
                "top_topic": str(r.get("top_topic", "—")),
                "top_topic_count": int(r.get("top_topic_count", 0)),
                "top_topic2": str(r.get("top_topic2", "—")),
                "top_topic2_count": int(r.get("top_topic2_count", 0)),
                "examples_html": str(r.get("examples_html", "<div style='color:#666'>No examples</div>")),
                "radius": float(r.get("radius", 4000.0)),
            }
        }

        feat["kind"] = "City"
        feat["label"] = place
        feat["count"] = int(r["count"])
        feat["top_topic"] = str(r.get("top_topic", "—"))
        feat["top_topic_count"] = int(r.get("top_topic_count", 0))
        feat["top_topic2"] = str(r.get("top_topic2", "—"))
        feat["top_topic2_count"] = int(r.get("top_topic2_count", 0))
        feat["examples_html"] = str(r.get("examples_html", "<div style='color:#666'>No examples</div>"))

        feats.append(feat)

    return {"type": "FeatureCollection", "features": feats}


# ----------------------------
# UI Top
# ----------------------------
colA, colB = st.columns([1, 1])
with colA:
    st.title("Dashboard")
with colB:
    if st.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.rerun()


# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Cannot find {DATA_PATH}. Update DATA_PATH at top.")
    st.stop()

df = load_data_no_cache(DATA_PATH)


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

map_mode = st.sidebar.selectbox("Map mode", ["Country (filled)", "City bubbles", "Both"], index=0)

topN_city = st.sidebar.slider("Max cities to geocode/map (top by frequency)", 25, 300, 150, step=25)
months_back = st.sidebar.slider("Topic popularity window (last N months)", 1, 36, 6, step=1)
top_k_topics = st.sidebar.slider("Top topics to show", 5, 30, 12, step=1)
max_rows_wordcloud = st.sidebar.slider("Max rows for wordcloud (speed)", 50, 5000, 2000, step=50)


# ----------------------------
# Filter
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

place_stats = build_place_stats(df_f, max_examples=3)


# ----------------------------
# MAP
# ----------------------------
st.divider()
st.subheader("Map (hover to see details)")

tooltip_html = """
<div style="font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; width: 380px;">
  <div style="font-size: 22px; font-weight: 800; margin-bottom: 6px;">
    {kind}: {label}
  </div>

  <div style="color:#666; font-size: 13px; margin-bottom: 8px;">
    Hover card • Mentions & top topics
  </div>

  <div style="display:flex; gap:14px; align-items:center; margin-bottom: 10px;">
    <div style="
      width:78px; height:78px; border-radius: 50%;
      background:#2E6BB3; color:white;
      display:flex; align-items:center; justify-content:center;
      font-size: 30px; font-weight: 900;">
      {count}
    </div>

    <div style="flex:1;">
      <div style="font-size: 14px; font-weight: 800;">Top topic</div>
      <div style="font-size: 14px; color:#222;">
        {top_topic} <span style="color:#666;">({top_topic_count})</span>
      </div>

      <div style="margin-top:6px; font-size: 14px; font-weight: 800;">Second</div>
      <div style="font-size: 14px; color:#222;">
        {top_topic2} <span style="color:#666;">({top_topic2_count})</span>
      </div>
    </div>
  </div>

  <div style="margin-top:8px;">
    <div style="font-size: 14px; font-weight: 800; margin-bottom: 4px;">Examples</div>
    <div style="font-size: 12.5px; color:#222; line-height: 1.35;">
      {examples_html}
    </div>
  </div>
</div>
"""

layers = []

if map_mode in ["Country (filled)", "Both"]:
    world_geo = build_country_geojson(df_f, place_stats)
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
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
        cache = load_geocache(GEOCACHE_PATH)
        geo_df = geocode_places(city_counts, cache, max_places=topN_city)
        save_geocache(cache, GEOCACHE_PATH)

        if not geo_df.empty:
            def stat(place: str, field: str):
                s = place_stats.get(f"city:{place}")
                if not s:
                    return "—" if field not in ["top_topic_count", "top_topic2_count"] else 0
                return s[field]

            geo_df["top_topic"] = geo_df["place"].apply(lambda x: stat(x, "top_topic"))
            geo_df["top_topic_count"] = geo_df["place"].apply(lambda x: stat(x, "top_topic_count"))
            geo_df["top_topic2"] = geo_df["place"].apply(lambda x: stat(x, "top_topic2"))
            geo_df["top_topic2_count"] = geo_df["place"].apply(lambda x: stat(x, "top_topic2_count"))
            geo_df["examples_html"] = geo_df["place"].apply(lambda x: stat(x, "examples_html"))
            geo_df["radius"] = np.clip(np.sqrt(geo_df["count"].astype(float)) * 2500, 3000, 35000)

            city_geojson = build_city_points_geojson(geo_df)

            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=city_geojson,
                    stroked=False,
                    filled=True,
                    pickable=True,
                    get_fill_color=[30, 110, 190, 160],
                    get_point_radius="properties.radius",
                    point_radius_min_pixels=3,
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
            "maxWidth": "420px",
            "padding": "12px",
            "border": "1px solid rgba(0,0,0,0.12)",
            "borderRadius": "10px",
            "boxShadow": "0 8px 24px rgba(0,0,0,0.15)",
        },
    },
    height=720,
)

st.pydeck_chart(deck, use_container_width=True)


# ----------------------------
# BELOW MAP: Topic popularity + WordCloud
# ----------------------------
st.divider()

col_pop, col_wc = st.columns([1, 1], gap="large")

# We'll compute df_pop + ts data ONCE so both plots can use them
df_pop = df.copy()
if date_range is not None:
    start_d, end_d = date_range
    df_pop = df_pop[
        (df_pop["published_dt"].dt.date >= start_d) &
        (df_pop["published_dt"].dt.date <= end_d)
    ]
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

        # ---- Bar chart (top topics) ----
        topic_counts = (
            df_pop["topic"].dropna().value_counts().head(top_k_topics)
            .rename_axis("topic").reset_index(name="count")
        )

        fig_bar = px.bar(
            topic_counts.sort_values("count", ascending=True),
            x="count", y="topic", orientation="h",
            labels={"count": "Articles", "topic": "Topic"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Prepare time series data (we will plot it full-width later) ----
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
        fig2 = plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig2)

# FULL-WIDTH popularity over time (outside columns)
st.subheader("Popularity over time")

if "ts_counts" not in locals() or ts_counts.empty:
    st.info("Not enough data to plot popularity over time.")
else:
    fig_ts = px.line(
        ts_counts,
        x="month",
        y="count",
        color="topic",
        markers=True,
        labels={"month": "Month", "count": "Articles", "topic": "Topic"},
    )
    st.plotly_chart(fig_ts, use_container_width=True)

st.divider()
st.subheader("Articles in current view")
# Updated mapping to `country` and `city` rather than extracted_*
show_cols = [c for c in ["published", "topic", "title", "real_link", "link", "country", "city"] if c in df_f.columns]
st.dataframe(df_f[show_cols], use_container_width=True, hide_index=True)