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

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

DATA_PATH = "data/location_all_df.csv"
GEOCACHE_PATH = "geocode_cache.json"

CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
WORLD_GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

HOST_CITIES_2026 = {
    "toronto",
    "vancouver",
    "mexico city",
    "guadalajara",
    "monterrey",
    "atlanta",
    "boston",
    "dallas",
    "houston",
    "kansas city",
    "los angeles",
    "miami",
    "new york",
    "new york city",
    "new jersey",
    "philadelphia",
    "san francisco",
    "seattle",
}
HOST_COUNTRIES_2026 = {
    "United States",
    "Canada",
    "Mexico",
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

def is_host_country_2026(name: str) -> bool:
    return normalize_country(name) in HOST_COUNTRIES_2026

def load_data_no_cache(data_path: str) -> pd.DataFrame:
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

def normalize_city_name(name: str) -> str:
    s = "" if name is None else str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return HOST_CITY_ALIASES.get(s, s)

def is_host_city_2026(name: str) -> bool:
    return normalize_city_name(name) in HOST_CITIES_2026

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

def same_city(candidate: str, selected: str) -> bool:
    return normalize_city_name(candidate) == normalize_city_name(selected)

# ----------------------------
# Place stats for tooltip
# ----------------------------
def build_place_stats(df_articles: pd.DataFrame, max_examples: int = 3) -> dict:
    stats = {}

    def add(place_type, place, topic):
        key = f"{place_type}:{place}"
        if key not in stats:
            stats[key] = {
                "count": 0,
                "topic_counts": {},
            }

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

        most_talked_topics = ", ".join(
            [f"{topic} ({count})" for topic, count in topic_sorted]
        ) if topic_sorted else "—"

        final[key] = {
            "count": int(d["count"]),
            "most_talked_topics": most_talked_topics,
        }

    return final
# ----------------------------
# Geocoding for cities
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

        # flatten for tooltip reliability
        feat["label"] = name
        feat["kind"] = "Country"
        feat["count"] = cnt
        feat["host_label"] = host_label
        feat["most_talked_topics"] = most_talked_topics

    return world


def filter_articles_for_country(df_articles: pd.DataFrame, country_name: str) -> pd.DataFrame:
    if not country_name:
        return df_articles.iloc[0:0].copy()

    mask = df_articles["country"].fillna("").apply(
        lambda cell: any(normalize_country(c) == normalize_country(country_name) for c in split_places(cell))
    )
    out = df_articles[mask].copy()

    if "published_dt" in out.columns:
        out = out.sort_values("published_dt", ascending=False, na_position="last")
    return out
# ----------------------------
# Article detail helpers
# ----------------------------
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

import re
import html
import pandas as pd
import streamlit as st

import re
import html
import pandas as pd
import streamlit as st

def strip_html_tags(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def render_city_article_panel(df_articles: pd.DataFrame, city_name: str):
    st.subheader(f"Articles for {city_name}")

    if df_articles.empty:
        st.info("No articles found for this city in the current filters.")
        return

    host_text = "Official 2026 World Cup host city" if is_host_city_2026(city_name) else "Non-host city"
    st.caption(host_text)

    with st.container(border=True):
        for i, (_, row) in enumerate(df_articles.iterrows(), start=1):
            title = strip_html_tags(row.get("title", "Untitled"))
            url = row.get("real_link", None)

            if pd.notna(url) and str(url).strip():
                st.markdown(f"{i}. {title} [open link]({url})")
            else:
                st.markdown(f"{i}. {title}")


def render_country_article_panel(df_articles: pd.DataFrame, country_name: str):
    st.subheader(f"Articles for {country_name}")

    if df_articles.empty:
        st.info("No articles found for this country in the current filters.")
        return

    with st.container(border=True):
        for i, (_, row) in enumerate(df_articles.iterrows(), start=1):
            title = strip_html_tags(row.get("title", "Untitled"))
            url = row.get("real_link", None)

            if pd.notna(url) and str(url).strip():
                st.markdown(f"{i}. {title} [open link]({url})")
            else:
                st.markdown(f"{i}. {title}")
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

map_mode = st.sidebar.selectbox("Map mode", ["Country (filled)", "City bubbles", "Both"], index=2)

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

place_stats = build_place_stats(df_f, max_examples=5)

# ----------------------------
# MAP
# ----------------------------
st.divider()
st.subheader("Map")

tooltip_html = """
<div style="
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  width: 320px;
  line-height: 1.5;
">
  <div style="font-size: 20px; font-weight: 800; margin-bottom: 8px;">
    {label}
  </div>

  <div style="font-size: 14px; color: #444; margin-bottom: 4px;">
    {kind}
  </div>

  <div style="font-size: 14px; color: #9b1c1c; font-weight: 700; margin-bottom: 8px;">
    {host_label}
  </div>

  <div style="font-size: 14px; margin-bottom: 6px;">
    <b>Total articles talking about it:</b> {count}
  </div>

  <div style="font-size: 14px; color: #222;">
    <b>Most talked topics:</b> {most_talked_topics}
  </div>
</div>
"""

layers = []
geo_df = pd.DataFrame()

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
                    return "—" if field == "most_talked_topics" else 0
                return s[field]

            geo_df["most_talked_topics"] = geo_df["place"].apply(lambda x: stat(x, "most_talked_topics"))
            geo_df["is_host_city"] = geo_df["place"].apply(is_host_city_2026)
            geo_df["host_label"] = geo_df["is_host_city"].apply(
                lambda x: "Official 2026 World Cup host city" if x else "Non-host city"
            )
            geo_df["kind"] = "City"
            geo_df["label"] = geo_df["place"]
            geo_df["most_talked_topics"] = geo_df["place"].apply(lambda x: stat(x, "most_talked_topics"))
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

st.pydeck_chart(deck)

selected_city = None
selected_country = None

if map_mode == "Country (filled)":
    country_options = sorted(
        {
            normalize_country(c)
            for cell in df_f["country"].fillna("")
            for c in split_places(cell)
            if str(c).strip()
        }
    )

    if country_options:
        st.caption("Choose a country below to show all articles for that country.")
        selected_country = st.selectbox("Selected country", ["None"] + country_options, index=0)
        if selected_country == "None":
            selected_country = None

else:
    if not geo_df.empty:
        st.caption("Choose a city below to show all articles for that city.")
        city_options = ["None"] + sorted(geo_df["place"].dropna().astype(str).unique().tolist())
        selected_city = st.selectbox("Selected city", city_options, index=0)
        if selected_city == "None":
            selected_city = None

# ----------------------------
# CITY DETAIL PANEL
# ----------------------------
st.divider()

if map_mode == "Country (filled)":
    if selected_country:
        country_articles = filter_articles_for_country(df_f, selected_country)
        render_country_article_panel(country_articles, selected_country)
    else:
        st.subheader("Country details")
        st.info("Choose a country to show all articles for that country here.")
else:
    if selected_city:
        city_articles = filter_articles_for_city(df_f, selected_city)
        render_city_article_panel(city_articles, selected_city)
    else:
        st.subheader("City details")
        st.info("Choose a city to show all articles for that city here.")
# ----------------------------
# BELOW MAP: Topic popularity + WordCloud
# ----------------------------
st.divider()

col_pop, col_wc = st.columns([1, 1], gap="large")

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
            width=900,
            height=450,
            background_color="white",
            stopwords=set(STOPWORDS),
            collocations=False,
        ).generate(text)
        fig2 = plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig2)

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
show_cols = [c for c in ["published", "topic", "title", "real_link", "link", "country", "city"] if c in df_f.columns]
st.dataframe(df_f[show_cols], use_container_width=True, hide_index=True)