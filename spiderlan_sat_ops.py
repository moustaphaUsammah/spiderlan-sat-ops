from skyfield.api import load, wgs84
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt matplotlib==3.8.4
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import requests
import math
import json
from pathlib import Path
from datetime import timedelta

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="SpiderLan SAT Ops", layout="wide")

st.markdown("""
<style>
.block {
    background:#0f1116;
    border:1px solid #2a2f3a;
    border-radius:14px;
    padding:14px 16px;
    margin-bottom:12px;
}
.k {
    color:#9aa4b2;
    font-size:13px;
    margin-bottom:4px;
}
.v {
    font-size:22px;
    font-weight:700;
}
.s {
    color:#9aa4b2;
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("SpiderLan SAT Ops")
st.caption("Starlink-aware NTN operations, RF analysis, trust scoring, handover intelligence, incident logging, and integrated threat visualization.")

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "analysis_done": False,
    "results_df": None,
    "best_sat_name": None,
    "trust_data": None,
    "handover_data": None,
    "heat_points": None,
    "run_time": None,
    "last_action": None,
    "last_location": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# CONSTANTS + HELPERS
# =========================================================
C = 299_792_458.0
INCIDENT_FILE = Path("security_incidents.jsonl")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fspl_db(distance_km: float, frequency_ghz: float) -> float:
    return 92.45 + 20 * math.log10(max(distance_km, 0.001)) + 20 * math.log10(max(frequency_ghz, 0.001))


def performance_score(elev_deg: float, fspl: float) -> float:
    elev_component = clamp(elev_deg / 90.0, 0, 1) * 60.0
    loss_component = clamp((200.0 - fspl) / 80.0, 0, 1) * 40.0
    return elev_component + loss_component


def quality_label(score: float) -> str:
    if score >= 75:
        return "Excellent"
    if score >= 55:
        return "Good"
    if score >= 35:
        return "Fair"
    return "Weak"


def render_metric(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="block">
            <div class="k">{title}</div>
            <div class="v">{value}</div>
            <div class="s">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def estimate_doppler_hz(sat, observer, ts, base_dt, frequency_hz: float) -> float:
    t1 = ts.utc(
        base_dt.year, base_dt.month, base_dt.day,
        base_dt.hour, base_dt.minute, base_dt.second
    )
    next_dt = base_dt + timedelta(seconds=1)
    t2 = ts.utc(
        next_dt.year, next_dt.month, next_dt.day,
        next_dt.hour, next_dt.minute, next_dt.second
    )

    d1 = (sat - observer).at(t1).altaz()[2].km * 1000.0
    d2 = (sat - observer).at(t2).altaz()[2].km * 1000.0
    radial_velocity = (d2 - d1)
    doppler = -(radial_velocity / C) * frequency_hz
    return float(doppler)


def compute_trust_and_anomaly(sat, observer, ts, base_dt, frequency_hz: float, window_min: int = 10):
    elevations, azimuths, distances, dopplers = [], [], [], []

    for i in range(window_min):
        dt = base_dt + timedelta(minutes=i)
        t = ts.utc(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second
        )
        alt, az, dist = (sat - observer).at(t).altaz()

        elevations.append(float(alt.degrees))
        azimuths.append(float(az.degrees))
        distances.append(float(dist.km))
        dopplers.append(estimate_doppler_hz(sat, observer, ts, dt, frequency_hz))

    elev_diff = np.diff(elevations) if len(elevations) > 1 else np.array([0.0])
    az_diff = np.diff(azimuths) if len(azimuths) > 1 else np.array([0.0])
    doppler_diff = np.diff(dopplers) if len(dopplers) > 1 else np.array([0.0])

    flags = []

    if np.sum(np.abs(elev_diff) > 15) > 1:
        flags.append("Elevation discontinuity")
    if np.sum(np.abs(az_diff) > 40) > 1:
        flags.append("Azimuth discontinuity")
    if np.sum(np.abs(doppler_diff) > 8000) > 1:
        flags.append("Doppler discontinuity")
    if np.any(np.array(elevations) < -5):
        flags.append("Geometry inconsistency")

    avg_dist = float(np.mean(distances))
    trust = 100.0
    trust -= 5.0 * np.sum(np.abs(elev_diff) > 15)
    trust -= 5.0 * np.sum(np.abs(az_diff) > 40)
    trust -= 6.0 * np.sum(np.abs(doppler_diff) > 8000)

    if avg_dist > 2500:
        trust -= 5.0
    if len(flags) >= 3:
        trust -= 10.0

    trust = clamp(trust, 30.0, 100.0)
    anomaly_score = clamp(100.0 - trust, 0.0, 100.0)

    if trust >= 80:
        status = "Nominal"
        threat = "Green"
    elif trust >= 55:
        status = "Suspicious"
        threat = "Yellow"
    else:
        status = "High Risk"
        threat = "Red"

    return {
        "trust_score": round(trust, 2),
        "anomaly_score": round(anomaly_score, 2),
        "security_status": status,
        "threat_level": threat,
        "elevations": elevations,
        "azimuths": azimuths,
        "distances": distances,
        "dopplers": dopplers,
        "flags": flags
    }


def predict_handover(candidates_df, sat_lookup, observer, ts, base_dt, frequency_ghz: float, horizon_min: int = 15):
    best_now = candidates_df.iloc[0]["name"]
    timeline = []

    for minute in range(horizon_min + 1):
        dt = base_dt + timedelta(minutes=minute)
        t = ts.utc(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second
        )
        scored = []

        for name in candidates_df["name"].head(5):
            sat = sat_lookup[name]
            alt, az, dist = (sat - observer).at(t).altaz()
            if alt.degrees > 0:
                loss = fspl_db(float(dist.km), frequency_ghz)
                score = performance_score(float(alt.degrees), loss)
                scored.append((name, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            timeline.append({
                "minute": minute,
                "best_name": scored[0][0],
                "best_score": scored[0][1]
            })

    next_sat = None
    handover_min = None
    for row in timeline[1:]:
        if row["best_name"] != best_now:
            next_sat = row["best_name"]
            handover_min = row["minute"]
            break

    if next_sat is None:
        action = "Stay"
    elif handover_min <= 2:
        action = "Switch Now"
    elif handover_min <= 5:
        action = "Prepare Handover"
    else:
        action = "Stay"

    return {
        "best_now": best_now,
        "next_sat": next_sat,
        "handover_min": handover_min,
        "action": action,
        "timeline": timeline
    }


def write_incident(event: dict):
    with INCIDENT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_incidents() -> pd.DataFrame:
    if not INCIDENT_FILE.exists():
        return pd.DataFrame()
    rows = []
    with INCIDENT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_starlink_tle():
    tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    return load.tle_file(tle_url)


def cloudrf_area_request(api_key: str, lat: float, lon: float, height_m: float, frequency_mhz: float, radius_km: float):
    payload = {
        "site": "SpiderLan",
        "network": "Starlink",
        "engine": 2,
        "transmitter": {
            "lat": lat,
            "lon": lon,
            "alt": height_m,
            "frq": frequency_mhz,
            "txw": 1,
            "bwi": 20
        },
        "receiver": {
            "lat": 0,
            "lon": 0,
            "alt": height_m,
            "rxg": 3,
            "rxs": -100
        },
        "antenna": {
            "txg": 2.15,
            "txl": 0,
        
       
            "azi": 90,
            "tlt": 1,
            "hbw": 120,
            "vbw": 30,
            "pol": "v"
        },
        "environment": {
            "clt": "Minimal.clt",
            "elevation": 1,
            "landcover": 0,
            "buildings": 0,
            "obstacles": 0
        },
        "output": {
            "units": "m",
            "col": "RAINBOW.dBm",
            "out": 2,
            "nf": -100,
            "res": 10,
            "rad": radius_km
        }
    }

    headers = {
        "key": api_key,
        "Content-Type": "application/json"
    }

    return requests.post(
        "https://api.cloudrf.com/area",
        headers=headers,
        json=payload,
        timeout=90
    )
# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Inputs")
    latitude = st.number_input("Latitude", value=30.0331, format="%.6f")
    longitude = st.number_input("Longitude", value=31.3331, format="%.6f")
    height_m = st.number_input("Receiver Height (m)", value=1.5, format="%.1f")
    frequency_ghz = st.number_input("Frequency (GHz)", value=12.0, format="%.2f")
    analysis_horizon = st.slider("Prediction Horizon (min)", 5, 30, 15)
    automation_enabled = st.toggle("Automation Logging", value=True)
    cloudrf_key = st.text_input("CloudRF API Key", type="password")
    cloudrf_radius = st.number_input("CloudRF Radius (km)", value=5.0, format="%.1f")

    run = st.button("Run Analysis", type="primary")
    clear_state = st.button("Clear Results")

if clear_state:
    for k, v in defaults.items():
        st.session_state[k] = v
    st.experimental_rerun()

# =========================================================
# LOAD STARLINK
# =========================================================
try:
    satellites = load_starlink_tle()
    sat_lookup = {sat.name: sat for sat in satellites}
    st.success(f"Loaded {len(satellites)} Starlink satellites")
except Exception as e:
    st.error(f"Failed to load Starlink TLE: {e}")
    st.stop()

# =========================================================
# RUN ANALYSIS ONCE
# =========================================================
if run:
    observer = wgs84.latlon(latitude, longitude, height_m)
    ts = load.timescale()
    base_dt = ts.now().utc_datetime()
    t = ts.utc(
        base_dt.year, base_dt.month, base_dt.day,
        base_dt.hour, base_dt.minute, base_dt.second
    )

    rows = []
    for sat in satellites:
        alt, az, dist = (sat - observer).at(t).altaz()
        if alt.degrees > 0:
            d_km = float(dist.km)
            loss = fspl_db(d_km, frequency_ghz)
            score = performance_score(float(alt.degrees), loss)
            rows.append({
                "name": sat.name,
                "elevation_deg": float(alt.degrees),
                "azimuth_deg": float(az.degrees),
                "distance_km": d_km,
                "fspl_db": loss,
                "performance_score": score
            })

    if not rows:
        st.session_state.analysis_done = False
        st.warning("No visible Starlink satellites at this location right now.")
    else:
        df = pd.DataFrame(rows).sort_values("performance_score", ascending=False).reset_index(drop=True)
        best_name = df.iloc[0]["name"]
        best_sat = sat_lookup[best_name]

        trust = compute_trust_and_anomaly(
            best_sat,
            observer,
            ts,
            base_dt,
            frequency_ghz * 1e9,
            window_min=min(10, analysis_horizon)
        )

        handover = predict_handover(
            df,
            sat_lookup,
            observer,
            ts,
            base_dt,
            frequency_ghz,
            horizon_min=analysis_horizon
        )

        heat_points = []
        grid = np.linspace(-0.05, 0.05, 24)
        for dx in grid:
            for dy in grid:
                lat = latitude + float(dx)
                lon = longitude + float(dy)
                obs = wgs84.latlon(lat, lon, height_m)
                alt, az, dist = (best_sat - obs).at(t).altaz()
                if alt.degrees > 0:
                    val = 200.0 - fspl_db(float(dist.km), frequency_ghz)
                    heat_points.append([lat, lon, val])

        site_score = round(df.iloc[0]["performance_score"], 1)
        if trust["threat_level"] == "Red":
            action = "Investigate signal and avoid critical traffic"
        elif handover["handover_min"] is not None and handover["handover_min"] <= 5:
            action = "Prepare handover"
        elif site_score < 40:
            action = "Relocate or improve site position"
        else:
            action = "Normal operation"

        st.session_state.analysis_done = True
        st.session_state.results_df = df
        st.session_state.best_sat_name = best_name
        st.session_state.trust_data = trust
        st.session_state.handover_data = handover
        st.session_state.heat_points = heat_points
        st.session_state.run_time = base_dt
        st.session_state.last_action = action
        st.session_state.last_location = {
            "lat": latitude,
            "lon": longitude,
            "height_m": height_m,
            "frequency_ghz": frequency_ghz,
        }

        if automation_enabled:
            should_log = (trust["security_status"] != "Nominal") or (
                handover["handover_min"] is not None and handover["handover_min"] <= 5
            )
            if should_log:
                event = {
                    "timestamp_utc": base_dt.isoformat(),
                    "location": {"lat": latitude, "lon": longitude, "height_m": height_m},
                    "best_satellite": best_name,
                    "trust_score": trust["trust_score"],
                    "anomaly_score": trust["anomaly_score"],
                    "security_status": trust["security_status"],
                    "threat_level": trust["threat_level"],
                    "handover_min": handover["handover_min"],
                    "next_satellite": handover["next_sat"],
                    "action": action,
                    "flags": trust["flags"]
                }
                write_incident(event)

# =========================================================
# REQUIRE RESULTS
# =========================================================
if not st.session_state.analysis_done:
    st.info("Set parameters and click Run Analysis.")
    st.stop()

df = st.session_state.results_df
best_name = st.session_state.best_sat_name
trust = st.session_state.trust_data
handover = st.session_state.handover_data
heat_points = st.session_state.heat_points
last_action = st.session_state.last_action
top3 = df.head(3).copy()

site_score = round(df.iloc[0]["performance_score"], 1)
security_status = trust["security_status"]
handover_text = f"{handover['handover_min']} min" if handover["handover_min"] else "No handover soon"

# =========================================================
# USER-FACING CARDS
# =========================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric("Site Suitability", f"{site_score}/100", quality_label(site_score))
with c2:
    render_metric("Security Status", security_status, trust["threat_level"])
with c3:
    render_metric("Handover Risk", handover_text, handover["action"])
with c4:
    render_metric("Recommended Action", last_action, "User-facing decision")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Handover", "Security", "Threat Map", "CloudRF", "Incidents"]
)

with tab1:
    st.subheader("Top 3 Satellites")
    st.dataframe(top3, use_container_width=True)

    summary = pd.DataFrame([{
        "Current Satellite": best_name,
        "Connection Quality": quality_label(top3.iloc[0]["performance_score"]),
        "FSPL (dB)": round(top3.iloc[0]["fspl_db"], 2),
        "Elevation (deg)": round(top3.iloc[0]["elevation_deg"], 2),
        "Azimuth (deg)": round(top3.iloc[0]["azimuth_deg"], 2),
        "Security Status": trust["security_status"],
        "Threat Level": trust["threat_level"],
        "Action": last_action
    }])
    st.subheader("Operational Summary")
    st.dataframe(summary, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Analysis CSV",
        data=csv_data,
        file_name="spiderlan_sat_ops_results.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Handover Intelligence")

    if handover["next_sat"] is None:
        st.write("No handover predicted within the selected horizon.")
    else:
        st.write(f"Next recommended satellite: **{handover['next_sat']}**")
        st.write(f"Estimated handover window: **{handover['handover_min']} min**")
        st.write(f"Recommended action: **{handover['action']}**")

    timeline_df = pd.DataFrame(handover["timeline"])
    st.dataframe(timeline_df, use_container_width=True)

    if not timeline_df.empty:
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.plot(timeline_df["minute"], timeline_df["best_score"])
        ax_h.set_xlabel("Minutes Ahead")
        ax_h.set_ylabel("Best Score")
        ax_h.set_title("Best-Link Trajectory")
        ax_h.grid(True)
        st.pyplot(fig_h)

with tab3:
    st.subheader("Signal Trust & Security")

    c1, c2 = st.columns(2)
    with c1:
        render_metric("Trust Score", f"{trust['trust_score']}/100", "Higher is better")
    with c2:
        render_metric("Anomaly Score", f"{trust['anomaly_score']}/100", "Higher is more suspicious")

    st.subheader("Flags")
    if trust["flags"]:
        for flag in trust["flags"]:
            st.write(f"- {flag}")
    else:
        st.write("No rule-based anomalies detected.")

    trust_df = pd.DataFrame({
        "minute": list(range(len(trust["elevations"]))),
        "elevation_deg": trust["elevations"],
        "distance_km": trust["distances"],
        "doppler_hz": trust["dopplers"]
    })
    st.dataframe(trust_df, use_container_width=True)

    fig_t, ax_t = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_t[0].plot(trust_df["minute"], trust_df["elevation_deg"])
    ax_t[0].set_ylabel("Elevation")
    ax_t[0].grid(True)

    ax_t[1].plot(trust_df["minute"], trust_df["distance_km"])
    ax_t[1].set_ylabel("Distance")
    ax_t[1].grid(True)

    ax_t[2].plot(trust_df["minute"], trust_df["doppler_hz"])
    ax_t[2].set_ylabel("Doppler")
    ax_t[2].set_xlabel("Minutes Ahead")
    ax_t[2].grid(True)

    fig_t.suptitle("Trust Telemetry")
    st.pyplot(fig_t)

with tab4:
    st.subheader("Integrated RF + Threat Map")

    color = {"Green": "green", "Yellow": "orange", "Red": "red"}[trust["threat_level"]]
    map_obj = folium.Map(location=[latitude, longitude], zoom_start=12, tiles="OpenStreetMap")

    if heat_points:
        HeatMap(
            heat_points,
            radius=15,
            blur=20,
            min_opacity=0.4
        ).add_to(map_obj)

    folium.Marker(
        [latitude, longitude],
        popup=f"Receiver Site<br>Lat: {latitude}<br>Lon: {longitude}",
        tooltip="Receiver",
        icon=folium.Icon(color="blue")
    ).add_to(map_obj)

    folium.Circle(
        location=[latitude, longitude],
        radius=1200,
        color=color,
        fill=True,
        fill_opacity=0.15,
        popup=f"Threat Level: {trust['threat_level']}"
    ).add_to(map_obj)

    best_row = df.iloc[0]
    azimuth = float(best_row["azimuth_deg"])
    distance = 0.05
    end_lat = latitude + distance * np.cos(np.radians(azimuth))
    end_lon = longitude + distance * np.sin(np.radians(azimuth))

    folium.PolyLine(
        [(latitude, longitude), (end_lat, end_lon)],
        color="yellow",
        weight=4,
        tooltip="Best Signal Direction"
    ).add_to(map_obj)

    st_folium(map_obj, width=1200, height=600, key="integrated_rf_threat_map")

with tab5:
    st.subheader("CloudRF Integration")
    st.caption("Optional terrain-aware RF layer for the current site.")

    if st.button("Run CloudRF Area Model"):
        if not cloudrf_key.strip():
            st.error("Enter a valid CloudRF API key.")
        else:
            try:
                r = cloudrf_area_request(
                    cloudrf_key.strip(),
                    latitude,
                    longitude,
                    height_m,
                    frequency_ghz * 1000.0,
                    cloudrf_radius
                )

                st.write(f"CloudRF status: {r.status_code}")

                if r.status_code == 200:
                    data = r.json()
                    st.success("CloudRF response received.")
                    if "PNG_Mercator" in data:
                        st.image(data["PNG_Mercator"], caption="CloudRF Area Heatmap")
                    elif "PNG_WGS84" in data:
                        st.image(data["PNG_WGS84"], caption="CloudRF Area Heatmap")
                    else:
                        st.json(data)
                else:
                    st.error("CloudRF request failed.")
                    st.text(r.text)

            except Exception as e:
                st.error(f"CloudRF request error: {e}")

with tab6:
    st.subheader("Incident History")
    hist = load_incidents()

    if hist.empty:
        st.info("No incidents logged yet.")
    else:
        st.dataframe(hist.tail(20), use_container_width=True)

        json_data = hist.to_json(orient="records", force_ascii=False, indent=2)
        st.download_button(
            label="Download Incident History",
            data=json_data,
            file_name="spiderlan_sat_ops_incidents.json",
            mime="application/json"
        )
