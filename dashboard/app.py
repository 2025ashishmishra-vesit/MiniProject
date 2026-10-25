from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="RL Traffic Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.markdown("""
<style>
/* Force both Streamlit video containers to the same height */
div[data-testid="stVideo"] {
    height: 350px !important;
    overflow: hidden !important;
    background: black !important;
}

/* Force the video element to fill that container */
div[data-testid="stVideo"] video {
    width: 100% !important;
    height: 350px !important;
    max-height: 350px !important;
    object-fit: contain !important;
    background: black !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "logs" / "evaluation"

COMPARISON_FILE = EVALUATION_DIR / "comparison.csv"
STRATEGY_FILE = EVALUATION_DIR / "strategy_metrics.csv"
TIMESERIES_FILE = EVALUATION_DIR / "timeseries.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    comparison = pd.read_csv(COMPARISON_FILE)
    strategy = pd.read_csv(STRATEGY_FILE)
    timeseries = pd.read_csv(TIMESERIES_FILE)

    return comparison, strategy, timeseries


try:
    comparison_df, strategy_df, timeseries_df = load_data()

except FileNotFoundError as e:
    st.error(
        "Evaluation files were not found.\n\n"
        "Run evaluate_compare.py first so that comparison.csv, "
        "strategy_metrics.csv and timeseries.csv are generated."
    )

    st.code(str(e))

    st.stop()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚦 Smart Traffic Management")
st.subheader("Reinforcement Learning Analysis Dashboard")

st.markdown(
    """
    Compare the performance of the **Fixed-Time Traffic Signal Controller**
    against the trained **D3QN Reinforcement Learning Controller**.
    """
)

st.divider()

# ---------------------------------------------------------
# RECORDED SUMO SIMULATIONS (independent of evaluation CSVs)
# ---------------------------------------------------------

st.header("🎬 SUMO Simulation Playback")
st.caption("Recorded fixed-time and D3QN RL simulations (30 seconds each).")

VIDEO_DIR = Path(__file__).resolve().parent / "videos"
fixed_video = VIDEO_DIR / "fixed_time.mp4"
rl_video = VIDEO_DIR / "rl_controlled.mp4"

video_left, video_right = st.columns(2)

with video_left:
    st.subheader("🔴 Fixed-Time Control")
    if fixed_video.is_file():
        st.video(str(fixed_video), format="video/mp4")
    else:
        st.info(f"Video missing: {fixed_video.name}. Place it in dashboard/videos/.")

with video_right:
    st.subheader("🟢 D3QN RL Control")
    if rl_video.is_file():
        st.video(str(rl_video), format="video/mp4")
    else:
        st.info(f"Video missing: {rl_video.name}. Place it in dashboard/videos/.")

st.divider()


# ---------------------------------------------------------
# SCENARIO SELECTION
# ---------------------------------------------------------

scenarios = comparison_df["scenario"].unique().tolist()

selected_scenario = st.selectbox(
    "Select Traffic Scenario",
    scenarios,
)


comparison = comparison_df[
    comparison_df["scenario"] == selected_scenario
].iloc[0]


strategy = strategy_df[
    strategy_df["scenario"] == selected_scenario
]


fixed_row = strategy[
    strategy["strategy"] == "fixed_time"
].iloc[0]


rl_row = strategy[
    strategy["strategy"] == "rl_model"
].iloc[0]


time_data = timeseries_df[
    timeseries_df["scenario"] == selected_scenario
].copy()


# ---------------------------------------------------------
# CONTROLLER COMPARISON
# ---------------------------------------------------------

st.header("Controller Comparison")


left, right = st.columns(2)


# ---------------------------------------------------------
# FIXED TIME
# ---------------------------------------------------------

with left:

    st.subheader("🔴 Fixed-Time Controller")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Average Queue",
            f"{fixed_row['avg_queue']:.2f}"
        )

        st.metric(
            "Average Waiting Time",
            f"{fixed_row['avg_waiting_time']:.2f} s"
        )

    with c2:

        st.metric(
            "Throughput",
            f"{fixed_row['throughput_vehicles']:.0f} vehicles"
        )

        st.metric(
            "Throughput Rate",
            f"{fixed_row['throughput_rate_vph']:.1f} veh/hr"
        )


# ---------------------------------------------------------
# RL MODEL
# ---------------------------------------------------------

with right:

    st.subheader("🟢 D3QN RL Controller")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Average Queue",
            f"{rl_row['avg_queue']:.2f}",
            delta=f"{comparison['queue_reduction_pct']:.2f}% reduction"
        )

        st.metric(
            "Average Waiting Time",
            f"{rl_row['avg_waiting_time']:.2f} s",
            delta=f"{comparison['waiting_reduction_pct']:.2f}% reduction"
        )

    with c2:

        st.metric(
            "Throughput",
            f"{rl_row['throughput_vehicles']:.0f} vehicles"
        )

        st.metric(
            "Throughput Rate",
            f"{rl_row['throughput_rate_vph']:.1f} veh/hr",
            delta=f"{comparison['throughput_improvement_pct']:.2f}%"
        )


st.divider()


# ---------------------------------------------------------
# EMERGENCY VEHICLE PRIORITY
# ---------------------------------------------------------

st.header("🚑 Emergency Vehicle Priority Analysis")

if "emergency_active" in time_data.columns:

    # Handle either real booleans or True/False strings from CSV
    emergency_mask = (
        time_data["emergency_active"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    emergency_data = time_data[emergency_mask].copy()

    if not emergency_data.empty:

        latest = emergency_data.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Emergency Event",
                "DETECTED"
            )

        with c2:
            st.metric(
                "Vehicle",
                str(latest["emergency_vehicle_id"])
            )

        with c3:
            st.metric(
                "Approach",
                str(latest["emergency_direction"]).title()
            )

        with c4:
            action = latest["emergency_action"]

            st.metric(
                "Override Action",
                int(action) if pd.notna(action) else "-"
            )

        st.success(
            "🚑 Emergency priority was activated during this simulation."
        )

        st.subheader("Emergency Priority Events")

        event_table = emergency_data[
            [
                "strategy",
                "sim_time",
                "emergency_vehicle_id",
                "emergency_direction",
                "emergency_action",
                "phase",
                "mode"
            ]
        ].copy()

        event_table = event_table.rename(columns={
            "strategy": "Controller",
            "sim_time": "Simulation Time",
            "emergency_vehicle_id": "Vehicle",
            "emergency_direction": "Approach",
            "emergency_action": "Override Action",
            "phase": "Signal Phase",
            "mode": "Mode"
        })

        event_table["Controller"] = event_table["Controller"].replace({
            "fixed_time": "Fixed-Time",
            "rl_model": "D3QN RL"
        })

        st.dataframe(
            event_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No emergency vehicle priority event occurred "
            "during this simulation."
        )

else:

    st.warning(
        "Emergency data is not available in timeseries.csv. "
        "Regenerate the evaluation after adding the emergency fields."
    )

# ---------------------------------------------------------
# IMPROVEMENT SUMMARY
# ---------------------------------------------------------

st.header("Performance Improvement")

m1, m2, m3 = st.columns(3)


with m1:
    st.metric(
        "Queue Reduction",
        f"{comparison['queue_reduction_pct']:.2f}%"
    )


with m2:
    st.metric(
        "Waiting Time Reduction",
        f"{comparison['waiting_reduction_pct']:.2f}%"
    )


with m3:
    st.metric(
        "Throughput Improvement",
        f"{comparison['throughput_improvement_pct']:.2f}%"
    )


st.divider()


# ---------------------------------------------------------
# TIME SERIES GRAPHS
# ---------------------------------------------------------

st.header("Simulation Analysis")


# Convert names for nicer graph legends
time_data["Controller"] = time_data["strategy"].replace({
    "fixed_time": "Fixed-Time",
    "rl_model": "D3QN RL"
})


# ---------------------------------------------------------
# QUEUE GRAPH
# ---------------------------------------------------------

st.subheader("Queue Length Over Time")


queue_fig = px.line(
    time_data,
    x="sim_time",
    y="total_queue",
    color="Controller",
    labels={
        "sim_time": "Simulation Time (seconds)",
        "total_queue": "Queue Length",
    },
)

queue_fig.update_layout(
    height=450,
    legend_title_text="Controller",
)


st.plotly_chart(
    queue_fig,
    use_container_width=True
)


# ---------------------------------------------------------
# WAITING TIME GRAPH
# ---------------------------------------------------------

st.subheader("Average Waiting Time Over Time")


waiting_fig = px.line(
    time_data,
    x="sim_time",
    y="avg_waiting_time",
    color="Controller",
    labels={
        "sim_time": "Simulation Time (seconds)",
        "avg_waiting_time": "Average Waiting Time (seconds)",
    },
)


waiting_fig.update_layout(
    height=450,
    legend_title_text="Controller",
)


st.plotly_chart(
    waiting_fig,
    use_container_width=True
)


# ---------------------------------------------------------
# APPROACH QUEUES
# ---------------------------------------------------------

st.subheader("Queue by Intersection Approach")


queue_columns = [
    "queue_north",
    "queue_south",
    "queue_east",
    "queue_west",
]


if all(column in time_data.columns for column in queue_columns):

    controller = st.radio(
        "Controller",
        ["D3QN RL", "Fixed-Time"],
        horizontal=True,
    )


    selected_strategy = (
        "rl_model"
        if controller == "D3QN RL"
        else "fixed_time"
    )


    approach_data = time_data[
        time_data["strategy"] == selected_strategy
    ].copy()


    approach_data = approach_data.melt(
        id_vars=["sim_time"],
        value_vars=queue_columns,
        var_name="Approach",
        value_name="Queue",
    )


    approach_data["Approach"] = approach_data["Approach"].replace({
        "queue_north": "North",
        "queue_south": "South",
        "queue_east": "East",
        "queue_west": "West",
    })


    approach_fig = px.line(
        approach_data,
        x="sim_time",
        y="Queue",
        color="Approach",
        labels={
            "sim_time": "Simulation Time (seconds)"
        },
    )


    approach_fig.update_layout(
        height=450
    )


    st.plotly_chart(
        approach_fig,
        use_container_width=True
    )

else:

    st.info(
        "Approach queue columns were not found in timeseries.csv. "
        "Add queue_north, queue_south, queue_east and queue_west "
        "if you want directional queue graphs."
    )


st.divider()


# ---------------------------------------------------------
# RAW RESULTS
# ---------------------------------------------------------

with st.expander("View Raw Evaluation Results"):

    st.subheader("Comparison")

    st.dataframe(
        comparison_df[
            comparison_df["scenario"] == selected_scenario
        ],
        use_container_width=True,
    )


    st.subheader("Strategy Metrics")

    st.dataframe(
        strategy,
        use_container_width=True,
    )