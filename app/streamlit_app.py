"""Streamlit dashboard for real estate buyer segmentation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SOURCE_FILES = [DATA_DIR / "clients.csv", DATA_DIR / "properties.csv"]
MODEL_FILES = [
    MODELS_DIR / "segmented_clients.csv",
    MODELS_DIR / "cluster_summary.csv",
    MODELS_DIR / "training_metrics.json",
]

SEGMENT_COLORS = {
    "Global Investors": "#2563EB",
    "First-Time Buyers": "#DB2777",
    "Corporate Buyers": "#D97706",
    "Luxury Investors": "#059669",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(size=13),
)


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def data_version() -> str:
    """Cache key that changes whenever source or model files are updated."""
    tracked = SOURCE_FILES + MODEL_FILES + list(FIGURES_DIR.glob("*.png"))
    return "|".join(f"{path}:{_file_mtime(path)}" for path in tracked)


def source_data_is_stale() -> bool:
    """True when raw CSV files are newer than the last trained model output."""
    if not MODEL_FILES[0].exists():
        return True
    latest_source = max(_file_mtime(path) for path in SOURCE_FILES)
    latest_model = _file_mtime(MODEL_FILES[0])
    return latest_source > latest_model


@st.cache_data
def load_data(_version: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clients = pd.read_csv(MODELS_DIR / "segmented_clients.csv")
    summary = pd.read_csv(MODELS_DIR / "cluster_summary.csv")
    with open(MODELS_DIR / "training_metrics.json") as f:
        metrics = json.load(f)
    return clients, summary, metrics


def retrain_model() -> tuple[bool, str]:
    """Run the training pipeline and regenerate model artifacts."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_model.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, result.stdout.strip() or "Training completed successfully."
    return False, result.stderr.strip() or result.stdout.strip() or "Training failed."


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    if country := st.session_state.get("filter_country"):
        if country != "All":
            filtered = filtered[filtered["country"] == country]

    if region := st.session_state.get("filter_region"):
        if region != "All":
            filtered = filtered[filtered["region"] == region]

    if purpose := st.session_state.get("filter_purpose"):
        if purpose != "All":
            filtered = filtered[filtered["acquisition_purpose"] == purpose]

    if client_type := st.session_state.get("filter_client_type"):
        if client_type != "All":
            filtered = filtered[filtered["client_type"] == client_type]

    return filtered


def styled_figure(fig):
    fig.update_layout(**CHART_LAYOUT)
    return fig


def top_regions(df: pd.DataFrame, n: int = 10) -> list[str]:
    return df["region"].value_counts().head(n).index.tolist()


def segment_profile_chart(df: pd.DataFrame):
    profile = (
        df.groupby("segment_name")
        .agg(
            avg_investment_k=("total_investment", lambda s: s.mean() / 1000),
            avg_satisfaction=("satisfaction_score", "mean"),
            loan_rate=("loan_applied_flag", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    melted = profile.melt(
        id_vars="segment_name",
        var_name="Metric",
        value_name="Value",
    )
    labels = {
        "avg_investment_k": "Avg Investment ($K)",
        "avg_satisfaction": "Avg Satisfaction",
        "loan_rate": "Loan Rate (%)",
    }
    melted["Metric"] = melted["Metric"].map(labels)
    fig = px.bar(
        melted,
        x="segment_name",
        y="Value",
        color="Metric",
        barmode="group",
        title="Segment Profile Comparison",
        labels={"segment_name": "Buyer Segment", "Value": "Score"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return styled_figure(fig)


def render_sidebar(df: pd.DataFrame) -> None:
    st.sidebar.header("Filters")
    st.session_state["filter_country"] = st.sidebar.selectbox(
        "Country",
        ["All"] + sorted(df["country"].unique()),
        index=0,
    )
    st.session_state["filter_region"] = st.sidebar.selectbox(
        "Region",
        ["All"] + sorted(df["region"].unique()),
        index=0,
    )
    st.session_state["filter_purpose"] = st.sidebar.selectbox(
        "Acquisition Purpose",
        ["All"] + sorted(df["acquisition_purpose"].unique()),
        index=0,
    )
    st.session_state["filter_client_type"] = st.sidebar.selectbox(
        "Client Type",
        ["All"] + sorted(df["client_type"].unique()),
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Data Management")
    st.sidebar.caption(f"Loaded buyers: **{len(df):,}**")

    if source_data_is_stale():
        st.sidebar.warning(
            "Source CSV files in `data/` were updated. "
            "Click **Retrain Model** below to apply the new data."
        )

    if st.sidebar.button("Retrain Model", use_container_width=True):
        with st.spinner("Retraining on latest data..."):
            success, message = retrain_model()
        load_data.clear()
        if success:
            st.sidebar.success("Model retrained with latest data.")
            st.rerun()
        else:
            st.sidebar.error(message)

    if st.sidebar.button("Refresh Dashboard", use_container_width=True):
        load_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Where is data stored?")
    st.sidebar.info(
        "No database is used. Data is stored locally as files:\n\n"
        f"- Raw input: `{DATA_DIR}/`\n"
        f"- Trained output: `{MODELS_DIR}/`\n"
        f"- Charts/reports: `{OUTPUTS_DIR}/`"
    )


def overview_tab(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Buyer Segmentation Overview")

    if df.empty:
        st.warning("No buyers match the selected filters.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Buyers", f"{len(df):,}")
    col2.metric("Segments", df["segment_name"].nunique())
    col3.metric("Silhouette Score", metrics.get("silhouette_k4", "N/A"))
    col4.metric("Optimal k (Silhouette)", metrics.get("optimal_k_silhouette", "N/A"))

    c1, c2 = st.columns(2)

    with c1:
        dist = df["segment_name"].value_counts().reset_index()
        dist.columns = ["Segment", "Count"]
        fig = px.pie(
            dist,
            names="Segment",
            values="Count",
            hole=0.45,
            title="Buyer Segment Split",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    with c2:
        if (FIGURES_DIR / "cluster_evaluation.png").exists():
            st.image(
                str(FIGURES_DIR / "cluster_evaluation.png"),
                caption="Model Quality: Elbow Method and Silhouette Score",
            )

    st.plotly_chart(segment_profile_chart(df), use_container_width=True)
    st.caption(
        "Segment profile chart replaces the old dendrogram for a clearer business view "
        "of investment, satisfaction, and loan behavior."
    )


def investor_tab(df: pd.DataFrame) -> None:
    st.subheader("Investor Behavior Dashboard")

    if df.empty:
        st.warning("No buyers match the selected filters.")
        return

    behavior = (
        df.groupby("segment_name")
        .agg(
            avg_investment=("total_investment", "mean"),
            avg_properties=("num_properties", "mean"),
            investment_rate=("is_investor", "mean"),
            loan_rate=("loan_applied_flag", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
        )
        .reset_index()
    )
    behavior["investment_rate"] *= 100
    behavior["loan_rate"] *= 100

    fig = px.bar(
        behavior,
        x="segment_name",
        y="avg_investment",
        color="segment_name",
        title="Average Total Investment by Segment",
        labels={"segment_name": "Segment", "avg_investment": "Avg Investment ($)"},
        color_discrete_map=SEGMENT_COLORS,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.bar(
            behavior,
            x="segment_name",
            y="investment_rate",
            color="segment_name",
            title="Investment Purchase Rate (%)",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(styled_figure(fig2), use_container_width=True)

    with c2:
        fig3 = px.bar(
            behavior,
            x="segment_name",
            y="loan_rate",
            color="segment_name",
            title="Loan Dependency Rate (%)",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(styled_figure(fig3), use_container_width=True)

    purpose_mix = (
        df.groupby(["segment_name", "acquisition_purpose"])
        .size()
        .reset_index(name="count")
    )
    fig4 = px.bar(
        purpose_mix,
        x="segment_name",
        y="count",
        color="acquisition_purpose",
        barmode="group",
        title="Acquisition Purpose by Segment",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(styled_figure(fig4), use_container_width=True)


def geographic_tab(df: pd.DataFrame) -> None:
    st.subheader("Geographic Buyer Analysis")
    st.caption(
        "Charts focus on the top regions and countries so the view stays readable. "
        "Use sidebar filters to drill down further."
    )

    if df.empty:
        st.warning("No buyers match the selected filters.")
        return

    top_n = st.slider("Number of top regions to show", min_value=5, max_value=15, value=10)
    regions = top_regions(df, top_n)
    geo_df = df[df["region"].isin(regions)].copy()
    region_order = geo_df["region"].value_counts().index.tolist()[::-1]

    region_seg = (
        geo_df.groupby(["region", "segment_name"], as_index=False)
        .size()
        .rename(columns={"size": "buyers"})
    )
    region_totals = region_seg.groupby("region")["buyers"].transform("sum")
    region_seg["share_pct"] = (region_seg["buyers"] / region_totals * 100).round(1)

    st.markdown("#### Top Regions by Buyer Volume")
    fig_regions = px.bar(
        region_seg,
        y="region",
        x="buyers",
        color="segment_name",
        orientation="h",
        barmode="stack",
        category_orders={"region": region_order},
        title=f"Buyer Count in Top {top_n} Regions",
        labels={"buyers": "Buyers", "region": "Region", "segment_name": "Segment"},
        color_discrete_map=SEGMENT_COLORS,
    )
    fig_regions.update_layout(height=420, yaxis=dict(autorange="reversed"))
    st.plotly_chart(styled_figure(fig_regions), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Country View")
        country_seg = (
            df.groupby(["country", "segment_name"], as_index=False)
            .size()
            .rename(columns={"size": "buyers"})
        )
        fig_country = px.bar(
            country_seg,
            x="country",
            y="buyers",
            color="segment_name",
            barmode="group",
            title="Buyers by Country",
            labels={"country": "Country", "buyers": "Buyers", "segment_name": "Segment"},
            color_discrete_map=SEGMENT_COLORS,
        )
        st.plotly_chart(styled_figure(fig_country), use_container_width=True)

    with c2:
        st.markdown("#### Segment Mix by Region (%)")
        fig_mix = px.bar(
            region_seg,
            y="region",
            x="share_pct",
            color="segment_name",
            orientation="h",
            barmode="stack",
            category_orders={"region": region_order},
            title="Segment Share Within Each Region",
            labels={"share_pct": "Share (%)", "region": "Region", "segment_name": "Segment"},
            color_discrete_map=SEGMENT_COLORS,
            text="share_pct",
        )
        fig_mix.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
        fig_mix.update_layout(height=420, xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_figure(fig_mix), use_container_width=True)

    st.markdown("#### Region Summary Table")
    region_summary = (
        region_seg.sort_values(["region", "buyers"], ascending=[True, False])
        .groupby("region", as_index=False)
        .agg(
            total_buyers=("buyers", "sum"),
            leading_segment=("segment_name", "first"),
            leading_share_pct=("share_pct", "first"),
        )
        .sort_values("total_buyers", ascending=False)
    )
    region_summary["leading_share_pct"] = region_summary["leading_share_pct"].map(lambda x: f"{x:.1f}%")
    st.dataframe(region_summary, use_container_width=True, hide_index=True)


def build_segment_summary(seg_df: pd.DataFrame) -> dict:
    """Compute segment statistics from filtered data."""
    return {
        "buyers": len(seg_df),
        "avg_age": seg_df["age"].mean(),
        "pct_investment": seg_df["is_investor"].mean() * 100,
        "pct_loan": seg_df["loan_applied_flag"].mean() * 100,
        "avg_satisfaction": seg_df["satisfaction_score"].mean(),
        "avg_investment": seg_df["total_investment"].mean(),
        "top_country": seg_df["country"].mode().iloc[0] if len(seg_df) else "N/A",
        "top_region": seg_df["region"].mode().iloc[0] if len(seg_df) else "N/A",
        "top_referral": seg_df["referral_channel"].mode().iloc[0] if len(seg_df) else "N/A",
    }


def insights_tab(df: pd.DataFrame) -> None:
    st.subheader("Segment Insights Panel")

    if df.empty:
        st.warning("No buyers match the selected filters.")
        return

    available_segments = sorted(df["segment_name"].unique())
    segment = st.selectbox("Select Segment", available_segments)
    seg_df = df[df["segment_name"] == segment]

    if seg_df.empty:
        st.info(f"No buyers in the **{segment}** segment for the current filters.")
        return

    seg_summary = build_segment_summary(seg_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buyers", int(seg_summary["buyers"]))
    c2.metric("Avg Age", f"{seg_summary['avg_age']:.1f}")
    c3.metric("Avg Investment", f"${seg_summary['avg_investment']:,.0f}")
    c4.metric("Avg Satisfaction", f"{seg_summary['avg_satisfaction']:.2f}")

    st.markdown(
        f"""
        **Profile Highlights**
        - Investment rate: **{seg_summary['pct_investment']:.1f}%**
        - Loan dependency: **{seg_summary['pct_loan']:.1f}%**
        - Top country: **{seg_summary['top_country']}**
        - Top region: **{seg_summary['top_region']}**
        - Top referral channel: **{seg_summary['top_referral']}**
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        ref = seg_df["referral_channel"].value_counts().reset_index()
        ref.columns = ["Channel", "Count"]
        st.plotly_chart(
            styled_figure(px.bar(ref, x="Channel", y="Count", title=f"Referral Channels – {segment}")),
            use_container_width=True,
        )
    with c2:
        loan = seg_df["loan_applied"].value_counts().reset_index()
        loan.columns = ["Loan Applied", "Count"]
        st.plotly_chart(
            styled_figure(px.pie(loan, names="Loan Applied", values="Count", title=f"Financing – {segment}")),
            use_container_width=True,
        )

    st.dataframe(
        seg_df[
            [
                "client_id",
                "client_type",
                "country",
                "region",
                "acquisition_purpose",
                "total_investment",
                "num_properties",
                "satisfaction_score",
                "loan_applied",
            ]
        ].head(50),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Parcl Buyer Segmentation",
        page_icon="🏢",
        layout="wide",
    )

    st.title("Real Estate Buyer Segmentation & Investment Profiling")
    st.caption("Machine Learning Market Intelligence Dashboard – Parcl Co. Limited")

    try:
        clients, _, metrics = load_data(data_version())
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python scripts/train_model.py` first, "
            "then refresh this page."
        )
        return

    render_sidebar(clients)
    filtered = apply_filters(clients)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Investor Behavior", "Geographic Analysis", "Segment Insights"]
    )

    with tab1:
        overview_tab(filtered, metrics)
    with tab2:
        investor_tab(filtered)
    with tab3:
        geographic_tab(filtered)
    with tab4:
        insights_tab(filtered)


if __name__ == "__main__":
    main()
