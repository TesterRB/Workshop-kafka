# app.py
import os
import glob
import math
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import joblib

# ================================================================
# 0) PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Happiness Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🌍 Happiness Analytics Dashboard")
st.caption(
    "Exploratory analytics, model diagnostics and geospatial patterns of World Happiness data.")

# ================================================================
# 1) DB CONNECTION + DATA LOAD (CACHE)
# ================================================================
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
DW_SCHEMA = "happiness_dw"
TABLE_NAME = "happiness_predictions"  # << Nombre corregido


@st.cache_data(show_spinner=True)
def load_data():
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DW_SCHEMA}"
    )
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME};", engine)

    # Normalizar nombres esperados
    # Si tu tabla ya tiene estas columnas, esto no cambia nada.
    expected = [
        "Happiness_Rank", "Country", "Happiness_Score", "GDP_per_Capita", "Social_Support",
        "Life_Expectancy", "Freedom", "Government_Corruption", "Generosity", "Year",
        "Happiness_Score_Predicted"
    ]
    # Asegura tipos básicos
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    numeric_cols = [
        "Happiness_Score", "Happiness_Score_Predicted", "GDP_per_Capita", "Social_Support",
        "Life_Expectancy", "Freedom", "Government_Corruption", "Generosity"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derivados para diagnósticos
    if {"Happiness_Score", "Happiness_Score_Predicted"}.issubset(df.columns):
        df["Residual"] = df["Happiness_Score"] - \
            df["Happiness_Score_Predicted"]
        df["Abs_Error"] = df["Residual"].abs()

    # Relleno mínimo de Country (evita nulos en mapas/plots)
    if "Country" in df.columns:
        df["Country"] = df["Country"].fillna("Unknown")

    return df


df = load_data()

# ================================================================
# 2) SIDEBAR – GLOBAL FILTERS (start empty) + VIEW MODE
# ================================================================
st.sidebar.header("Global Filters")

# Year selectable (not slider) – default "All"
years = sorted(df["Year"].dropna().unique().tolist()
               ) if "Year" in df.columns else []
year_choice = st.sidebar.selectbox(
    "Year",
    ["All years"] + years,
    index=0
)

# Country multiselect – start empty => all
all_countries = sorted(df["Country"].dropna().unique(
).tolist()) if "Country" in df.columns else []
countries_choice = st.sidebar.multiselect(
    "Country (optional)", all_countries, default=[])

# GDP range filter (start empty => all). We build min/max then let user choose.
if "GDP_per_Capita" in df.columns:
    gdp_min = float(np.nanmin(df["GDP_per_Capita"]))
    gdp_max = float(np.nanmax(df["GDP_per_Capita"]))
    gdp_range = st.sidebar.slider(
        "GDP per Capita range (filter optional)",
        min_value=float(np.floor(gdp_min*100)/100),
        max_value=float(np.ceil(gdp_max*100)/100),
        value=(float(np.floor(gdp_min*100)/100),
               float(np.ceil(gdp_max*100)/100))
    )
else:
    gdp_range = None

# View toggle: Actual / Predicted / Both – default Actual
view_mode = st.sidebar.radio(
    "Which values to show in charts?",
    options=["Actual", "Predicted", "Both"],
    index=0,
    help="Controls charts only. The bottom ML metrics section always uses both actual and predicted from full data."
)

# Apply filters (empty means all)
filtered = df.copy()
if year_choice != "All years":
    filtered = filtered[filtered["Year"] == year_choice]

if countries_choice:
    filtered = filtered[filtered["Country"].isin(countries_choice)]

if gdp_range and "GDP_per_Capita" in filtered.columns:
    filtered = filtered[
        (filtered["GDP_per_Capita"] >= gdp_range[0]) & (
            filtered["GDP_per_Capita"] <= gdp_range[1])
    ]

# ================================================================
# Helper: choose which value(s) build traces by view_mode
# ================================================================


def add_actual_trace(fig, x, y, name="Actual", **kwargs):
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=name, **kwargs))


def add_pred_trace(fig, x, y, name="Predicted", **kwargs):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                  name=name, line=dict(dash="dash"), **kwargs))


def metric_block(df_scope):
    # KPIs: mean actual, mean predicted, MAE or MAPE, #countries, selected year
    if df_scope.empty:
        return (np.nan, np.nan, np.nan, 0, year_choice)

    mean_actual = df_scope["Happiness_Score"].mean(
        skipna=True) if "Happiness_Score" in df_scope else np.nan
    mean_pred = df_scope["Happiness_Score_Predicted"].mean(
        skipna=True) if "Happiness_Score_Predicted" in df_scope else np.nan

    # MAE & MAPE (on rows with both)
    valid = df_scope.dropna(
        subset=["Happiness_Score", "Happiness_Score_Predicted"])
    mae = mean_absolute_error(
        valid["Happiness_Score"], valid["Happiness_Score_Predicted"]) if len(valid) else np.nan
    try:
        mape = mean_absolute_percentage_error(
            valid["Happiness_Score"], valid["Happiness_Score_Predicted"]) if len(valid) else np.nan
    except Exception:
        mape = np.nan

    n_countries = valid["Country"].nunique() if "Country" in valid.columns else (
        df_scope.shape[0] if len(valid) else 0)
    return (mean_actual, mean_pred, (mae, mape), n_countries, year_choice)


# ================================================================
# 3) TOP KPI CARDS
# ================================================================
col1, col2, col3, col4 = st.columns(4)
mean_actual, mean_pred, (mae_v, mape_v), n_ctry, yr = metric_block(filtered)

col1.metric("Mean Happiness (Actual)",
            f"{mean_actual:.3f}" if not np.isnan(mean_actual) else "—")
col2.metric("Mean Happiness (Predicted)",
            f"{mean_pred:.3f}" if not np.isnan(mean_pred) else "—")
if not np.isnan(mae_v):
    col3.metric("MAE / MAPE", f"{mae_v:.3f}  |  {mape_v*100:.2f}%" if not np.isnan(
        mape_v) else f"{mae_v:.3f}  |  —")
else:
    col3.metric("MAE / MAPE", "—")
col4.metric("# Countries (in selection)", f"{n_ctry}")

st.caption(f"Year selected: **{yr}**" if yr !=
           "All years" else "Year selected: **All years**")

# ================================================================
# 4) LAYOUT – MAP (left) + TOP/BOTTOM BARS (right)
# ================================================================
row1_left, row1_right = st.columns([1.2, 1])

# ---- Choropleth Map
with row1_left:
    st.subheader("Choropleth — Happiness by Country")
    map_df = filtered.copy()
    value_col = "Happiness_Score" if view_mode in [
        "Actual", "Both"] else "Happiness_Score_Predicted"

    if not map_df.empty and value_col in map_df.columns:
        # If multiple years, default to showing the selected or last year available
        # Here we show the current filtered state (already filtered by year if chosen)
        agg = map_df.groupby(["Country", "Year"], as_index=False)[
            value_col].mean()
        map_fig = px.choropleth(
            agg,
            locations="Country",
            locationmode="country names",
            color=value_col,
            hover_name="Country",
            animation_frame="Year" if year_choice == "All years" else None,
            color_continuous_scale="Viridis",
            title=None
        )
        map_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.info("No data available for the current filters.")

# ---- Top/Bottom N bars
with row1_right:
    st.subheader("Top / Bottom N — Happiness Score")
    N = st.slider("Select N", min_value=5, max_value=30, value=10, step=1)
    bars_df = filtered.copy()

    if not bars_df.empty:
        # Work on a single year for top/bottom comparability
        if year_choice == "All years":
            # Use latest year present in the filtered data
            use_year = int(bars_df["Year"].dropna().max()
                           ) if "Year" in bars_df else None
            bars_df = bars_df[bars_df["Year"] ==
                              use_year] if use_year is not None else bars_df
        else:
            use_year = year_choice

        metric_col = "Happiness_Score"
        base = bars_df.dropna(subset=[metric_col, "Country"])
        agg = base.groupby("Country", as_index=False)[metric_col].mean()

        topN = agg.nlargest(N, metric_col)
        botN = agg.nsmallest(N, metric_col)

        c1, c2 = st.columns(2)
        if not topN.empty:
            fig_top = px.bar(
                topN.sort_values(metric_col),
                x=metric_col, y="Country",
                orientation="h",
                title=f"Top {N} Countries (Year: {use_year if use_year else '—'})"
            )
            fig_top.update_layout(yaxis=dict(
                categoryorder="array", categoryarray=topN.sort_values(metric_col)["Country"]))
            c1.plotly_chart(fig_top, use_container_width=True)
        else:
            c1.info("No data for Top N.")

        if not botN.empty:
            fig_bot = px.bar(
                botN.sort_values(metric_col),
                x=metric_col, y="Country",
                orientation="h",
                title=f"Bottom {N} Countries (Year: {use_year if use_year else '—'})"
            )
            fig_bot.update_layout(yaxis=dict(
                categoryorder="array", categoryarray=botN.sort_values(metric_col)["Country"]))
            c2.plotly_chart(fig_bot, use_container_width=True)
        else:
            c2.info("No data for Bottom N.")
    else:
        st.info("No data available for Top/Bottom charts.")

# ================================================================
# 5) SCATTER: GDP vs Happiness (bubble size = Generosity)
# ================================================================
st.subheader("GDP vs Happiness — bubble size = Generosity")
scatter_df = filtered.copy()
if not scatter_df.empty:
    show_cols = {"Country", "Year", "GDP_per_Capita", "Generosity"}
    traces = []
    fig = go.Figure()
    if view_mode in ["Actual", "Both"] and "Happiness_Score" in scatter_df:
        fig.add_trace(go.Scatter(
            x=scatter_df["GDP_per_Capita"],
            y=scatter_df["Happiness_Score"],
            mode="markers",
            marker=dict(size=np.clip(
                (scatter_df["Generosity"].fillna(0)+0.05)*25, 6, 24)),
            name="Actual",
            text=scatter_df["Country"],
            hovertemplate=(
                "Country: %{text}<br>Year: %{customdata[0]}<br>"
                "GDP/cap: %{x:.3f}<br>Happiness: %{y:.3f}<br>Generosity: %{customdata[1]:.3f}"
            ),
            customdata=np.stack(
                [scatter_df["Year"], scatter_df["Generosity"]], axis=1)
        ))
    if view_mode in ["Predicted", "Both"] and "Happiness_Score_Predicted" in scatter_df:
        fig.add_trace(go.Scatter(
            x=scatter_df["GDP_per_Capita"],
            y=scatter_df["Happiness_Score_Predicted"],
            mode="markers",
            marker=dict(symbol="triangle-up",
                        size=np.clip((scatter_df["Generosity"].fillna(0)+0.05)*25, 6, 24)),
            name="Predicted",
            text=scatter_df["Country"],
            hovertemplate=(
                "Country: %{text}<br>Year: %{customdata[0]}<br>"
                "GDP/cap: %{x:.3f}<br>Happiness (Pred): %{y:.3f}<br>Generosity: %{customdata[1]:.3f}"
            ),
            customdata=np.stack(
                [scatter_df["Year"], scatter_df["Generosity"]], axis=1)
        ))
    fig.update_layout(xaxis_title="GDP per Capita",
                      yaxis_title="Happiness Score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data for scatter plot.")

# ================================================================
# 6) CORRELATION HEATMAP (select variables)
# ================================================================
st.subheader("Correlation Matrix (select variables)")
corr_vars_default = ["Happiness_Score", "GDP_per_Capita", "Social_Support", "Life_Expectancy",
                     "Freedom", "Government_Corruption", "Generosity", "Happiness_Score_Predicted"]
available_numeric = [c for c in corr_vars_default if c in filtered.columns]
vars_chosen = st.multiselect(
    "Variables for correlation",
    options=available_numeric,
    default=available_numeric
)
if len(vars_chosen) >= 2:
    corr_df = filtered[vars_chosen].dropna()
    if not corr_df.empty:
        corr = corr_df.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No rows with all selected variables present.")
else:
    st.info("Select at least two variables.")

# ================================================================
# 7) FEATURE IMPORTANCE (from model if available; else proxy)
# ================================================================
st.subheader("Feature Importance (Model)")


def try_load_model():
    # Try patterns
    patterns = ["models/Best_*_Model.pkl",
                "data/processed/Regression_Model_*.pkl", "models/*.pkl"]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    # Use the latest
    path = max(candidates, key=os.path.getmtime)
    try:
        model = joblib.load(path)
        return model, path
    except Exception:
        return None


loaded = try_load_model()
if loaded:
    model, model_path = loaded
    # If linear-like with coef_ and feature_names_in_
    try:
        coefs = np.abs(model.coef_).flatten()
        names = list(getattr(model, "feature_names_in_", [
                     f"X{i}" for i in range(len(coefs))]))
        imp = pd.DataFrame({"Feature": names, "Importance": coefs}).sort_values(
            "Importance", ascending=False)
        st.caption(f"Loaded model: `{os.path.basename(model_path)}`")
        fig_imp = px.bar(imp, x="Importance", y="Feature",
                         orientation="h", title="Model Coefficient Magnitudes")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info(
            "Model loaded, but couldn't extract coefficients. Falling back to proxy importance.")
        loaded = None

if not loaded:
    # Proxy: |corr| with Happiness_Score on filtered scope
    proxy_cols = ["GDP_per_Capita", "Social_Support", "Life_Expectancy",
                  "Freedom", "Government_Corruption", "Generosity"]
    proxy_cols = [c for c in proxy_cols if c in filtered.columns]
    if "Happiness_Score" in filtered and proxy_cols:
        tmp = filtered[["Happiness_Score"] + proxy_cols].dropna()
        if not tmp.empty:
            corr = tmp.corr(numeric_only=True)["Happiness_Score"].drop(
                "Happiness_Score").abs().sort_values(ascending=False)
            imp = pd.DataFrame(
                {"Feature": corr.index, "Importance (|corr|)": corr.values})
            fig_imp_proxy = px.bar(imp, x="Importance (|corr|)", y="Feature", orientation="h",
                                   title="Proxy Importance (|corr with Happiness_Score|)")
            st.plotly_chart(fig_imp_proxy, use_container_width=True)
        else:
            st.info("Not enough data to compute proxy importance.")

# ================================================================
# 8) TIME SERIES — Top/Bottom N countries (multi-line)
# ================================================================
st.subheader("Time Series — Top/Bottom N Countries")
ts_N = st.slider("Select N for time series panels", 3, 15, 5, 1, key="tsN")

ts_df = filtered.copy()
if not ts_df.empty and "Happiness_Score" in ts_df:
    # Rank countries by average Happiness_Score over current scope
    rank = ts_df.groupby("Country", as_index=False)["Happiness_Score"].mean()
    topC = rank.nlargest(ts_N, "Happiness_Score")["Country"].tolist()
    botC = rank.nsmallest(ts_N, "Happiness_Score")["Country"].tolist()

    left_ts, right_ts = st.columns(2)

    def make_ts(countries, title):
        sub = ts_df[ts_df["Country"].isin(countries)]
        fig = go.Figure()
        for ctry in sorted(sub["Country"].unique()):
            subc = sub[sub["Country"] == ctry].sort_values("Year")
            if view_mode in ["Actual", "Both"] and "Happiness_Score" in subc:
                fig.add_trace(go.Scatter(x=subc["Year"], y=subc["Happiness_Score"],
                                         mode="lines+markers", name=f"{ctry} (A)"))
            if view_mode in ["Predicted", "Both"] and "Happiness_Score_Predicted" in subc:
                fig.add_trace(go.Scatter(x=subc["Year"], y=subc["Happiness_Score_Predicted"],
                                         mode="lines+markers", name=f"{ctry} (P)", line=dict(dash="dash")))
        fig.update_layout(title=title, xaxis_title="Year",
                          yaxis_title="Happiness Score")
        return fig

    left_ts.plotly_chart(
        make_ts(topC, f"Top {ts_N} — Time Series"), use_container_width=True)
    right_ts.plotly_chart(
        make_ts(botC, f"Bottom {ts_N} — Time Series"), use_container_width=True)
else:
    st.info("Not enough data for time series.")

# ================================================================
# 9) ACTUAL vs PREDICTED — identity line & grouped bars
# ================================================================
st.subheader("Actual vs Predicted — per Country/Year")

avp_df = filtered.dropna(
    subset=["Happiness_Score", "Happiness_Score_Predicted"])
if not avp_df.empty:
    tabs = st.tabs(["Scatter with identity", "Grouped bars (single year)"])

    with tabs[0]:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=avp_df["Happiness_Score"], y=avp_df["Happiness_Score_Predicted"],
            mode="markers",
            text=avp_df["Country"] + " - " + avp_df["Year"].astype(str),
            hovertemplate="Country-Year: %{text}<br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}"
        ))
        lim_min = float(min(avp_df["Happiness_Score"].min(
        ), avp_df["Happiness_Score_Predicted"].min()))
        lim_max = float(max(avp_df["Happiness_Score"].max(
        ), avp_df["Happiness_Score_Predicted"].max()))
        fig_sc.add_trace(go.Scatter(x=[lim_min, lim_max], y=[
                         lim_min, lim_max], mode="lines", name="Identity", line=dict(color="gray")))
        fig_sc.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[1]:
        # pick a single year for grouped bars
        if year_choice == "All years":
            year_bar = int(avp_df["Year"].dropna().max())
        else:
            year_bar = year_choice
        sub = avp_df[avp_df["Year"] == year_bar]
        if sub.empty:
            st.info("No data for the selected year.")
        else:
            sub = sub.sort_values("Happiness_Score", ascending=False).head(25)
            fig_bar = go.Figure(data=[
                go.Bar(name="Actual", x=sub["Country"],
                       y=sub["Happiness_Score"]),
                go.Bar(name="Predicted",
                       x=sub["Country"], y=sub["Happiness_Score_Predicted"])
            ])
            fig_bar.update_layout(
                barmode="group", xaxis_tickangle=-45, title=f"Actual vs Predicted — {year_bar}")
            st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Need both Actual and Predicted to show this section.")

# ================================================================
# 10) DISTRIBUTIONS — Box/Violin
# ================================================================
st.subheader("Distributions — Boxplots / Violin")
dist_var = st.selectbox(
    "Select variable",
    options=[c for c in ["Happiness_Score", "Happiness_Score_Predicted", "GDP_per_Capita", "Social_Support",
                         "Life_Expectancy", "Freedom", "Government_Corruption", "Generosity"] if c in filtered.columns],
    index=0
)
dist_mode = st.radio("Plot type", ["Boxplot", "Violin"], horizontal=True)
if not filtered.empty:
    if dist_mode == "Boxplot":
        fig_d = px.box(filtered, x="Year", y=dist_var,
                       points="outliers", title=f"{dist_var} by Year")
    else:
        fig_d = px.violin(filtered, x="Year", y=dist_var,
                          box=True, points="all", title=f"{dist_var} by Year")
    st.plotly_chart(fig_d, use_container_width=True)
else:
    st.info("No data to plot distributions.")

# ================================================================
# 11) RADAR PROFILE (2–3 countries)
# ================================================================
st.subheader("Radar — Country multivariate profile")
radar_vars = [c for c in ["GDP_per_Capita", "Social_Support", "Life_Expectancy",
                          "Freedom", "Government_Corruption", "Generosity"] if c in filtered.columns]
radar_countries = st.multiselect("Select 1–3 countries", options=sorted(
    filtered["Country"].unique().tolist()), default=[], max_selections=3)
if radar_countries and radar_vars:
    # Normalize per variable (0–1) for radar comparability
    dat = filtered.copy()
    # Choose a single year if "All years": use latest to compare snapshot
    if year_choice == "All years" and "Year" in dat.columns:
        latest = int(dat["Year"].max())
        dat = dat[dat["Year"] == latest]
    # Aggregate by mean per country for the year
    agg = dat.groupby("Country", as_index=False)[
        radar_vars].mean().set_index("Country")
    sub = agg.loc[[c for c in radar_countries if c in agg.index]].copy()
    # Min-max scale each variable
    for v in radar_vars:
        mn, mx = agg[v].min(), agg[v].max()
        if mx > mn:
            sub[v] = (sub[v]-mn)/(mx-mn)
        else:
            sub[v] = 0.5

    # Build radar
    fig_rad = go.Figure()
    categories = radar_vars + [radar_vars[0]]
    for ctry in sub.index:
        vals = sub.loc[ctry, radar_vars].values.tolist()
        vals += [vals[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=categories, fill="toself", name=ctry))
    fig_rad.update_layout(polar=dict(radialaxis=dict(
        visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig_rad, use_container_width=True)
else:
    st.info("Choose 1–3 countries and ensure variables are available.")

# ================================================================
# 12) RESIDUALS — histogram + residual vs predicted
# ================================================================
st.subheader("Model Residuals")
res_df = filtered.dropna(subset=["Residual", "Happiness_Score_Predicted"])
if not res_df.empty:
    c1, c2 = st.columns(2)
    with c1:
        fig_h = px.histogram(res_df, x="Residual", nbins=30,
                             title="Residuals Histogram")
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        fig_r = px.scatter(res_df, x="Happiness_Score_Predicted", y="Residual", hover_name="Country", trendline="ols",
                           title="Residual vs Predicted")
        st.plotly_chart(fig_r, use_container_width=True)
else:
    st.info("Residuals require both actual and predicted values.")

# ================================================================
# 13) SMALL MULTIPLES by Year (GDP vs Happiness)
# ================================================================
st.subheader("Small Multiples — GDP vs Happiness by Year")
sm_df = filtered.dropna(subset=["GDP_per_Capita", "Happiness_Score"])
if not sm_df.empty:
    fig_sm = px.scatter(
        sm_df, x="GDP_per_Capita", y="Happiness_Score",
        facet_col="Year", facet_col_wrap=4,
        hover_name="Country", trendline=None, opacity=0.8,
        title=None
    )
    fig_sm.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sm, use_container_width=True)
else:
    st.info("Not enough data for small multiples.")

# ================================================================
# 14) INTERACTIVE TABLE
# ================================================================
st.subheader("Interactive Table — raw data")
show_cols = [c for c in [
    "Country", "Year", "Happiness_Rank", "Happiness_Score", "Happiness_Score_Predicted",
    "GDP_per_Capita", "Social_Support", "Life_Expectancy", "Freedom", "Government_Corruption", "Generosity", "Residual", "Abs_Error"
] if c in filtered.columns]
if not filtered.empty and show_cols:
    st.dataframe(filtered[show_cols].sort_values(
        ["Year", "Country"]), use_container_width=True, height=420)
else:
    st.info("No rows to display.")

# ================================================================
# 15) BOTTOM PANEL — ML METRICS (ALWAYS FROM FULL DATA, UNFILTERED BY VIEW TOGGLE)
# ================================================================
st.markdown("---")
st.header("Model Metrics (Global, not affected by the view toggle)")


def compute_global_metrics(dataframe):
    d = dataframe.dropna(
        subset=["Happiness_Score", "Happiness_Score_Predicted"]).copy()
    if d.empty:
        return None
    y = d["Happiness_Score"].values
    yhat = d["Happiness_Score_Predicted"].values
    mae = mean_absolute_error(y, yhat)
    rmse = math.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    try:
        mape = mean_absolute_percentage_error(y, yhat)
    except Exception:
        mape = np.nan
    bias = float(np.mean(y - yhat))
    # AIC/BIC on simple residuals (Gaussian, same n, k ~ #features used — unknown; we report using k=8 as proxy)
    n = len(y)
    rss = np.sum((y - yhat)**2)
    sigma2 = rss / n
    # k: use number of predictors in data table proxy
    k = len([c for c in ["GDP_per_Capita", "Social_Support", "Life_Expectancy",
            "Freedom", "Government_Corruption", "Generosity"] if c in dataframe.columns])
    # loglik (Gaussian)
    loglik = -0.5*n*(np.log(2*np.pi*sigma2) + 1)
    AIC = -2*loglik + 2*(k+1)
    BIC = -2*loglik + (k+1)*np.log(n)
    return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape, Bias=bias, AIC=AIC, BIC=BIC, n=n)


global_metrics = compute_global_metrics(df)
if global_metrics:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("R²", f"{global_metrics['R2']:.4f}")
    m2.metric("RMSE", f"{global_metrics['RMSE']:.4f}")
    m3.metric("MAE", f"{global_metrics['MAE']:.4f}")
    m4.metric("MAPE", f"{global_metrics['MAPE']*100:.2f}%")
    m5.metric("Bias (y - ŷ)", f"{global_metrics['Bias']:.4f}")
    m6.metric("n (obs)", f"{global_metrics['n']}")
    st.caption(
        f"AIC: {global_metrics['AIC']:.2f} | BIC: {global_metrics['BIC']:.2f}")
else:
    st.info("Not enough data to compute model metrics.")
