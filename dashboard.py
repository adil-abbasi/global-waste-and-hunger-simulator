import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dwaste & Hunger", layout="wide")
st.title("🌍 Dwaste & Hunger Simulator")

# Load
df = pd.read_csv('data/processed/merged_data.csv').dropna(subset=['GHI_Score'])
avg_waste = df['Waste_Kg_Capita'].mean()
avg_gdp = 12000

# Sidebar
st.sidebar.header("Controls")
red_pct = st.sidebar.slider("Waste Reduction %", 0, 50, 20)
optimism_choices = ["Conservative", "Moderate", "Optimistic"]
optimism = st.sidebar.selectbox("Optimism Level (Impact Scale)", optimism_choices, index=1)
sensitivities = [0.05, 0.1, 0.2]
optimism_idx = optimism_choices.index(optimism)
year = st.sidebar.selectbox("Focus Year", sorted(df['Year'].unique()), index=len(sorted(df['Year'].unique())) - 1)
selected_country = st.sidebar.selectbox("Selected Country", df['Country'].unique())

# Get selected row
row = df[(df['Country'] == selected_country) & (df['Year'] == year)]
if row.empty:
    st.error("No data."); st.stop()
row = row.iloc[0]
baseline_waste = row['Waste_Kg_Capita'] if row['Waste_Kg_Capita'] > 0 else avg_waste
gdp_val = row.get('GDP_Per_Capita', avg_gdp)
st.sidebar.write(f"Baseline Waste: {baseline_waste:.1f} kg/cap | GDP: ${gdp_val:,.0f}")

# Empirical Delta
actual_ghi = row['GHI_Score']
base_sensitivity = sensitivities[optimism_idx]
waste_scale = min(baseline_waste / avg_waste, 2)
gdp_scale = max(1 / (gdp_val / avg_gdp), 0.5)
delta = - (base_sensitivity * (red_pct / 10) * waste_scale * gdp_scale)
sim_ghi = actual_ghi + delta

st.sidebar.write(f"Delta: {delta:.2f} (optimism: {base_sensitivity}, waste scale {waste_scale:.1f}x, GDP scale {gdp_scale:.1f}x)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Current GHI (Actual)", f"{actual_ghi:.1f}")
with col2:
    st.metric("Simulated GHI", f"{sim_ghi:.1f}", delta=f"{delta:.1f}")

label = "better" if delta < 0 else "neutral"
st.write(f"**Impact:** {red_pct}% waste cut → {abs(delta):.1f} points {label} (lower GHI = less hunger).")

# Bar Chart: Current vs Simulated Metrics
st.subheader("📈 Selected Country: Current vs Simulated Metrics")
bar_data = pd.DataFrame({
    'Metric': ['Actual Hunger (GHI)', 'Predicted Hunger (GHI)', 'Actual Waste (kg/cap)', 'Reduced Waste (kg/cap)'],
    'Value': [actual_ghi, sim_ghi, baseline_waste, baseline_waste * (1 - red_pct / 100)],
    'Color': ['red', 'green', 'blue', 'lightblue']
})
fig_bar = px.bar(bar_data, x='Metric', y='Value', color='Color', title=f"{selected_country}: Metrics Comparison ({year})",
                 color_discrete_map={'red': 'red', 'green': 'green', 'blue': 'blue', 'lightblue': 'lightblue'})
fig_bar.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45)
fig_bar.update_traces(marker_line_color='darkgray', marker_line_width=1)
st.plotly_chart(fig_bar, use_container_width=True)

# Policy Recommendation
with st.expander("💡 Policy Recommendation", expanded=False):
    target_ghi = 10
    required_delta = actual_ghi - target_ghi
    required_pct = max(0, (required_delta / (base_sensitivity * waste_scale * gdp_scale)) * 10)
    st.write(f"To reach GHI <{target_ghi}, cut waste by **{required_pct:.0f}%** (at {optimism} level).")
    st.write("**Global Context:** 50% cut could avert 150M undernourished (FAO est).")

# Download Sim Report
st.sidebar.markdown("---")
report_data = pd.DataFrame({
    'Country': [selected_country],
    'Year': [year],
    'Actual GHI': [actual_ghi],
    'Sim GHI': [sim_ghi],
    'Delta': [delta],
    'Waste Cut %': [red_pct],
    'Baseline Waste': [baseline_waste],
    'Reduced Waste': [baseline_waste * (1 - red_pct / 100)]
})
csv = report_data.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download Sim Report (CSV)",
    data=csv,
    file_name=f"{selected_country}_sim_{year}.csv",
    mime="text/csv"
)

# Selected Country Trends
with st.expander("📊 Selected Country: Hunger & Waste Trends", expanded=True):
    country_df = df[df['Country'] == selected_country].sort_values('Year')
    if len(country_df) > 1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=country_df['Year'], y=country_df['GHI_Score'], name='GHI Score', line=dict(color='red', width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=country_df['Year'], y=country_df['Waste_Kg_Capita'], name='Waste (kg/cap)', line=dict(color='blue', width=3)), secondary_y=True)
        fig.update_xaxes(title="Year", gridcolor='lightgray')
        fig.update_yaxes(title="GHI Score", secondary_y=False, gridcolor='lightgray')
        fig.update_yaxes(title="Waste kg/cap", secondary_y=True, gridcolor='lightgray')
        fig.update_layout(title=f"{selected_country}: Hunger vs Waste Trends", hovermode='x unified', template='plotly_white', height=500)
        fig.add_annotation(x=year, y=actual_ghi, text=f"2025: {actual_ghi:.1f}", showarrow=True, arrowhead=2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Limited data—add more years via OWID/FAO.")

# Dynamic 5-Year GHI Forecast (Now Responsive to Waste Reduction)
st.subheader("🔮 5-Year GHI Forecast (Linear Trend with Policy Scenario)")
country_df = df[df['Country'] == selected_country].sort_values('Year')
scenario = st.radio("Forecast Scenario", ["Baseline (No Policy)", "With Waste Reduction Policy"], index=0)
if len(country_df) > 1:
    # Historical trend
    X_hist = country_df['Year'].values.reshape(-1, 1)
    y_hist = country_df['GHI_Score'].values
    model = LinearRegression().fit(X_hist, y_hist)
    slope = model.coef_[0]  # Historical slope (e.g., -0.5/year improving)

    # Future years
    future_years = np.array(range(year, year + 6)).reshape(-1, 1)
    baseline_forecast = model.predict(future_years)

    # Policy scenario: Adjust slope with delta (e.g., faster improvement)
    policy_slope = slope + (delta / 5)  # Spread delta over 5 years
    policy_intercept = actual_ghi - policy_slope * year
    policy_forecast = policy_intercept + policy_slope * future_years.flatten()

    # Confidence bands (std from historical)
    hist_std = np.std(y_hist)
    upper = baseline_forecast + hist_std
    lower = baseline_forecast - hist_std

    # Plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=future_years.flatten(), y=baseline_forecast, mode='lines+markers',
                                      name='Baseline Forecast', line=dict(color='orange', dash='dash'), marker=dict(size=6)))
    fig_forecast.add_trace(go.Scatter(x=future_years.flatten(), y=policy_forecast, mode='lines+markers',
                                      name='With Waste Reduction', line=dict(color='green', width=3), marker=dict(size=6)))
    fig_forecast.add_trace(go.Scatter(x=future_years.flatten(), y=upper, fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                                      showlegend=False, name='Upper Band'))
    fig_forecast.add_trace(go.Scatter(x=future_years.flatten(), y=lower, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                                      showlegend=False, name='Lower Band', fillcolor='rgba(128,128,128,0.2)'))
    fig_forecast.update_layout(title=f"{selected_country} GHI Forecast ({scenario})", xaxis_title="Year", yaxis_title="GHI Score",
                               template='plotly_white', height=500, hovermode='x unified')
    fig_forecast.add_annotation(x=year, y=actual_ghi, text=f"Current: {actual_ghi:.1f}", showarrow=True, arrowhead=2)
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.write(f"**Policy Effect:** Waste cut accelerates GHI decline by {abs(delta / 5):.2f} points/year (spread over 5 years).")
else:
    st.info("Need multi-year data for forecast.")

# Global Map
st.subheader("🗺️ Global Hunger Map (2025)")
latest = df[df['Year'] == 2025].copy()
fig_map = px.choropleth(latest, locations='Country', color='GHI_Score',
                        title='Global Hunger Index 2025',
                        color_continuous_scale='Reds', hover_data=['Waste_Kg_Capita'],
                        labels={'GHI_Score': 'GHI Score'})
fig_map.update_traces(text=latest['Country'])  # Names in hover
fig_map.update_layout(template='plotly_white', height=600, geo=dict(showframe=False, showcoastlines=True))

# Add inside labels for key countries
key_countries = latest.head(10)['Country'].tolist()
country_locs = {
    'India': (20, 77), 'Pakistan': (30, 69), 'China': (35, 105), 'Nigeria': (9, 8), 'Brazil': (-10, -55),
    'USA': (37, -95), 'Indonesia': (-5, 120), 'Bangladesh': (25, 90), 'Russia': (60, 100), 'Mexico': (23, -102)
}
loc_lons, loc_lats, loc_texts = [], [], []
for country in key_countries:
    if country in country_locs:
        lon, lat = country_locs[country]
        loc_lons.append(lon)
        loc_lats.append(lat)
        loc_texts.append(country)
if loc_lons:
    fig_map.add_trace(go.Scattergeo(
        lon=loc_lons, lat=loc_lats, text=loc_texts, mode='text',
        textfont=dict(size=10, color='white'), showlegend=False,
        hoverinfo='skip'
    ))
st.plotly_chart(fig_map, use_container_width=True)

# Global Summary
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    global_ghi = df[df['Year'] == 2025]['GHI_Score'].mean()
    st.metric("Global Avg GHI (2025)", f"{global_ghi:.1f}")
with col2:
    global_waste = df[df['Year'] == 2025]['Waste_Kg_Capita'].mean()
    st.metric("Global Avg Waste", f"{global_waste:.1f} kg/cap")
with col3:
    averted = abs(delta) * (df[df['Year'] == 2025].shape[0] * 0.1)  # Rough est
    st.metric("Est. Averted Hunger (Scale)", f"{averted:.0f}M people")