import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Devices - Commercial Intelligence",
    page_icon="🎯",
    layout="wide"
)

st.title("Commercial Intelligence Dashboard")
st.markdown("""
Commercial intelligence dashboard: evaluating price elasticity, competition impact, 
customer segmentation, and deal velocity.
""")

# ------------------------------------------------------------------------------
# 2. DATA LOADING AND PREPARATION
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_name = "FME DS Interview - Deal_level_data 09.2026.csv"
    df = pd.read_csv(file_name)
    
    # Clean discount anomalies (0% <= discount <= 50%)
    df_clean = df[(df['discount_pct'] >= 0) & (df['discount_pct'] <= 50)].copy()
    
    # Fill missing values with medians
    df_clean['deal_size_units'] = df_clean['deal_size_units'].fillna(df_clean['deal_size_units'].median())
    df_clean['sales_rep_tenure_years'] = df_clean['sales_rep_tenure_years'].fillna(df_clean['sales_rep_tenure_years'].median())
    df_clean['days_to_close'] = df_clean['days_to_close'].fillna(df_clean['days_to_close'].median())
    
    # Calculate unit price
    df_clean['unit_price_eur'] = df_clean['list_price_eur'] / df_clean['deal_size_units']
    
    # Group discounts into bins
    bins = [-1, 5, 10, 15, 20, 25, 30, 50]
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30%+']
    df_clean['discount_bin'] = pd.cut(df_clean['discount_pct'], bins=bins, labels=labels)
    
    return df_clean

df_clean = load_data()

# ------------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# ------------------------------------------------------------------------------
st.sidebar.header("Global Filters")

selected_regions = st.sidebar.multiselect(
    "Region:", options=df_clean['region'].unique(), default=df_clean['region'].unique()
)

selected_products = st.sidebar.multiselect(
    "Product Type:", options=df_clean['product_type'].unique(), default=df_clean['product_type'].unique()
)

selected_segments = st.sidebar.multiselect(
    "Customer Segment:", options=df_clean['customer_segment'].unique(), default=df_clean['customer_segment'].unique()
)

selected_competitor = st.sidebar.multiselect(
    "Competitor Present:", options=df_clean['competitor_present'].unique(), default=df_clean['competitor_present'].unique()
)

# Apply filters
filtered_df = df_clean[
    (df_clean['region'].isin(selected_regions)) &
    (df_clean['product_type'].isin(selected_products)) &
    (df_clean['customer_segment'].isin(selected_segments)) &
    (df_clean['competitor_present'].isin(selected_competitor))
]

# Helper function to compute 5-quantile profile curve [Min, Q1, Median, Q3, Max]
def calc_quantiles(series):
    if len(series) == 0:
        return [0, 0, 0, 0, 0]
    return [float(x) for x in np.percentile(series.dropna(), [0, 25, 50, 75, 100])]

# ------------------------------------------------------------------------------
# 4. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 1. Market & Volume Overview", 
    "📈 2. Win Rate & Competition", 
    "⏱ 3. Deal Velocity"
])

# ==============================================================================
# TAB 1: MARKET OVERVIEW & VOLUMES
# ==============================================================================
with tab1:
    st.header("Summary Statistics: Medians, Ranges, and Volumes")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deals (N)", len(filtered_df))
    col2.metric("Median Discount", f"{filtered_df['discount_pct'].median():.1f}%")
    col3.metric("Average Win Rate", f"{(filtered_df['win_flag'].mean() * 100):.1f}%")
    col4.metric("Median Volume (units)", f"{filtered_df['deal_size_units'].median():.0f}")
    
    st.markdown("---")
    
    st.subheader("Summary by Segment and Product")
    
    # Calculate summary metrics using unit_price_eur
    summary_df = filtered_df.groupby(['customer_segment', 'product_type']).agg(
        deals_count=('deal_id', 'count'),
        win_rate=('win_flag', 'mean'),
        
        # Unit Price
        price_median=('unit_price_eur', 'median'),
        price_min=('unit_price_eur', 'min'),
        price_max=('unit_price_eur', 'max'),
        
        # Discount
        disc_median=('discount_pct', 'median'),
        disc_min=('discount_pct', 'min'),
        disc_max=('discount_pct', 'max'),
        
        # Days to Close
        days_median=('days_to_close', 'median'),
        days_min=('days_to_close', 'min'),
        days_max=('days_to_close', 'max')
    ).reset_index()
    
    # Formatted explicit ranges [Min – Max]
    summary_df['price_range'] = summary_df.apply(lambda r: f"€{r['price_min']:,.0f} – €{r['price_max']:,.0f}", axis=1)
    summary_df['disc_range'] = summary_df.apply(lambda r: f"{r['disc_min']:.1f}% – {r['disc_max']:.1f}%", axis=1)
    summary_df['days_range'] = summary_df.apply(lambda r: f"{r['days_min']:.0f} – {r['days_max']:.0f} d", axis=1)
    summary_df['Win_Rate_Str'] = (summary_df['win_rate'] * 100).round(1).astype(str) + '%'
    
    # Calculate quantile profile curves for sparkline charts
    quantile_data = []
    for (seg, prod), group in filtered_df.groupby(['customer_segment', 'product_type']):
        quantile_data.append({
            'customer_segment': seg,
            'product_type': prod,
            'price_profile': calc_quantiles(group['unit_price_eur']),
            'disc_profile': calc_quantiles(group['discount_pct']),
            'days_profile': calc_quantiles(group['days_to_close'])
        })
    quantile_df = pd.DataFrame(quantile_data)
    
    if not quantile_df.empty:
        summary_df = summary_df.merge(quantile_df, on=['customer_segment', 'product_type'], how='left')
    else:
        summary_df['price_profile'] = [[] for _ in range(len(summary_df))]
        summary_df['disc_profile'] = [[] for _ in range(len(summary_df))]
        summary_df['days_profile'] = [[] for _ in range(len(summary_df))]
    
    st.dataframe(
        summary_df,
        column_config={
            "customer_segment": "Customer Segment",
            "product_type": "Product Type",
            "deals_count": st.column_config.NumberColumn("Deals (N)", format="%d"),
            "Win_Rate_Str": "Win Rate",
            "price_median": st.column_config.NumberColumn("Median Price (€)", format="€%d"),
            "price_range": "Price Range (Min – Max)",
            "price_profile": st.column_config.LineChartColumn("Price Trend (Q0–Q4)", y_min=0),
            "disc_median": st.column_config.NumberColumn("Median Discount", format="%.1f%%"),
            "disc_range": "Discount Range (Min – Max)",
            "disc_profile": st.column_config.LineChartColumn("Discount Trend (Q0–Q4)", y_min=0, y_max=50),
            "days_median": st.column_config.NumberColumn("Median Days", format="%d d"),
            "days_range": "Days Range (Min – Max)",
            "days_profile": st.column_config.LineChartColumn("Days Trend (Q0–Q4)", y_min=0),
        },
        column_order=[
            "customer_segment", "product_type", "deals_count", "Win_Rate_Str",
            "price_median", "price_range", "price_profile",
            "disc_median", "disc_range", "disc_profile",
            "days_median", "days_range", "days_profile"
        ],
        hide_index=True,
        width="stretch"
    )

    st.subheader("Visual Distribution & Spread Analysis (Box Plots with Deal Points)")
    box_col1, box_col2, box_col3 = st.columns(3)
    
    with box_col1:
        fig_box_price = px.box(
            filtered_df, x='product_type', y='unit_price_eur', color='customer_segment',
            points='all',
            title="Unit Price Distribution (€)", 
            labels={'unit_price_eur': 'Unit Price (€)', 'product_type': 'Product Type', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_box_price, use_container_width=True)
        
    with box_col2:
        fig_box_disc = px.box(
            filtered_df, x='product_type', y='discount_pct', color='customer_segment',
            points='all',
            title="Discount Distribution (%)", 
            labels={'discount_pct': 'Discount (%)', 'product_type': 'Product Type', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_box_disc, use_container_width=True)
        
    with box_col3:
        fig_box_days = px.box(
            filtered_df, x='product_type', y='days_to_close', color='customer_segment',
            points='all',
            title="Days to Close Distribution", 
            labels={'days_to_close': 'Days to Close', 'product_type': 'Product Type', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_box_days, use_container_width=True)

    st.subheader("Volume & Value Discount Check")
    scat_col1, scat_col2, scat_col3 = st.columns(3)
    
    with scat_col1:
        fig_vol_units = px.scatter(
            filtered_df, x='deal_size_units', y='discount_pct', color='customer_segment',
            trendline='ols', opacity=0.6,
            title="Deal Size (Units) vs Discount",
            labels={'deal_size_units': 'Deal Size (Units)', 'discount_pct': 'Discount (%)', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_vol_units, use_container_width=True)
        
    with scat_col2:
        fig_vol_value = px.scatter(
            filtered_df, x='list_price_eur', y='discount_pct', color='customer_segment',
            trendline='ols', opacity=0.6,
            title="Total Deal Value (€) vs Discount",
            labels={'list_price_eur': 'Total Deal Value (€)', 'discount_pct': 'Discount (%)', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_vol_value, use_container_width=True)
        
    with scat_col3:
        fig_unit_price = px.scatter(
            filtered_df, x='unit_price_eur', y='discount_pct', color='customer_segment',
            trendline='ols', opacity=0.6,
            title="Unit Price (€) vs Discount",
            labels={'unit_price_eur': 'Average Unit Price (€)', 'discount_pct': 'Discount (%)', 'customer_segment': 'Segment'}
        )
        st.plotly_chart(fig_unit_price, use_container_width=True)
        
        # ------------------------------------------------------------------------------
        # SPLIT VIOLIN PLOT (INTERACTIVE PLOTLY)
        # ------------------------------------------------------------------------------
    st.subheader("Discount Distribution: Won vs Lost")

    if not filtered_df.empty:
        df_violin = filtered_df.copy()
    
        # Подсчет количества сделок
        counts = df_violin.groupby(['product_type', 'win_flag']).size().unstack(fill_value=0)
    
        x_map = {
            pt: f"{pt}<br><b>(Lost N={counts.loc[pt, 0]} | Won N={counts.loc[pt, 1]})</b>"
            for pt in df_violin['product_type'].unique() if pt in counts.index
        }    
        df_violin['x_display'] = df_violin['product_type'].map(x_map)

        fig_split = go.Figure()

        # Левая половина: Lost (0)
        fig_split.add_trace(go.Violin(
            x=df_violin['x_display'][df_violin['win_flag'] == 0],
            y=df_violin['discount_pct'][df_violin['win_flag'] == 0],
            legendgroup='Lost (0)',
            scalegroup='discount',
            name='Lost (0)',
            side='negative',
            line_color='#c97a7e',
            fillcolor='#e09b9e',
            meanline_visible=True,
            box_visible=True
        ))

        # Правая половина: Won (1)
        fig_split.add_trace(go.Violin(
            x=df_violin['x_display'][df_violin['win_flag'] == 1],
            y=df_violin['discount_pct'][df_violin['win_flag'] == 1],
            legendgroup='Won (1)',
            scalegroup='discount',
            name='Won (1)',
            side='positive',
            line_color='#4c956c',
            fillcolor='#60ab81',
            meanline_visible=True,
            box_visible=True
        ))

        fig_split.update_layout(
            violinmode='overlay',
            title="Discount Distribution: Split Violin Plot",
            yaxis_title="Discount (%)",
            xaxis_title="Product Type",
            legend_title_text="Deal Outcome (win_flag)"
        )

        st.plotly_chart(fig_split, use_container_width=True)
    else:
        st.warning("No data available for selected filters.")
# ==============================================================================
# TAB 2: WIN RATE & COMPETITION (STATISTICAL ANALYSIS)
# ==============================================================================
with tab2:
    st.header("Win Rate & Competition: Statistical Analysis")
    
    n_selected = len(filtered_df)
    n_won = int(filtered_df['win_flag'].sum())
    n_lost = n_selected - n_won
    overall_win_rate = (n_won / n_selected) if n_selected > 0 else 0
    se_overall = np.sqrt(overall_win_rate * (1 - overall_win_rate) / n_selected) if n_selected > 0 else 0
    
    # Chi-square test for competitor presence
    if len(filtered_df['competitor_present'].unique()) > 1 and len(filtered_df['win_flag'].unique()) > 1:
        ct = pd.crosstab(filtered_df['competitor_present'], filtered_df['win_flag'])
        chi2_stat, p_val, dof, _ = chi2_contingency(ct)
    else:
        chi2_stat, p_val = 0.0, 1.0

    # Key selection metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Deals in Selection (N)", n_selected)
    m_col2.metric("Won / Lost", f"{n_won} / {n_lost}")
    m_col3.metric("Win Rate (95% CI)", f"{overall_win_rate*100:.1f}% (±{1.96*se_overall*100:.1f}%)")
    m_col4.metric("Competitor presence (p-value)", f"p = {p_val:.3f}", delta="Significant" if p_val < 0.05 else "Not Significant")

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Win Rate by Discount Bins and Competitor Presence")
        comp_win = filtered_df.groupby(['discount_bin', 'competitor_present'], observed=False).agg(
            n_deals=('win_flag', 'count'),
            n_won=('win_flag', 'sum'),
            win_rate=('win_flag', 'mean')
        ).reset_index()
        
        comp_win['se'] = np.sqrt(comp_win['win_rate'] * (1 - comp_win['win_rate']) / comp_win['n_deals'])
        comp_win['ci_95'] = 1.96 * comp_win['se']
        
        fig_comp = px.bar(
            comp_win, x='discount_bin', y='win_rate', color='competitor_present', barmode='group',
            text='n_deals', error_y='ci_95',
            title="Win Rate by Discount (Labels = Deal Count N)",
            labels={'discount_bin': 'Discount Bin', 'win_rate': 'Win Rate', 'competitor_present': 'Competitor Present?'}
        )
        fig_comp.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_b:
        st.subheader("Win Rate by Customer Segment")
        seg_win = filtered_df.groupby(['discount_bin', 'customer_segment'], observed=False).agg(
            n_deals=('win_flag', 'count'),
            win_rate=('win_flag', 'mean')
        ).reset_index()
        
        fig_seg = px.line(
            seg_win, x='discount_bin', y='win_rate', color='customer_segment', markers=True,
            title="Win Rate Elasticity by Customer Type",
            labels={'discount_bin': 'Discount Bin', 'win_rate': 'Win Rate', 'customer_segment': 'Segment'}
        )
        fig_seg.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("📋 Detailed Statistics by Discount and Competition")
    
    stat_table = comp_win.copy()
    stat_table['Win_Rate_Pct'] = (stat_table['win_rate'] * 100).round(1).astype(str) + '%'
    stat_table['SE_Pct'] = (stat_table['se'] * 100).round(2).astype(str) + '%'
    stat_table['CI_95_Range'] = (
        ((stat_table['win_rate'] - stat_table['ci_95']).clip(lower=0) * 100).round(1).astype(str) + 
        '% - ' + 
        ((stat_table['win_rate'] + stat_table['ci_95']).clip(upper=1) * 100).round(1).astype(str) + '%'
    )
    
    st.dataframe(
        stat_table[['discount_bin', 'competitor_present', 'n_deals', 'n_won', 'Win_Rate_Pct', 'SE_Pct', 'CI_95_Range']],
        column_config={
            "discount_bin": "Discount Bin",
            "competitor_present": "Competitor?",
            "n_deals": "Total Deals (N)",
            "n_won": "Won",
            "Win_Rate_Pct": "Win Rate (%)",
            "SE_Pct": "Standard Error (SE)",
            "CI_95_Range": "95% Confidence Interval"
        },
        hide_index=True,
        width="stretch"
    )

# ==============================================================================
# TAB 3: DEAL VELOCITY
# ==============================================================================
with tab3:
    st.header("Impact of Discounts on Deal Velocity (Days to Close)")
    st.markdown("""
    * **LATAM (Machine):** Strongest dependency — a 1% discount accelerates signing by **~1.5 days**.
    * **Nordics (Machine):** A 1% discount accelerates signing by **~0.8 days**.
    * **Consumables (All regions):** Closing time is relatively stable (58–63 days) and independent of the discount.
    """)
    
    fig_speed = px.scatter(
        filtered_df, x='discount_pct', y='days_to_close', color='product_type',
        facet_col='region', facet_col_wrap=3, trendline='ols', opacity=0.6,
        title="Days to Close vs Discount (by Region and Product)",
        labels={'discount_pct': 'Discount (%)', 'days_to_close': 'Days to Close', 'product_type': 'Product Type'}
    )
    fig_speed.update_yaxes(matches=None)
    st.plotly_chart(fig_speed, use_container_width=True)
