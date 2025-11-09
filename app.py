# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Stock Prediction & Alert System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize system
@st.cache_resource
def init_system():
    from data.alpaca_fetcher import AlpacaDataFetcher
    from models.multi_timeframe_predictor import MultiTimeframePredictor
    from monitoring.alert_system import AlertSystem

    fetcher = AlpacaDataFetcher()
    predictor = MultiTimeframePredictor()

    # Load or train model
    model_path = 'models/saved/timeframe_models.pkl'
    models_loaded = False
    if os.path.exists(model_path):
        try:
            predictor.load_models(model_path)
            models_loaded = True
        except Exception as e:
            st.error(f"Error loading models: {e}")

    alert_system = AlertSystem(fetcher, predictor)

    return fetcher, predictor, alert_system, models_loaded



fetcher, predictor, alert_system, models_loaded = init_system()

# Streamlit UI
st.title("📈 Stock Analysis & Alert System with Regression Channels")

# Sidebar
mode = st.sidebar.selectbox(
    "Mode",
    ["Single Stock Analysis", "Watchlist Monitor", "Alert History", "Model Training"]
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 14px;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px;
        font-weight: 700;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 14px;
        font-weight: 600;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 16px;
    }
    .buy-signal {
        background-color: #1a472a;
        color: #4ade80;
        border: 2px solid #4ade80;
    }
    .sell-signal {
        background-color: #4a1a1a;
        color: #f87171;
        border: 2px solid #f87171;
    }
</style>
""", unsafe_allow_html=True)


def plot_regression_channels(df, ticker):
    """Create comprehensive chart with regression channels"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            f'{ticker} with Regression Channels (252-day)',
            'SD Position (Standard Deviations from Mean)',
            'RSI',
            'Volume & Channel Width'
        )
    )

    # Main price chart with channels
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )

    # Regression line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['regression_line'],
            name='Regression Line',
            line=dict(color='white', width=2, dash='solid'),
            showlegend=True
        ),
        row=1, col=1
    )

    # SD Channels with gradient coloring
    channel_colors = {
        1: {'upper': 'rgba(255,255,0,0.3)', 'lower': 'rgba(173,216,230,0.3)'},
        2: {'upper': 'rgba(255,165,0,0.3)', 'lower': 'rgba(100,149,237,0.3)'},
        3: {'upper': 'rgba(255,0,0,0.3)', 'lower': 'rgba(0,0,255,0.3)'}
    }

    for sd in [3, 2, 1]:  # Draw from outside in for better layering
        # Upper channel
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[f'upper_{sd}sd'],
                name=f'+{sd}σ',
                line=dict(color=channel_colors[sd]['upper'], width=1, dash='dot'),
                showlegend=True,
                hovertemplate=f'+{sd}σ: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )

        # Lower channel
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[f'lower_{sd}sd'],
                name=f'-{sd}σ',
                line=dict(color=channel_colors[sd]['lower'], width=1, dash='dot'),
                showlegend=True,
                hovertemplate=f'-{sd}σ: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )

        # Fill between channels
        if sd == 1:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
                    y=df[f'upper_{sd}sd'].tolist() + df[f'lower_{sd}sd'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(200,200,200,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

    # Add moving averages for context
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='cyan', width=1),
                showlegend=True
            ),
            row=1, col=1
        )

    # SD Position indicator with color zones
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['sd_position'],
            name='SD Position',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)',
            showlegend=True,
            hovertemplate='SD Position: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add horizontal lines at critical SD levels
    sd_levels = [(-3, 'red', 'Extreme Oversold'),
                 (-2, 'orange', 'Oversold'),
                 (0, 'gray', 'Mean'),
                 (2, 'orange', 'Overbought'),
                 (3, 'red', 'Extreme Overbought')]

    for level, color, label in sd_levels:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label, row=2, col=1)

    # RSI with overbought/oversold zones
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='orange', width=1.5),
                showlegend=True
            ),
            row=3, col=1
        )

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    # Volume bars with color based on price action
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green'
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=True,
            opacity=0.5
        ),
        row=4, col=1
    )

    # Channel width overlay (secondary y-axis effect)
    if 'channel_width' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['channel_width'] * df['volume'].max() / df['channel_width'].max(),
                name='Channel Width (scaled)',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=True,
                yaxis='y2'
            ),
            row=4, col=1
        )

    # Update layout
    fig.update_layout(
        height=1000,
        title={
            'text': f"{ticker} - Regression Channel Analysis",
            'font': {'size': 20}
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        hovermode='x unified',
        margin=dict(r=150)  # Make room for legend
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    return fig


def display_regression_metrics(df):
    """Display key regression channel metrics"""
    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sd_pos = latest['sd_position'] if 'sd_position' in df.columns else 0
        color = "🔴" if abs(sd_pos) >= 2 else "🟡" if abs(sd_pos) >= 1 else "🟢"
        st.metric(
            "SD Position",
            f"{sd_pos:.2f}σ {color}",
            f"{'Overbought' if sd_pos > 2 else 'Oversold' if sd_pos < -2 else 'Normal'}"
        )

    with col2:
        if 'regression_line' in df.columns:
            reg_line = latest['regression_line']
            current_price = latest['close']
            pct_from_reg = ((current_price - reg_line) / reg_line) * 100
            st.metric(
                "Regression Line",
                f"${reg_line:.2f}",
                f"{pct_from_reg:+.1f}% from mean"
            )

    with col3:
        if 'regression_slope' in df.columns:
            daily_trend = latest['regression_slope']
            annual_trend = daily_trend * 252 / latest['close'] * 100
            st.metric(
                "Trend (Annual)",
                f"{annual_trend:+.1f}%",
                f"R² = {latest['regression_r2']:.3f}"
            )

    with col4:
        if 'channel_width' in df.columns:
            channel_pct = (latest['channel_width'] / latest['close']) * 100
            st.metric(
                "Channel Width",
                f"{channel_pct:.1f}%",
                "Volatility measure"
            )


if mode == "Single Stock Analysis":
    st.header("Stock Price Prediction with Regression Channels")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    with col2:
        days_back = st.number_input("Days of History", min_value=100, max_value=500, value=365)
    with col3:
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_button or ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch data
            df = fetcher.get_bars(
                ticker,
                start_date=datetime.now() - timedelta(days=days_back),
                end_date=datetime.now()
            )

            if df.empty:
                st.error(f"❌ No data found for {ticker}")
            else:
                # Add all indicators including regression channels
                df = fetcher.add_technical_indicators(df)

                # Get predictions
                if 'regression_line' in df.columns:
                    result = predictor.predict(ticker, df)

                    st.success(f"✅ Analysis complete for {ticker}")

                    # Display regression metrics
                    st.subheader("📊 Regression Channel Metrics")
                    display_regression_metrics(df)

                    # Price predictions
                    st.subheader("🎯 Price Predictions")
                    pred_cols = st.columns(len(result.predictions))
                    for i, (timeframe, price) in enumerate(result.predictions.items()):
                        with pred_cols[i]:
                            expected_return = ((price - result.current_price) / result.current_price) * 100
                            confidence = result.confidence_scores[timeframe]

                            # Color code based on return
                            if expected_return > 5:
                                color = "🟢"
                            elif expected_return < -5:
                                color = "🔴"
                            else:
                                color = "🟡"

                            st.metric(
                                timeframe.replace('_', ' ').title(),
                                f"${price:.2f}",
                                f"{expected_return:+.1f}% {color}",
                                help=f"Confidence: {confidence:.1%}"
                            )

                    # Trading signals
                    st.subheader("📈 Trading Signals")
                    signal_col1, signal_col2 = st.columns(2)

                    with signal_col1:
                        # Technical signals
                        st.write("**Technical Indicators:**")
                        for key, value in result.technical_signals.items():
                            if key != 'support_level' and key != 'resistance_level':
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")

                    with signal_col2:
                        # Support/Resistance
                        st.write("**Key Levels:**")
                        if 'support_level' in result.technical_signals:
                            st.write(f"• Support: ${result.technical_signals['support_level']:.2f}")
                        if 'resistance_level' in result.technical_signals:
                            st.write(f"• Resistance: ${result.technical_signals['resistance_level']:.2f}")
                        if 'upper_2sd' in df.columns:
                            st.write(f"• +2σ Level: ${df['upper_2sd'].iloc[-1]:.2f}")
                            st.write(f"• -2σ Level: ${df['lower_2sd'].iloc[-1]:.2f}")

                    # Recommendation box
                    rec_style = "buy-signal" if "BUY" in result.recommendation else "sell-signal" if "SELL" in result.recommendation else ""
                    st.markdown(f"""
                    <div class="alert-box {rec_style}">
                        <h3>💡 Recommendation</h3>
                        <p><strong>{result.recommendation}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Risk metrics
                    with st.expander("📉 Risk Metrics", expanded=False):
                        risk_col1, risk_col2, risk_col3 = st.columns(3)
                        with risk_col1:
                            st.metric("Historical Volatility",
                                      f"{result.risk_metrics['historical_volatility']:.1%}")
                        with risk_col2:
                            st.metric("Max Drawdown (3M)",
                                      f"{result.risk_metrics['max_drawdown_3m']:.1%}")
                        with risk_col3:
                            if 'sharpe_estimate' in result.risk_metrics:
                                st.metric("Sharpe Estimate",
                                          f"{result.risk_metrics['sharpe_estimate']:.2f}")

                    # Chart with regression channels
                    st.subheader("📊 Interactive Chart")
                    fig = plot_regression_channels(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    # Additional analysis
                    with st.expander("📋 Detailed Statistics", expanded=False):
                        stats_col1, stats_col2 = st.columns(2)

                        with stats_col1:
                            st.write("**Price Statistics:**")
                            st.write(f"• Current Price: ${df['close'].iloc[-1]:.2f}")
                            st.write(f"• 20-Day Avg: ${df['sma_20'].iloc[-1]:.2f}")
                            st.write(f"• 50-Day Avg: ${df['sma_50'].iloc[-1]:.2f}")
                            st.write(f"• 52W High: ${df['high'].tail(252).max():.2f}")
                            st.write(f"• 52W Low: ${df['low'].tail(252).min():.2f}")

                        with stats_col2:
                            st.write("**Regression Statistics:**")
                            if 'sd_position' in df.columns:
                                st.write(
                                    f"• Days at current regime: {int(df['days_above_2sd'].iloc[-1] or df['days_below_2sd'].iloc[-1])}")
                                st.write(f"• 3σ touches (20d): {int(df['touches_3sd_20d'].iloc[-1])}")
                                st.write(
                                    f"• Channel trend: {'Expanding' if df['channel_width_expanding'].iloc[-1] else 'Contracting'}")

                                # Historical SD distribution
                                sd_positions = df['sd_position'].dropna()
                                st.write(f"• Avg SD Position: {sd_positions.mean():.2f}")
                                st.write(f"• Time beyond ±2σ: {(abs(sd_positions) > 2).mean():.1%}")

                else:
                    st.warning("Regression channels could not be calculated. Need more historical data.")

elif mode == "Watchlist Monitor":
    st.header("🔍 Real-Time Watchlist Monitor")

    # Watchlist input
    default_watchlist = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,PLTR,SOFI"
    watchlist_input = st.text_area(
        "Enter tickers (comma-separated)",
        value=default_watchlist,
        help="Enter stock symbols separated by commas"
    )
    watchlist = [t.strip().upper() for t in watchlist_input.split(',')]

    col1, col2, col3 = st.columns(3)
    with col1:
        scan_interval = st.selectbox("Scan Interval", ["1 min", "5 min", "15 min", "30 min"])
    with col2:
        min_confidence = st.slider("Min Alert Confidence", 0.5, 0.9, 0.6, 0.05)
    with col3:
        start_monitoring = st.button("Start Monitoring", type="primary", use_container_width=True)

    # Alert configuration
    with st.expander("⚙️ Alert Settings", expanded=False):
        alert_col1, alert_col2 = st.columns(2)
        with alert_col1:
            st.checkbox("3-Sigma Touches", value=True, key="alert_3sigma")
            st.checkbox("2-Sigma Reversals", value=True, key="alert_2sigma")
            st.checkbox("Regression Crosses", value=True, key="alert_regression")
        with alert_col2:
            st.checkbox("RSI Extremes", value=True, key="alert_rsi")
            st.checkbox("Volume Spikes", value=True, key="alert_volume")
            st.checkbox("ML Predictions", value=True, key="alert_ml")

    if start_monitoring:
        alert_system.add_to_watchlist(watchlist)
        alert_system.alert_conditions['min_confidence'] = min_confidence

        # Create placeholders for dynamic updates
        status_placeholder = st.empty()
        alert_container = st.container()
        metrics_placeholder = st.empty()

        # Monitoring loop
        try:
            while True:
                current_time = datetime.now().strftime('%H:%M:%S')
                status_placeholder.info(f"⏰ Scanning {len(watchlist)} stocks at {current_time}...")

                # Scan for alerts
                new_alerts = alert_system.scan_for_alerts()

                # Display alerts
                if new_alerts:
                    with alert_container:
                        st.success(f"🚨 {len(new_alerts)} new alerts found at {current_time}!")

                        # Group alerts by priority
                        high_priority = [a for a in new_alerts if a.confidence >= 0.75]
                        med_priority = [a for a in new_alerts if 0.6 <= a.confidence < 0.75]
                        low_priority = [a for a in new_alerts if a.confidence < 0.6]

                        # Display high priority alerts
                        if high_priority:
                            st.subheader("🔴 High Priority Alerts")
                            for alert in high_priority:
                                col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
                                with col1:
                                    st.write(f"**{alert.ticker}** - {alert.alert_type.value}")
                                with col2:
                                    st.write(alert.message)
                                with col3:
                                    st.write(f"Target: ${alert.target_price:.2f}")
                                with col4:
                                    st.write(f"Conf: {alert.confidence:.0%}")

                        # Display medium priority alerts
                        if med_priority:
                            st.subheader("🟡 Medium Priority Alerts")
                            for alert in med_priority:
                                with st.expander(f"{alert.ticker} - {alert.alert_type.value}"):
                                    st.write(alert.message)
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Current", f"${alert.current_price:.2f}")
                                        st.metric("Target", f"${alert.target_price:.2f}")
                                    with col2:
                                        st.metric("Confidence", f"{alert.confidence:.0%}")
                                        if alert.metadata:
                                            for key, value in alert.metadata.items():
                                                st.write(f"• {key}: {value}")

                # Update metrics summary
                with metrics_placeholder.container():
                    st.subheader("📊 Watchlist Summary")
                    summary_cols = st.columns(len(watchlist[:5]))  # Show first 5

                    for i, symbol in enumerate(watchlist[:5]):
                        with summary_cols[i]:
                            try:
                                # Get latest data for symbol
                                df_summary = fetcher.get_bars(
                                    symbol,
                                    start_date=datetime.now() - timedelta(days=2),
                                    end_date=datetime.now()
                                )
                                if not df_summary.empty:
                                    latest_price = df_summary['close'].iloc[-1]
                                    change = df_summary['close'].pct_change().iloc[-1] * 100
                                    color = "🟢" if change > 0 else "🔴"
                                    st.metric(symbol, f"${latest_price:.2f}", f"{change:+.1f}% {color}")
                            except:
                                pass

                # Wait for next scan
                interval_seconds = {
                    "1 min": 60,
                    "5 min": 300,
                    "15 min": 900,
                    "30 min": 1800
                }[scan_interval]

                for i in range(interval_seconds):
                    status_placeholder.info(f"Next scan in {interval_seconds - i} seconds...")
                    time.sleep(1)

        except KeyboardInterrupt:
            st.info("Monitoring stopped by user")

elif mode == "Alert History":
    st.header("📜 Alert History & Analysis")

    if alert_system.alerts:
        # Convert alerts to DataFrame for analysis
        alert_df = pd.DataFrame([
            {
                'Time': alert.timestamp,
                'Ticker': alert.ticker,
                'Type': alert.alert_type.value,
                'Message': alert.message,
                'Current Price': f"${alert.current_price:.2f}",
                'Target Price': f"${alert.target_price:.2f}",
                'Confidence': f"{alert.confidence:.0%}",
                'SD Position': alert.metadata.get('sd_position', 'N/A') if alert.metadata else 'N/A'
            }
            for alert in alert_system.alerts
        ])

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", len(alert_df))
        with col2:
            st.metric("Unique Tickers", alert_df['Ticker'].nunique())
        with col3:
            st.metric("Most Common Alert", alert_df['Type'].mode()[0] if not alert_df.empty else "N/A")
        with col4:
            avg_conf = alert_df['Confidence'].str.rstrip('%').astype(float).mean()
            st.metric("Avg Confidence", f"{avg_conf:.0f}%")

        # Filters
        st.subheader("🔍 Filter Alerts")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            selected_ticker = st.selectbox(
                "Filter by Ticker",
                ["All"] + sorted(alert_df['Ticker'].unique().tolist())
            )

        with filter_col2:
            selected_type = st.selectbox(
                "Filter by Type",
                ["All"] + sorted(alert_df['Type'].unique().tolist())
            )

        with filter_col3:
            date_range = st.date_input(
                "Date Range",
                value=(alert_df['Time'].min(), alert_df['Time'].max()),
                max_value=datetime.now()
            )

        # Apply filters
        filtered_df = alert_df.copy()
        if selected_ticker != "All":
            filtered_df = filtered_df[filtered_df['Ticker'] == selected_ticker]
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df['Type'] == selected_type]
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Time'].dt.date >= date_range[0]) &
                (filtered_df['Time'].dt.date <= date_range[1])
                ]

        # Display filtered alerts
        st.subheader(f"📋 Alert Details ({len(filtered_df)} alerts)")
        st.dataframe(
            filtered_df.sort_values('Time', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Alert type distribution
        if not filtered_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Alert Type Distribution")
                type_counts = filtered_df['Type'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=0.3
                )])
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("📈 Alerts Over Time")
                time_counts = filtered_df.groupby(filtered_df['Time'].dt.date).size()
                fig_time = go.Figure(data=[go.Bar(
                    x=time_counts.index,
                    y=time_counts.values
                )])
                fig_time.update_layout(height=300)
                st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No alerts generated yet. Start monitoring to see alerts here.")

elif mode == "Model Training":
    st.header("🤖 Model Training & Configuration")

    st.info("""
    Train the prediction models on historical data. The model will learn from regression channels,
    technical indicators, and price patterns to make predictions.
    """)

    # Training configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Symbols")
        default_symbols = """PLTR,HOOD,SOFI,MP,IONQ,AMD,NVDA,TSLA,AAPL,SPY,QQQ,MSFT,GOOGL,JPM,XOM,JNJ,GLD,TLT"""
        training_tickers = st.text_area(
            "Enter training symbols (comma-separated)",
            value=default_symbols,
            height=150,
            help="Include diverse symbols for better model generalization"
        )

    with col2:
        st.subheader("Training Parameters")
        lookback_days = st.number_input(
            "Historical Days for Training",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="More data generally improves model quality"
        )

        regression_period = st.number_input(
            "Regression Channel Period",
            min_value=100,
            max_value=1000,
            value=400,
            help="252 trading days = 1 year"
        )

        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "random_forest", "ensemble"],
            help="XGBoost typically performs best"
        )

    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in training_tickers.split(',')]

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()

        all_data = []
        errors = []

        # Fetch data for each symbol
        for i, ticker in enumerate(tickers):
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"Fetching data for {ticker}... ({i + 1}/{len(tickers)})")

            try:
                df = fetcher.get_bars(
                    ticker,
                    start_date=datetime.now() - timedelta(days=lookback_days),
                    end_date=datetime.now()
                )

                if not df.empty and len(df) >= regression_period:
                    # Add indicators including regression channels
                    df = fetcher.add_technical_indicators(df)
                    all_data.append(df)

                    with log_container:
                        st.success(f"✅ {ticker}: {len(df)} days fetched")
                else:
                    with log_container:
                        st.warning(f"⚠️ {ticker}: Insufficient data ({len(df)} days)")

            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                with log_container:
                    st.error(f"❌ {ticker}: {str(e)}")

        if all_data:
            status_text.text("Combining data and training models...")

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            with log_container:
                st.info(f"📊 Total training samples: {len(combined_df):,} from {len(all_data)} symbols")

            # Train models
            try:
                os.makedirs('models/saved', exist_ok=True)

                # Initialize and train predictor
                predictor.train_models(combined_df)

                # Save models
                save_path = 'models/saved/timeframe_models.pkl'
                predictor.save_models(save_path)

                st.success(f"""
                ✅ **Training Complete!**
                - Models saved to: {save_path}
                - Symbols used: {len(all_data)}
                - Total samples: {len(combined_df):,}
                - Model type: {model_type}

                You can now use the trained model for predictions!
                """)

                # Display training summary
                with st.expander("📊 Training Summary", expanded=True):
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.write("**Successfully Trained On:**")
                        for ticker in [t for t in tickers if t not in [e.split(':')[0] for e in errors]]:
                            st.write(f"• {ticker}")

                    with summary_col2:
                        if errors:
                            st.write("**Failed Symbols:**")
                            for error in errors:
                                st.write(f"• {error}")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
        else:
            st.error("❌ No valid data fetched. Please check your symbols and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Stock Prediction & Alert System with Regression Channels | 
    Using 252-day Linear Regression with Residual-Based Standard Deviations
</div>
""", unsafe_allow_html=True)