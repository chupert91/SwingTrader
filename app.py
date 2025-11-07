# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time


# Initialize system
@st.cache_resource
def init_system():
    from data.alpaca_fetcher import AlpacaDataFetcher
    from models.multi_timeframe_predictor import MultiTimeframePredictor
    from monitoring.alert_system import AlertSystem

    fetcher = AlpacaDataFetcher()
    predictor = MultiTimeframePredictor()

    # Load or train model
    try:
        predictor.load_model('models/timeframe_models.pkl')
    except:
        st.warning("No trained model found. Please train the model first.")

    alert_system = AlertSystem(fetcher, predictor)

    return fetcher, predictor, alert_system


fetcher, predictor, alert_system = init_system()

# Streamlit UI
st.title("📈 Stock Analysis & Alert System")

# Sidebar
mode = st.sidebar.selectbox(
    "Mode",
    ["Single Stock Analysis", "Watchlist Monitor", "Alert History", "Model Training"]
)

if mode == "Single Stock Analysis":
    st.header("Stock Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    with col2:
        analyze_button = st.button("Analyze", type="primary")

    if analyze_button:
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch data
            df = fetcher.get_bars(
                ticker,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now()
            )
            df = fetcher.add_technical_indicators(df)

            # Get predictions
            result = predictor.predict(ticker, df)

            # Display results
            st.success(f"Analysis complete for {ticker}")

            # Current price and predictions
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${result.current_price:.2f}")
            with col2:
                st.metric("Trend", result.technical_signals['trend'].upper())
            with col3:
                st.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")

            # Predictions table
            st.subheader("📊 Price Predictions")
            pred_df = pd.DataFrame([
                {
                    'Timeframe': tf,
                    'Predicted Price': f"${price:.2f}",
                    'Expected Return': f"{((price - result.current_price) / result.current_price * 100):.1f}%",
                    'Confidence': f"{conf:.1%}"
                }
                for tf, price in result.predictions.items()
                for conf in [result.confidence_scores[tf]]
            ])
            st.dataframe(pred_df, use_container_width=True)

            # Recommendation
            if "BUY" in result.recommendation:
                st.success(f"💡 {result.recommendation}")
            elif "SELL" in result.recommendation:
                st.error(f"⚠️ {result.recommendation}")
            else:
                st.info(f"📌 {result.recommendation}")

            # Chart
            st.subheader("📈 Price Chart")

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Price', 'RSI', 'Volume')
            )

            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # Add moving averages
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', line=dict(width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', line=dict(width=1)),
                row=1, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # Volume
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['volume'], name='Volume'),
                row=3, col=1
            )

            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

elif mode == "Watchlist Monitor":
    st.header("🔍 Real-Time Watchlist Monitor")

    # Watchlist input
    default_watchlist = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA"
    watchlist_input = st.text_area(
        "Enter tickers (comma-separated)",
        value=default_watchlist
    )
    watchlist = [t.strip().upper() for t in watchlist_input.split(',')]

    col1, col2 = st.columns(2)
    with col1:
        scan_interval = st.selectbox("Scan Interval", ["1 min", "5 min", "15 min", "30 min"])
    with col2:
        start_monitoring = st.button("Start Monitoring", type="primary")

    if start_monitoring:
        alert_system.add_to_watchlist(watchlist)

        # Alert display area
        alert_container = st.empty()
        status_container = st.empty()

        # Continuous monitoring
        while True:
            status_container.info(f"Scanning {len(watchlist)} stocks... {datetime.now().strftime('%H:%M:%S')}")

            # Scan for alerts
            new_alerts = alert_system.scan_for_alerts()

            if new_alerts:
                with alert_container.container():
                    st.success(f"🚨 {len(new_alerts)} new alerts found!")

                    for alert in new_alerts:
                        alert_color = "🟢" if "BUY" in alert.message or "bullish" in alert.message.lower() else "🔴"

                        with st.expander(f"{alert_color} {alert.ticker} - {alert.alert_type.value}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${alert.current_price:.2f}")
                            with col2:
                                st.metric("Target Price", f"${alert.target_price:.2f}")
                            with col3:
                                st.metric("Confidence", f"{alert.confidence:.1%}")

                            st.write(alert.message)
                            st.caption(f"Time: {alert.timestamp.strftime('%H:%M:%S')}")

                            if alert.metadata:
                                st.json(alert.metadata)

            # Wait for next scan
            interval_seconds = {
                "1 min": 60,
                "5 min": 300,
                "15 min": 900,
                "30 min": 1800
            }[scan_interval]

            time.sleep(interval_seconds)

elif mode == "Alert History":
    st.header("📜 Alert History")

    if alert_system.alerts:
        # Convert alerts to DataFrame
        alert_df = pd.DataFrame([
            {
                'Time': alert.timestamp,
                'Ticker': alert.ticker,
                'Type': alert.alert_type.value,
                'Message': alert.message,
                'Current Price': alert.current_price,
                'Target Price': alert.target_price,
                'Confidence': alert.confidence
            }
            for alert in alert_system.alerts
        ])

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_ticker = st.selectbox(
                "Filter by Ticker",
                ["All"] + list(alert_df['Ticker'].unique())
            )
        with col2:
            selected_type = st.selectbox(
                "Filter by Type",
                ["All"] + list(alert_df['Type'].unique())
            )

        # Apply filters
        if selected_ticker != "All":
            alert_df = alert_df[alert_df['Ticker'] == selected_ticker]
        if selected_type != "All":
            alert_df = alert_df[alert_df['Type'] == selected_type]

        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No alerts generated yet. Start monitoring to see alerts.")

elif mode == "Model Training":
    st.header("🤖 Model Training")

    st.info("Train the prediction models on historical data")

    training_tickers = st.text_area(
        "Training Tickers (more diverse = better)",
        value="SPY,QQQ,AAPL,MSFT,GOOGL,PLTR,TSLA,JPM,SOFI,HOOD,WMT,PG"
    )

    if st.button("Start Training", type="primary"):
        tickers = [t.strip() for t in training_tickers.split(',')]

        progress_bar = st.progress(0)
        status_text = st.empty()

        all_data = []
        for i, ticker in enumerate(tickers):
            status_text.text(f"Fetching data for {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))

            df = fetcher.get_bars(
                ticker,
                start_date=datetime.now() - timedelta(days=500),
                end_date=datetime.now()
            )
            if not df.empty:
                df = fetcher.add_technical_indicators(df)
                all_data.append(df)

        if all_data:
            status_text.text("Training models...")
            combined_df = pd.concat(all_data, ignore_index=True)

            predictor.train_models(combined_df)
            predictor.save_models('models/timeframe_models.pkl')

            st.success("✅ Models trained successfully!")
        else:
            st.error("No data fetched. Please check your tickers.")