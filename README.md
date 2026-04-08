Cryptocurrency Price Predictor
Overview
The Cryptocurrency Price Predictor is a Flask-based API that uses a Long Short-Term Memory (LSTM) neural network to predict cryptocurrency prices. It fetches historical data using Yahoo Finance, preprocesses it, trains the LSTM model, and serves predictions through an API. Additionally, a simple web interface allows users to visualize the predicted prices with Chart.js.
Features
- LSTM Model for time-series prediction of cryptocurrency prices.
- Flask API to fetch predictions programmatically.
- Web Interface for user interaction and data visualization.
- CORS Enabled to support requests from any origin.
- Historical data fetched using yFinance.

Technologies Used
- Python (Backend)- Flask (API Framework)
- Flask-CORS (Enable CORS)
- NumPy & Pandas (Data Manipulation)
- yFinance (Data Fetching)
- Keras (Machine Learning - LSTM)
- scikit-learn (Data Scaling)

- HTML & JavaScript (Frontend)- Chart.js (Visualization)

- Git (Version Control)


## AURA-X PRIME (Phase 1 Implemented)

This repository now includes a Phase 1 Rust ingestion core at:

- `./aura_x_prime_ingestion`

### What it includes

- Tokio async runtime based ingestion engine
- Parallel source ingestion with task fan-out (`tokio::task::JoinSet`)
- Source catalog sized to 100+ logical sources (stocks, crypto, news, social, macro)
- WebSocket ingestion via `tokio-tungstenite`
- HTTP/HTML ingestion via `reqwest` + `scraper`
- Retry with exponential backoff
- Per-source lightweight rate limiting
- Header rotation using rotating User-Agent headers

### Run

```bash
cd aura_x_prime_ingestion
cargo run
```

> Note: some external endpoints can rate-limit or require API keys; failures are logged per source and do not crash the full ingestion batch.

## AURA-X PRIME (Phase 2 Implemented)

The Rust module now includes a lightweight stream-processing pipeline using Tokio channels:

- `ingestion -> processing -> storage`
- Processing stage normalizes records into a unified schema
- Storage stage serializes normalized events with Protobuf binary encoding

### Phase 2 output

- Output file: `aura_x_prime_ingestion/output/aura_stream.pb`
- Format: length-delimited Protobuf event stream

### Optional environment variables

- `FRED_API_KEY` (optional): used for FRED macro API requests when provided

## AURA-X PRIME (Phase 3 Implemented)

A market feature engine now runs after normalization to generate compact candle/indicator records.

### Phase 3 pipeline

- `ingestion -> processing -> stream storage`
- `processing -> market engine -> market feature storage`
- Market engine computes per-source:
  - Candle-like fields (`open`, `high`, `low`, `close`, `volume`)
  - Indicators (`ema_fast`, `ema_slow`, `rsi14`, `momentum`)

### Phase 3 output

- Output file: `aura_x_prime_ingestion/output/aura_market_features.pb`
- Format: length-delimited Protobuf market feature stream
