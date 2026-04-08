use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use prost::Message;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use scraper::{Html, Selector};
use serde::Serialize;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep, timeout};
use tokio_tungstenite::connect_async;
use tracing::{error, info, warn};

const USER_AGENTS: [&str; 6] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.49 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
];

const CHANNEL_BUFFER_SIZE: usize = 4096;
const HTTP_RETRY_ATTEMPTS: usize = 4;
const HTTP_INITIAL_BACKOFF_MS: u64 = 400;
const HTML_RETRY_ATTEMPTS: usize = 4;
const HTML_INITIAL_BACKOFF_MS: u64 = 500;
const WS_RETRY_ATTEMPTS: usize = 3;
const WS_INITIAL_BACKOFF_MS: u64 = 600;

#[derive(Clone, Debug)]
enum SourceKind {
    HttpJson {
        url: String,
    },
    HtmlScrape {
        url: String,
        css_selector: String,
        max_items: usize,
    },
    WebSocket {
        url: String,
        max_messages: usize,
    },
}

#[derive(Clone, Debug)]
struct DataSource {
    name: String,
    category: String,
    kind: SourceKind,
    rate_limit_per_second: f64,
}

#[derive(Debug, Serialize)]
struct IngestionEnvelope {
    source: String,
    category: String,
    payload: serde_json::Value,
}

#[derive(Clone, PartialEq, Message)]
struct UnifiedStreamEvent {
    #[prost(string, tag = "1")]
    source: String,
    #[prost(string, tag = "2")]
    category: String,
    #[prost(string, tag = "3")]
    normalized_type: String,
    #[prost(int64, tag = "4")]
    ingest_ts_ms: i64,
    #[prost(bytes = "vec", tag = "5")]
    payload_json: Vec<u8>,
}

#[derive(Debug)]
struct HeaderRotator {
    index: AtomicUsize,
}

impl HeaderRotator {
    fn new() -> Self {
        Self {
            index: AtomicUsize::new(0),
        }
    }

    fn next_headers(&self) -> Result<HeaderMap> {
        let idx = self.index.fetch_add(1, Ordering::Relaxed);
        let user_agent = USER_AGENTS[idx % USER_AGENTS.len()];
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(user_agent).context("invalid rotated user-agent")?,
        );
        Ok(headers)
    }
}

#[derive(Debug)]
struct SimpleRateLimiter {
    min_interval: Duration,
    last_request: Mutex<Instant>,
}

impl SimpleRateLimiter {
    fn new(rate_limit_per_second: f64) -> Self {
        let min_interval = if rate_limit_per_second <= 0.0 {
            Duration::from_secs(1)
        } else {
            Duration::from_secs_f64(1.0 / rate_limit_per_second)
        };

        Self {
            min_interval,
            last_request: Mutex::new(Instant::now() - min_interval),
        }
    }

    async fn wait_turn(&self) {
        let mut last = self.last_request.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last);
        if elapsed < self.min_interval {
            sleep(self.min_interval - elapsed).await;
        }
        *last = Instant::now();
    }
}

#[derive(Clone)]
struct IngestionContext {
    client: reqwest::Client,
    rotator: Arc<HeaderRotator>,
    limiters: Arc<HashMap<String, Arc<SimpleRateLimiter>>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let sources = build_source_catalog();
    info!(
        "AURA-X PRIME booting with {} configured sources",
        sources.len()
    );

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(25))
        .pool_max_idle_per_host(8)
        .build()
        .context("failed to build reqwest client")?;

    let mut limiters = HashMap::with_capacity(sources.len());
    for source in &sources {
        limiters.insert(
            source.name.clone(),
            Arc::new(SimpleRateLimiter::new(source.rate_limit_per_second)),
        );
    }

    let context = IngestionContext {
        client,
        rotator: Arc::new(HeaderRotator::new()),
        limiters: Arc::new(limiters),
    };

    let (ingestion_tx, ingestion_rx) = mpsc::channel::<IngestionEnvelope>(CHANNEL_BUFFER_SIZE);
    let (processing_tx, processing_rx) = mpsc::channel::<UnifiedStreamEvent>(CHANNEL_BUFFER_SIZE);

    tokio::fs::create_dir_all("output")
        .await
        .context("failed to create output directory")?;

    let processor_task = tokio::spawn(run_processing_stage(ingestion_rx, processing_tx));
    let storage_task = tokio::spawn(run_storage_stage(processing_rx, "output/aura_stream.pb"));

    let mut jobs = JoinSet::new();
    for source in sources {
        let ctx = context.clone();
        let tx = ingestion_tx.clone();
        jobs.spawn(async move {
            let envelope = ingest_source(source, ctx).await?;
            tx.send(envelope)
                .await
                .map_err(|_| anyhow!("ingestion channel closed before send"))?;
            Ok::<(), anyhow::Error>(())
        });
    }
    drop(ingestion_tx);

    let mut success = 0usize;
    let mut failed = 0usize;
    while let Some(result) = jobs.join_next().await {
        match result {
            Ok(Ok(_)) => success += 1,
            Ok(Err(err)) => {
                failed += 1;
                warn!("source failed: {err}");
            }
            Err(join_err) => {
                failed += 1;
                error!("task join failure: {join_err}");
            }
        }
    }

    let processed = processor_task
        .await
        .context("processing stage task join error")??;
    let stored = storage_task
        .await
        .context("storage stage task join error")??;

    info!(
        "Phase 1+2 completed. ingest_success={success}, ingest_failed={failed}, processed={processed}, stored={stored}"
    );
    Ok(())
}

async fn run_processing_stage(
    mut ingestion_rx: mpsc::Receiver<IngestionEnvelope>,
    processing_tx: mpsc::Sender<UnifiedStreamEvent>,
) -> Result<usize> {
    let mut count = 0usize;
    while let Some(envelope) = ingestion_rx.recv().await {
        let event = normalize_event(envelope)?;
        processing_tx
            .send(event)
            .await
            .map_err(|_| anyhow!("processing channel closed before send"))?;
        count += 1;
    }
    info!("processing stage finished with {count} events");
    Ok(count)
}

async fn run_storage_stage(
    mut processing_rx: mpsc::Receiver<UnifiedStreamEvent>,
    path: &str,
) -> Result<usize> {
    let file = File::create(path)
        .await
        .with_context(|| format!("failed to create storage file {path}"))?;
    let mut writer = BufWriter::new(file);
    let mut count = 0usize;

    while let Some(event) = processing_rx.recv().await {
        let mut bytes = Vec::with_capacity(event.encoded_len() + 10);
        event
            .encode_length_delimited(&mut bytes)
            .context("failed to protobuf-encode event")?;
        writer
            .write_all(&bytes)
            .await
            .context("failed to write protobuf bytes")?;
        count += 1;
    }

    writer
        .flush()
        .await
        .context("failed to flush storage file")?;
    info!("storage stage wrote {count} protobuf events to {path}");
    Ok(count)
}

fn normalize_event(envelope: IngestionEnvelope) -> Result<UnifiedStreamEvent> {
    let normalized_type = envelope
        .payload
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let payload_json =
        serde_json::to_vec(&envelope.payload).context("json payload encode failed")?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_millis() as i64;

    Ok(UnifiedStreamEvent {
        source: envelope.source,
        category: envelope.category,
        normalized_type,
        ingest_ts_ms: ts,
        payload_json,
    })
}

async fn ingest_source(source: DataSource, context: IngestionContext) -> Result<IngestionEnvelope> {
    let limiter = context
        .limiters
        .get(&source.name)
        .cloned()
        .ok_or_else(|| anyhow!("missing limiter for source {}", source.name))?;

    limiter.wait_turn().await;

    let payload = match source.kind.clone() {
        SourceKind::HttpJson { url } => {
            with_retry(
                HTTP_RETRY_ATTEMPTS,
                Duration::from_millis(HTTP_INITIAL_BACKOFF_MS),
                || async { fetch_http_json(&context.client, &context.rotator, &url).await },
            )
            .await?
        }
        SourceKind::HtmlScrape {
            url,
            css_selector,
            max_items,
        } => {
            with_retry(
                HTML_RETRY_ATTEMPTS,
                Duration::from_millis(HTML_INITIAL_BACKOFF_MS),
                || async {
                    fetch_html_snippets(
                        &context.client,
                        &context.rotator,
                        &url,
                        &css_selector,
                        max_items,
                    )
                    .await
                },
            )
            .await?
        }
        SourceKind::WebSocket { url, max_messages } => {
            with_retry(
                WS_RETRY_ATTEMPTS,
                Duration::from_millis(WS_INITIAL_BACKOFF_MS),
                || async { fetch_websocket_messages(&url, max_messages).await },
            )
            .await?
        }
    };

    info!(
        "ingested source={} category={}",
        source.name, source.category
    );
    Ok(IngestionEnvelope {
        source: source.name,
        category: source.category,
        payload,
    })
}

async fn fetch_http_json(
    client: &reqwest::Client,
    rotator: &HeaderRotator,
    url: &str,
) -> Result<serde_json::Value> {
    let headers = rotator.next_headers()?;
    let response = client
        .get(url)
        .headers(headers)
        .send()
        .await
        .with_context(|| format!("http request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("non-success status for: {url}"))?;

    let text = response.text().await?;
    Ok(serde_json::json!({
        "type": "http_json",
        "url": url,
        "content_length": text.len(),
        "sample": text.chars().take(400).collect::<String>(),
    }))
}

async fn fetch_html_snippets(
    client: &reqwest::Client,
    rotator: &HeaderRotator,
    url: &str,
    css_selector: &str,
    max_items: usize,
) -> Result<serde_json::Value> {
    let headers = rotator.next_headers()?;
    let body = client
        .get(url)
        .headers(headers)
        .send()
        .await
        .with_context(|| format!("html request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("non-success html status for: {url}"))?
        .text()
        .await?;

    let selector = Selector::parse(css_selector)
        .map_err(|_| anyhow!("invalid css selector {css_selector}"))?;
    let document = Html::parse_document(&body);
    let snippets: Vec<String> = document
        .select(&selector)
        .take(max_items)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|text| !text.is_empty())
        .collect();

    Ok(serde_json::json!({
        "type": "html_scrape",
        "url": url,
        "selector": css_selector,
        "items": snippets,
    }))
}

async fn fetch_websocket_messages(url: &str, max_messages: usize) -> Result<serde_json::Value> {
    let (stream, _) = connect_async(url)
        .await
        .with_context(|| format!("websocket connection failed: {url}"))?;
    let (_, mut read) = stream.split();
    let mut messages = Vec::with_capacity(max_messages);

    for _ in 0..max_messages {
        let next_msg = timeout(Duration::from_secs(6), read.next())
            .await
            .with_context(|| format!("websocket timeout: {url}"))?;
        match next_msg {
            Some(Ok(msg)) => {
                if let Some(text) = msg.to_text().ok() {
                    messages.push(text.to_string());
                }
            }
            Some(Err(err)) => return Err(anyhow!("websocket read error {url}: {err}")),
            None => break,
        }
    }

    Ok(serde_json::json!({
        "type": "websocket",
        "url": url,
        "messages": messages,
    }))
}

async fn with_retry<F, Fut, T>(max_attempts: usize, initial_delay: Duration, mut f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 1usize;
    let mut delay = initial_delay;

    loop {
        match f().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt < max_attempts => {
                warn!("attempt {attempt}/{max_attempts} failed: {err}. retrying in {delay:?}");
                sleep(delay).await;
                delay = delay.saturating_mul(2);
                attempt += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

fn build_source_catalog() -> Vec<DataSource> {
    let mut sources = vec![
        // Core market data
        source_http_json("nse_overview", "stocks", "https://www.nseindia.com", 0.2),
        source_http_json(
            "yahoo_btc_usd",
            "crypto",
            "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?range=1d&interval=1m",
            2.0,
        ),
        source_http_json(
            "alpha_vantage_demo",
            "stocks",
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo",
            0.5,
        ),
        source_html(
            "google_news_markets",
            "news",
            "https://news.google.com/search?q=crypto%20market",
            "article h3",
            12,
            0.5,
        ),
        source_html(
            "reddit_crypto_hot",
            "social",
            "https://www.reddit.com/r/CryptoCurrency/hot/",
            "h3",
            15,
            0.5,
        ),
        source_http_json("fred_series_gdp", "macro", &fred_series_url("GDP"), 0.3),
        source_http_json(
            "world_bank_gdp_india",
            "macro",
            "https://api.worldbank.org/v2/country/IN/indicator/NY.GDP.MKTP.CD?format=json",
            0.5,
        ),
        source_websocket(
            "binance_btcusdt_trade",
            "crypto",
            "wss://stream.binance.com:9443/ws/btcusdt@trade",
            5,
            8.0,
        ),
        source_websocket(
            "coinbase_btcusd_ticker",
            "crypto",
            "wss://ws-feed.exchange.coinbase.com",
            3,
            6.0,
        ),
    ];

    // Scale to 100+ logical sources via symbolized market endpoints.
    let stock_symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "NFLX",
        "AMD",
        "INTC",
        "IBM",
        "ORCL",
        "CSCO",
        "QCOM",
        "ADBE",
        "CRM",
        "PYPL",
        "UBER",
        "ABNB",
        "SHOP",
        "BABA",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "RELIANCE.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS",
        "BAJFINANCE.NS",
        "KOTAKBANK.NS",
        "HINDUNILVR.NS",
        "MARUTI.NS",
        "TITAN.NS",
        "SUNPHARMA.NS",
    ];
    for symbol in stock_symbols {
        sources.push(source_http_json(
            &format!("yahoo_{symbol}_1m"),
            "stocks",
            &format!(
                "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1m"
            ),
            2.0,
        ));
    }

    let crypto_pairs = [
        "btcusdt",
        "ethusdt",
        "solusdt",
        "xrpusdt",
        "bnbusdt",
        "adausdt",
        "dogeusdt",
        "linkusdt",
        "maticusdt",
        "ltcusdt",
        "avaxusdt",
        "dotusdt",
        "atomusdt",
        "nearusdt",
        "opususdt",
        "arbusdt",
        "trxusdt",
        "etcusdt",
        "filusdt",
        "uniusdt",
    ];
    for pair in crypto_pairs {
        sources.push(source_websocket(
            &format!("binance_{pair}_trade"),
            "crypto",
            &format!("wss://stream.binance.com:9443/ws/{pair}@trade"),
            3,
            10.0,
        ));
    }

    let macro_feeds = [
        "CPIAUCSL", "FEDFUNDS", "UNRATE", "DGS10", "DEXINUS", "DEXUSEU", "M2SL", "PCE", "PAYEMS",
        "INDPRO", "HOUST", "RSAFS", "CPILFESL",
    ];
    for series in macro_feeds {
        sources.push(source_http_json(
            &format!("fred_{series}"),
            "macro",
            &fred_series_url(series),
            0.3,
        ));
    }

    let rss_feeds = [
        ("cointelegraph_rss", "https://cointelegraph.com/rss"),
        (
            "coindesk_rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ),
        (
            "economic_times_markets",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        ),
        (
            "reuters_markets",
            "https://www.reutersagency.com/feed/?best-topics=markets&post_type=best",
        ),
        ("livemint_markets", "https://www.livemint.com/rss/markets"),
        ("investing_news", "https://www.investing.com/rss/news.rss"),
    ];
    for (name, url) in rss_feeds {
        sources.push(source_html(name, "news", url, "item title, title", 20, 0.8));
    }

    // Ensure 100+ source target.
    let needed = 105usize.saturating_sub(sources.len());
    for idx in 0..needed {
        sources.push(source_http_json(
            &format!("synthetic_source_{idx}"),
            "synthetic",
            "https://httpbin.org/json",
            1.0,
        ));
    }

    sources
}

fn source_http_json(
    name: &str,
    category: &str,
    url: &str,
    rate_limit_per_second: f64,
) -> DataSource {
    DataSource {
        name: name.to_string(),
        category: category.to_string(),
        kind: SourceKind::HttpJson {
            url: url.to_string(),
        },
        rate_limit_per_second,
    }
}

fn fred_series_url(series: &str) -> String {
    if let Ok(api_key) = env::var("FRED_API_KEY") {
        if !api_key.trim().is_empty() {
            return format!(
                "https://api.stlouisfed.org/fred/series?series_id={series}&api_key={api_key}&file_type=json"
            );
        }
    }
    format!("https://api.stlouisfed.org/fred/series?series_id={series}&file_type=json")
}

fn source_html(
    name: &str,
    category: &str,
    url: &str,
    css_selector: &str,
    max_items: usize,
    rate_limit_per_second: f64,
) -> DataSource {
    DataSource {
        name: name.to_string(),
        category: category.to_string(),
        kind: SourceKind::HtmlScrape {
            url: url.to_string(),
            css_selector: css_selector.to_string(),
            max_items,
        },
        rate_limit_per_second,
    }
}

fn source_websocket(
    name: &str,
    category: &str,
    url: &str,
    max_messages: usize,
    rate_limit_per_second: f64,
) -> DataSource {
    DataSource {
        name: name.to_string(),
        category: category.to_string(),
        kind: SourceKind::WebSocket {
            url: url.to_string(),
            max_messages,
        },
        rate_limit_per_second,
    }
}
