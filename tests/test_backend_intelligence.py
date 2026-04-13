import unittest
from unittest.mock import patch

try:
    import backend as _backend
    BACKEND_IMPORT_ERROR = None
except Exception as exc:
    _backend = None
    BACKEND_IMPORT_ERROR = exc


def _sample_prediction_payload():
    return {
        "symbol": "BTC-USD",
        "predictions": [100.0, 101.0, 102.5, 103.0],
        "confidence_bands": [
            {"low": 99.0, "high": 101.0},
            {"low": 100.0, "high": 102.0},
            {"low": 101.5, "high": 103.5},
            {"low": 102.0, "high": 104.0},
        ],
        "model_info": {
            "last_close": 99.5,
            "residual_std": 1.0,
            "test_metrics": {"mape_pct": 4.0},
        },
    }


def _sample_market_context():
    return {
        "latest_features": [
            {"source": "BTC-USD", "momentum": 0.2, "rsi14": 48.0, "volume": 1000.0},
            {"source": "BTC-USD", "momentum": 0.4, "rsi14": 44.0, "volume": 1200.0},
            {"source": "BTC-USD", "momentum": 0.7, "rsi14": 39.0, "volume": 2600.0},
        ],
        "latest_signals": [
            {"source": "BTC-USD", "recommendation": "bullish", "confidence": 0.8, "signal_ts_ms": 1710000000000}
        ],
        "signal_summary": {
            "sources": [{"source": "BTC-USD", "bullish": 10, "bearish": 3, "total": 13}],
        },
    }


class BackendIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _backend is None:
            raise unittest.SkipTest(f"backend runtime dependencies unavailable: {BACKEND_IMPORT_ERROR}")

    def setUp(self):
        self.backend = _backend
        self.client = self.backend.app.test_client()
        with self.backend._strategy_performance_lock:
            self.backend._strategy_performance_log.clear()

    def test_generate_auto_trade_decision_returns_intelligence_payload(self):
        decision = self.backend.generate_auto_trade_decision(
            symbol="BTC-USD",
            prediction_payload=_sample_prediction_payload(),
            market_context=_sample_market_context(),
        )
        self.assertIn("action", decision)
        self.assertIn(decision["action"], {"buy", "sell", "hold"})
        self.assertIn("intelligence", decision)
        self.assertIn("market_dna", decision["intelligence"])
        self.assertIn("multi_agent", decision["intelligence"])
        self.assertIn("confidence_engine", decision["intelligence"])
        self.assertIn("explainability", decision["intelligence"])
        self.assertGreaterEqual(len(decision["intelligence"]["explainability"]), 1)

    @patch("backend.load_rust_market_context")
    @patch("backend.run_single_prediction")
    def test_auto_intelligence_endpoint(self, mock_run_single_prediction, mock_market_context):
        mock_run_single_prediction.return_value = (_sample_prediction_payload(), None)
        mock_market_context.return_value = _sample_market_context()

        response = self.client.post("/trade/auto/intelligence", json={"symbol": "BTC-USD"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("intelligence", body)
        self.assertIn("decision_preview", body)
        self.assertIn("action", body["decision_preview"])
        self.assertIn("risk_level", body["decision_preview"])

    @patch("backend.load_rust_market_context")
    @patch("backend.run_single_prediction")
    def test_auto_decision_endpoint_and_performance_telemetry(self, mock_run_single_prediction, mock_market_context):
        mock_run_single_prediction.return_value = (_sample_prediction_payload(), None)
        mock_market_context.return_value = _sample_market_context()

        response = self.client.post("/trade/auto/decision", json={"symbol": "BTC-USD"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("decision", body)
        self.assertIn("intelligence", body["decision"])

        perf = self.client.get("/trade/auto/performance?symbol=BTC-USD")
        self.assertEqual(perf.status_code, 200)
        perf_body = perf.get_json()
        self.assertGreaterEqual(perf_body.get("total_records", 0), 1)


if __name__ == "__main__":
    unittest.main()
