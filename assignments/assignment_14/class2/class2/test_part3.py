"""
Part 3 — Deployment Layer Tests
Tests for api_multimodel.py: canary routing, A/B testing, metrics, logging, performance.
Run: pytest class2/test_part3.py -v
"""

import io
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api_multimodel import app, prediction_logs, _rate_store

client = TestClient(app)


# ======================== Test Image Helpers ========================

def create_test_image(width=100, height=100, color=(255, 0, 0), fmt="PNG"):
    """Create a valid test image in memory"""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ======================== Fixtures ========================

@pytest.fixture(autouse=True)
def clear_state():
    """Clear prediction logs and rate limits before each test"""
    prediction_logs.clear()
    _rate_store.clear()
    yield
    prediction_logs.clear()
    _rate_store.clear()


@pytest.fixture
def test_images_set():
    """Generate a set of diverse test images"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 128), (64, 64, 64), (192, 192, 192), (32, 32, 32)
    ]
    return [create_test_image(100, 100, c) for c in colors]


# ======================== Canary Deployment Tests ========================

class TestCanaryDeployment:
    """Test canary traffic routing between v1 and v2"""

    def test_traffic_distribution(self):
        """100 requests should show roughly 70/30 split"""
        versions = {"v1.0": 0, "v2.0": 0}
        for i in range(100):
            _rate_store.clear()  # bypass rate limiter for traffic test
            img = create_test_image(100, 100, (i * 2, 0, 0))
            response = client.post("/predict", files={"file": (f"test_{i}.png", img, "image/png")})
            assert response.status_code == 200
            versions[response.json()["model_version"]] += 1

        # Allow wide tolerance due to randomness
        assert versions["v1.0"] > 40, f"v1 got only {versions['v1.0']}% — expected majority"
        assert versions["v2.0"] > 10, f"v2 got only {versions['v2.0']}% — expected some traffic"

    def test_force_v1(self):
        """Forcing v1 should always return v1.0"""
        img = create_test_image()
        response = client.post(
            "/predict?model_version=v1",
            files={"file": ("test.png", img, "image/png")}
        )
        assert response.json()["model_version"] == "v1.0"

    def test_force_v2(self):
        """Forcing v2 should always return v2.0"""
        img = create_test_image()
        response = client.post(
            "/predict?model_version=v2",
            files={"file": ("test.png", img, "image/png")}
        )
        assert response.json()["model_version"] == "v2.0"


# ======================== A/B Testing Tests ========================

class TestABTesting:
    """Test A/B comparison endpoint"""

    def test_predict_both_returns_200(self):
        img = create_test_image()
        response = client.post("/predict-both", files={"file": ("test.png", img, "image/png")})
        assert response.status_code == 200

    def test_predict_both_has_required_fields(self):
        img = create_test_image()
        data = client.post("/predict-both", files={"file": ("test.png", img, "image/png")}).json()
        assert "v1_prediction" in data
        assert "v2_prediction" in data
        assert "v1_confidence" in data
        assert "v2_confidence" in data
        assert "agreement" in data
        assert "v1_latency_ms" in data
        assert "v2_latency_ms" in data

    def test_agreement_is_boolean(self):
        img = create_test_image()
        data = client.post("/predict-both", files={"file": ("test.png", img, "image/png")}).json()
        assert isinstance(data["agreement"], bool)

    def test_predictions_are_valid_classes(self):
        valid_classes = {'animal', 'name_board', 'pedestrian', 'pothole', 'road_sign', 'speed_breaker', 'vehicle'}
        img = create_test_image()
        data = client.post("/predict-both", files={"file": ("test.png", img, "image/png")}).json()
        assert data["v1_prediction"] in valid_classes
        assert data["v2_prediction"] in valid_classes


# ======================== Metrics & Monitoring Tests ========================

class TestMetricsAndMonitoring:
    """Test metrics and monitoring endpoints"""

    def test_metrics_empty_initially(self):
        data = client.get("/metrics").json()
        assert data["total_requests"] == 0

    def test_metrics_track_requests(self):
        """Metrics should count requests after predictions"""
        for i in range(3):
            img = create_test_image()
            client.post("/predict?model_version=v1", files={"file": (f"t{i}.png", img, "image/png")})

        data = client.get("/metrics").json()
        assert data["total_requests"] == 3
        assert data["v1_requests"] == 3

    def test_metrics_track_both_versions(self):
        img = create_test_image()
        client.post("/predict?model_version=v1", files={"file": ("t.png", img, "image/png")})
        img = create_test_image()
        client.post("/predict?model_version=v2", files={"file": ("t.png", img, "image/png")})

        data = client.get("/metrics").json()
        assert data["v1_requests"] == 1
        assert data["v2_requests"] == 1

    def test_metrics_latency_positive(self):
        img = create_test_image()
        client.post("/predict", files={"file": ("t.png", img, "image/png")})
        data = client.get("/metrics").json()
        assert data["avg_latency_ms"] > 0


# ======================== Logging & Analytics Tests ========================

class TestLoggingAndAnalytics:
    """Test prediction logging and analytics endpoints"""

    def test_logs_empty_initially(self):
        data = client.get("/logs").json()
        assert data["total_logs"] == 0

    def test_logs_record_predictions(self):
        img = create_test_image()
        client.post("/predict", files={"file": ("t.png", img, "image/png")})
        data = client.get("/logs").json()
        assert data["total_logs"] == 1
        assert len(data["logs"]) == 1
        log = data["logs"][0]
        assert "timestamp" in log
        assert "prediction" in log
        assert "model_version" in log
        assert "latency_ms" in log

    def test_logs_pagination(self):
        """Logs should respect limit parameter"""
        for i in range(5):
            img = create_test_image()
            client.post("/predict?model_version=v1", files={"file": (f"t{i}.png", img, "image/png")})

        data = client.get("/logs?limit=3").json()
        assert data["total_logs"] == 5
        assert data["returned_logs"] == 3

    def test_stats_endpoint(self):
        img = create_test_image()
        client.post("/predict?model_version=v1", files={"file": ("t.png", img, "image/png")})
        data = client.get("/stats").json()
        assert data["total_requests"] == 1
        assert "latency_stats" in data
        assert "confidence_stats" in data


# ======================== Deployment Strategy Tests ========================

class TestDeploymentStrategies:
    """Test model isolation and deployment safety"""

    def test_model_isolation(self):
        """v1 and v2 should produce independent results"""
        img = create_test_image()
        r1 = client.post("/predict?model_version=v1", files={"file": ("t.png", img, "image/png")})

        img = create_test_image()
        r2 = client.post("/predict?model_version=v2", files={"file": ("t.png", img, "image/png")})

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["model_version"] == "v1.0"
        assert r2.json()["model_version"] == "v2.0"

    def test_predict_both_logs_two_entries(self):
        """A/B test should create log entries for both models"""
        img = create_test_image()
        client.post("/predict-both", files={"file": ("t.png", img, "image/png")})
        data = client.get("/logs").json()
        assert data["total_logs"] == 2


# ======================== Rate Limiting Tests ========================

class TestRateLimiting:
    """Test rate limiting behavior"""

    def test_rate_limit_enforced(self):
        """Should get 429 after exceeding 5 requests"""
        for i in range(5):
            img = create_test_image()
            response = client.post("/predict", files={"file": (f"t{i}.png", img, "image/png")})
            assert response.status_code == 200

        # 6th request should be rate limited
        img = create_test_image()
        response = client.post("/predict", files={"file": ("t5.png", img, "image/png")})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


# ======================== Performance SLA Tests ========================

class TestPerformanceSLAs:
    """Test performance requirements"""

    def test_prediction_latency_under_500ms(self):
        """Single prediction should complete within 500ms (after warmup)"""
        # Warmup
        img = create_test_image()
        client.post("/predict?model_version=v1", files={"file": ("warmup.png", img, "image/png")})

        # Measure
        img = create_test_image()
        start = time.time()
        response = client.post("/predict?model_version=v1", files={"file": ("t.png", img, "image/png")})
        elapsed_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 500, f"Latency {elapsed_ms:.0f}ms exceeds 500ms SLA"

    def test_ab_comparison_latency(self):
        """A/B comparison should complete within 1000ms"""
        # Warmup
        img = create_test_image()
        client.post("/predict-both", files={"file": ("warmup.png", img, "image/png")})

        img = create_test_image()
        start = time.time()
        response = client.post("/predict-both", files={"file": ("t.png", img, "image/png")})
        elapsed_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 1000, f"A/B latency {elapsed_ms:.0f}ms exceeds 1000ms SLA"

    def test_success_rate(self, test_images_set):
        """At least 99% of valid requests should succeed"""
        successes = 0
        total = len(test_images_set)

        for i, img in enumerate(test_images_set):
            _rate_store.clear()  # bypass rate limiter for bulk test
            response = client.post(
                "/predict?model_version=v1",
                files={"file": (f"test_{i}.png", img, "image/png")}
            )
            if response.status_code == 200:
                successes += 1

        success_rate = successes / total * 100
        assert success_rate >= 99.0, f"Success rate {success_rate:.1f}% below 99% SLA"
