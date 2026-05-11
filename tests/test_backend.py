import io
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_model():
    mock_model = MagicMock()
    mock_model.names = {0: "tumor"}

    mock_box = MagicMock()
    mock_box.cls = 0
    mock_box.conf = 0.92
    mock_box.xyxy.tolist.return_value = [[120.0, 80.0, 340.0, 290.0]]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]

    return mock_model


@pytest.fixture
def client():
    mock_model = _make_mock_model()
    mock_yolo_cls = MagicMock(return_value=mock_model)

    # Patch ultralytics.YOLO before importing backend so module-level
    # YOLO(MODEL_PATH) uses the mock instead of loading a real model.
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = mock_yolo_cls

    with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
        # Remove cached backend module so it re-imports with the mock
        sys.modules.pop("backend", None)

        import backend

        backend.model = mock_model
        from fastapi.testclient import TestClient

        yield TestClient(backend.app)

        # Cleanup
        sys.modules.pop("backend", None)


@pytest.fixture
def sample_image_bytes():
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


class TestHealthCheck:
    def test_root_returns_online(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["service"] == "Brain Tumor Detection API"
        assert data["version"] == "1.0.0"

    def test_root_model_loaded(self, client):
        response = client.get("/")
        data = response.json()
        assert data["model_loaded"] is True


class TestModelInfo:
    def test_model_info_returns_classes(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "YOLOv8"
        assert "classes" in data
        assert "num_classes" in data


class TestPredict:
    def test_predict_single_image(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            files=[("files", ("test.jpg", sample_image_bytes, "image/jpeg"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["successful"] == 1
        assert len(data["results"]) == 1

    def test_predict_returns_detections(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            files=[("files", ("test.jpg", sample_image_bytes, "image/jpeg"))],
        )
        data = response.json()
        result = data["results"][0]
        assert result["status"] == "success"
        assert result["filename"] == "test.jpg"
        assert "image_size" in result
        assert "detections" in result
        assert "detection_count" in result

    def test_predict_detection_fields(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            files=[("files", ("test.jpg", sample_image_bytes, "image/jpeg"))],
        )
        data = response.json()
        det = data["results"][0]["detections"][0]
        assert "class_id" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox_xyxy" in det
        assert len(det["bbox_xyxy"]) == 4

    def test_predict_multiple_images(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            files=[
                ("files", ("img1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("img2.jpg", sample_image_bytes, "image/jpeg")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

    def test_predict_invalid_file_type(self, client):
        response = client.post(
            "/predict",
            files=[("files", ("test.txt", b"not an image", "text/plain"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["failed"] == 1
        assert "failed_files" in data

    def test_predict_oversized_file(self, client):
        large_bytes = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/predict",
            files=[("files", ("big.jpg", large_bytes, "image/jpeg"))],
        )
        data = response.json()
        assert data["failed"] >= 1


class TestValidation:
    def test_validate_image_accepts_jpeg(self, client):
        import backend

        mock_file = MagicMock()
        mock_file.content_type = "image/jpeg"
        assert backend.validate_image(mock_file) is True

    def test_validate_image_accepts_png(self, client):
        import backend

        mock_file = MagicMock()
        mock_file.content_type = "image/png"
        assert backend.validate_image(mock_file) is True

    def test_validate_image_rejects_gif(self, client):
        import backend

        mock_file = MagicMock()
        mock_file.content_type = "image/gif"
        assert backend.validate_image(mock_file) is False

    def test_validate_image_rejects_text(self, client):
        import backend

        mock_file = MagicMock()
        mock_file.content_type = "text/plain"
        assert backend.validate_image(mock_file) is False
