from PIL import Image

from frontend.frontend import crop_zoom, draw_bboxes, get_detection_summary


def _make_test_image(width=200, height=200):
    return Image.new("RGB", (width, height), color="gray")


def _make_detection(class_name="tumor", confidence=0.9, bbox=None):
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox or [50.0, 50.0, 150.0, 150.0],
    }


class TestDrawBboxes:
    def test_returns_image(self):
        img = _make_test_image()
        det = [_make_detection()]
        result = draw_bboxes(img.copy(), det)
        assert isinstance(result, Image.Image)

    def test_filters_by_confidence(self):
        img = _make_test_image()
        det = [_make_detection(confidence=0.3)]
        result = draw_bboxes(img.copy(), det, confidence_threshold=0.5)
        assert isinstance(result, Image.Image)

    def test_high_confidence_draws(self):
        img = _make_test_image()
        det = [_make_detection(confidence=0.85)]
        result = draw_bboxes(img.copy(), det, confidence_threshold=0.5)
        assert isinstance(result, Image.Image)

    def test_empty_detections(self):
        img = _make_test_image()
        result = draw_bboxes(img.copy(), [])
        assert isinstance(result, Image.Image)


class TestCropZoom:
    def test_crop_returns_image(self):
        img = _make_test_image(400, 400)
        result = crop_zoom(img, [100, 100, 200, 200])
        assert isinstance(result, Image.Image)

    def test_crop_respects_padding(self):
        img = _make_test_image(400, 400)
        result = crop_zoom(img, [100, 100, 200, 200], padding=20)
        assert result.size[0] == 140
        assert result.size[1] == 140

    def test_crop_clamps_to_image_bounds(self):
        img = _make_test_image(200, 200)
        result = crop_zoom(img, [0, 0, 50, 50], padding=100)
        assert result.size[0] <= 200
        assert result.size[1] <= 200


class TestDetectionSummary:
    def test_positive_detection(self):
        dets = [_make_detection(confidence=0.9)]
        summary, status = get_detection_summary(dets, 0.5)
        assert status == "positive"
        assert "1 tumor" in summary

    def test_negative_detection(self):
        dets = [_make_detection(confidence=0.3)]
        summary, status = get_detection_summary(dets, 0.5)
        assert status == "negative"

    def test_empty_detections(self):
        summary, status = get_detection_summary([], 0.5)
        assert status == "negative"

    def test_multiple_detections(self):
        dets = [_make_detection(confidence=0.9), _make_detection(confidence=0.85)]
        summary, status = get_detection_summary(dets, 0.5)
        assert "2 tumor" in summary
        assert status == "positive"
