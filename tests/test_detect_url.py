"""Unit tests for /api/v1/detect-url without requiring httpx/TestClient."""
import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from PIL import Image

from app import errors
from app.config import Settings
from app.errors import ServiceError
from app.pipeline.pipeline import ScanResult
from app.routers.detect import detect_by_url
from app.schemas import DetectUrlRequest


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, fmt)
    return buf.getvalue()


def detection(**overrides):
    base = {
        "label": "coffee mug",
        "headword": "mug",
        "score": 0.9,
        "source": "od",
        "reliability": "HIGH",
        "box": [1.0, 2.0, 30.0, 40.0],
    }
    return {**base, **overrides}


def make_request(settings: Settings | None = None, loaded: bool = True):
    engines = Mock()
    engines.florence.loaded = loaded
    engines.clip.loaded = False
    engines.device = "cpu"
    state = SimpleNamespace(
        settings=settings or Settings(device="cpu", warmup_on_load=False, max_upload_mb=1),
        engines=engines,
        inference_slot=asyncio.Semaphore(1),
    )
    app = SimpleNamespace(state=state)
    return SimpleNamespace(app=app)


class DetectUrlTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.routers.detect.process_image")
    @patch("app.routers.detect.requests.get")
    async def test_success_downloads_image_and_returns_detections(self, mock_get, mock_process):
        mock_get.return_value = Mock(
            status_code=200,
            content=image_bytes("PNG", (64, 48)),
            headers={"Content-Type": "image/png"},
        )
        mock_process.return_value = ScanResult([detection()], None, 64, 48)

        req = make_request()
        res = await detect_by_url(
            req,
            DetectUrlRequest(image_url="https://minio.local/bucket/scans/7/abc.png?sig=1"),
            x_request_id="req-url-1",
        )

        mock_get.assert_called_once_with(
            "https://minio.local/bucket/scans/7/abc.png?sig=1",
            timeout=req.app.state.settings.image_download_timeout_s,
            allow_redirects=True,
        )
        self.assertEqual(res.request_id, "req-url-1")
        self.assertEqual(res.image_width, 64)
        self.assertEqual(res.image_height, 48)
        self.assertEqual(len(res.detections), 1)
        self.assertEqual(res.detections[0].label, "coffee mug")
        self.assertEqual(res.detections[0].headword, "mug")

    async def test_rejects_non_http_url_as_invalid_request(self):
        req = make_request()
        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="file:///etc/passwd"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, errors.INVALID_REQUEST)

    @patch("app.routers.detect.requests.get")
    async def test_storage_http_error_is_model_error_502(self, mock_get):
        mock_get.return_value = Mock(status_code=403, content=b"Forbidden", headers={})
        req = make_request()

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/img.png"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.code, errors.MODEL_ERROR)

    @patch("app.routers.detect.requests.get", side_effect=requests.Timeout("timed out"))
    async def test_storage_network_exception_is_model_error_502(self, _mock_get):
        req = make_request()

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/img.png"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.code, errors.MODEL_ERROR)

    @patch("app.routers.detect.requests.get")
    async def test_too_large_downloaded_image_is_invalid_image_413(self, mock_get):
        mock_get.return_value = Mock(
            status_code=200,
            content=b"0" * (1024 * 1024 + 1),
            headers={"Content-Type": "image/png"},
        )
        req = make_request()

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/big.png"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(ctx.exception.code, errors.INVALID_IMAGE)

    @patch("app.routers.detect.requests.get")
    async def test_unsupported_image_format_is_invalid_image_415(self, mock_get):
        mock_get.return_value = Mock(
            status_code=200,
            content=image_bytes("GIF"),
            headers={"Content-Type": "application/octet-stream"},
        )
        req = make_request()

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/img.bin"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 415)
        self.assertEqual(ctx.exception.code, errors.INVALID_IMAGE)

    @patch("app.routers.detect.requests.get")
    async def test_undecodable_bytes_is_invalid_image_400(self, mock_get):
        mock_get.return_value = Mock(
            status_code=200,
            content=b"not an image",
            headers={"Content-Type": "image/png"},
        )
        req = make_request()

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/bad.png"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, errors.INVALID_IMAGE)

    async def test_models_loading_is_model_not_ready_503(self):
        req = make_request(loaded=False)

        with self.assertRaises(ServiceError) as ctx:
            await detect_by_url(req, DetectUrlRequest(image_url="https://minio.local/img.png"), x_request_id="req-1")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, errors.MODEL_NOT_READY)


if __name__ == "__main__":
    unittest.main()

