"""Unit tests for /api/v1/detect-async and HMAC webhook delivery."""
import asyncio
import hashlib
import hmac
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from PIL import Image

from app import errors
from app.config import Settings
from app.errors import ServiceError
from app.pipeline.pipeline import ScanResult
from app.routers.detect import detect_async
from app.schemas import DetectAsyncRequest
from app.webhook import send_webhook, sign_payload


def png_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "PNG")
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


def make_request(settings: Settings | None = None, loaded: bool = True, pending_jobs: int = 0):
    engines = Mock()
    engines.florence.loaded = loaded
    engines.clip.loaded = False
    engines.device = "cpu"
    state = SimpleNamespace(
        settings=settings
        or Settings(
            device="cpu",
            warmup_on_load=False,
            max_upload_mb=1,
            webhook_secret="wh-secret",
            webhook_retry_delay_s=0.0,
            max_queue_size=2,
        ),
        engines=engines,
        inference_slot=asyncio.Semaphore(1),
        pending_jobs=pending_jobs,
        background_tasks=set(),
    )
    app = SimpleNamespace(state=state)
    return SimpleNamespace(app=app)


class DetectAsyncTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.webhook.requests.post")
    @patch("app.routers.detect.process_image")
    @patch("app.routers.detect.requests.get")
    async def test_detect_async_queues_job_and_posts_signed_done_webhook(
        self, mock_get, mock_process, mock_post,
    ):
        mock_get.return_value = Mock(
            status_code=200,
            content=png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        mock_process.return_value = ScanResult([detection()], None, 64, 48)
        mock_post.return_value = Mock(status_code=200)

        req = make_request()
        body = DetectAsyncRequest(
            request_id="req-async-1",
            image_url="https://minio.local/scans/7/abc.png",
            callback_url="https://backend.local/api/webhooks/scan",
        )
        accepted = await detect_async(req, body)
        self.assertEqual(accepted.request_id, "req-async-1")
        self.assertEqual(accepted.status, "QUEUED")

        await asyncio.gather(*list(req.app.state.background_tasks))

        self.assertEqual(req.app.state.pending_jobs, 0)
        mock_post.assert_called_once()
        post_args, post_kwargs = mock_post.call_args
        self.assertEqual(post_args[0], "https://backend.local/api/webhooks/scan")
        raw_body = post_kwargs["data"].decode("utf-8")
        payload = json.loads(raw_body)
        self.assertEqual(payload["request_id"], "req-async-1")
        self.assertEqual(payload["status"], "DONE")
        self.assertEqual(payload["image_width"], 64)
        self.assertEqual(payload["image_height"], 48)
        self.assertEqual(payload["detections"][0]["label"], "coffee mug")

        headers = post_kwargs["headers"]
        ts = int(headers["X-Webhook-Timestamp"])
        expected_sig = sign_payload("wh-secret", ts, raw_body)
        self.assertEqual(headers["X-Webhook-Signature"], expected_sig)

    @patch("app.webhook.requests.post")
    @patch("app.routers.detect.requests.get")
    async def test_detect_async_posts_failed_webhook_on_invalid_image(self, mock_get, mock_post):
        mock_get.return_value = Mock(
            status_code=200,
            content=b"not-an-image",
            headers={"Content-Type": "image/png"},
        )
        mock_post.return_value = Mock(status_code=200)

        req = make_request()
        body = DetectAsyncRequest(
            request_id="req-async-2",
            image_url="https://minio.local/scans/7/bad.png",
            callback_url="https://backend.local/api/webhooks/scan",
        )
        await detect_async(req, body)
        await asyncio.gather(*list(req.app.state.background_tasks))

        self.assertEqual(req.app.state.pending_jobs, 0)
        raw_body = mock_post.call_args.kwargs["data"].decode("utf-8")
        payload = json.loads(raw_body)
        self.assertEqual(payload["request_id"], "req-async-2")
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["error_code"], errors.INVALID_IMAGE)

    async def test_detect_async_rejects_when_queue_full_503(self):
        req = make_request(pending_jobs=2)
        body = DetectAsyncRequest(
            request_id="req-async-3",
            image_url="https://minio.local/scans/7/a.png",
            callback_url="https://backend.local/api/webhooks/scan",
        )
        with self.assertRaises(ServiceError) as ctx:
            await detect_async(req, body)

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, errors.QUEUE_FULL)

    async def test_detect_async_rejects_invalid_urls_400(self):
        req = make_request()
        body = DetectAsyncRequest(
            request_id="req-async-4",
            image_url="ftp://minio.local/img.png",
            callback_url="https://backend.local/api/webhooks/scan",
        )
        with self.assertRaises(ServiceError) as ctx:
            await detect_async(req, body)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, errors.INVALID_REQUEST)

    @patch("app.webhook.requests.post")
    async def test_send_webhook_retries_on_5xx_and_signs_hmac_sha256(self, mock_post):
        mock_post.side_effect = [
            requests.ConnectionError("temporary error"),
            Mock(status_code=502),
            Mock(status_code=200),
        ]
        settings = Settings(
            device="cpu",
            warmup_on_load=False,
            service_token="fallback-secret",
            webhook_max_retries=3,
            webhook_retry_delay_s=0.0,
        )
        ok = await send_webhook("https://backend.local/api/webhooks/scan", {"request_id": "r1", "status": "DONE"}, settings)

        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 3)
        last_kwargs = mock_post.call_args.kwargs
        raw_body = last_kwargs["data"].decode("utf-8")
        ts = last_kwargs["headers"]["X-Webhook-Timestamp"]
        expected = hmac.new(
            b"fallback-secret",
            f"{ts}.{raw_body}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(last_kwargs["headers"]["X-Webhook-Signature"], expected)


if __name__ == "__main__":
    unittest.main()

