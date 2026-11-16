from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time

import requests

from app.config import Settings

logger = logging.getLogger("ai-service")


def sign_payload(secret: str, timestamp: int, raw_body: str) -> str:
    message = f"{timestamp}.{raw_body}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _post_sync(url: str, raw_body: str, headers: dict[str, str], timeout_s: float) -> requests.Response:
    return requests.post(url, data=raw_body.encode("utf-8"), headers=headers, timeout=timeout_s)


async def send_webhook(callback_url: str, payload: dict, settings: Settings) -> bool:
    raw_body = json.dumps(payload, separators=(",", ":"))
    secret = settings.effective_webhook_secret
    loop = asyncio.get_running_loop()

    for attempt in range(max(1, settings.webhook_max_retries)):
        headers = {"Content-Type": "application/json"}
        if secret:
            ts = int(time.time())
            headers["X-Webhook-Timestamp"] = str(ts)
            headers["X-Webhook-Signature"] = sign_payload(secret, ts, raw_body)

        try:
            resp = await loop.run_in_executor(
                None, _post_sync, callback_url, raw_body, headers, settings.webhook_timeout_s,
            )
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(
                "request_id=%s webhook attempt=%d returned status=%d",
                payload.get("request_id"), attempt + 1, resp.status_code,
            )
            if resp.status_code < 500:
                return False
        except requests.RequestException:
            logger.warning(
                "request_id=%s webhook attempt=%d failed with network error",
                payload.get("request_id"), attempt + 1, exc_info=True,
            )

        if attempt + 1 < settings.webhook_max_retries:
            await asyncio.sleep(settings.webhook_retry_delay_s * (attempt + 1))

    return False

