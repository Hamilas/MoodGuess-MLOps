"""
Metrics collection middleware.

This middleware automatically collects Prometheus metrics for all requests.
"""

import contextlib
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

import app.monitoring.prometheus as _prom


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collects and exposes Prometheus metrics for HTTP requests."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        with contextlib.suppress(Exception):
            _prom.increment_active_requests()

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            try:
                _prom.record_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                )
                _prom.record_request_duration(
                    endpoint=request.url.path, method=request.method, duration=duration
                )
            except Exception:
                pass  # nosec B110 - metrics recording must never break the actual request
            return response

        except Exception as e:
            duration = time.time() - start_time
            try:
                _prom.record_request(
                    endpoint=request.url.path, method=request.method, status_code=500
                )
                _prom.record_request_duration(
                    endpoint=request.url.path, method=request.method, duration=duration
                )
            except Exception:
                pass  # nosec B110 - metrics recording must never break the actual request
            raise e

        finally:
            with contextlib.suppress(Exception):
                _prom.decrement_active_requests()
