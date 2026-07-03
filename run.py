#!/usr/bin/env python3
"""A development server launcher for the sentiment analysis service.

Usage:
    python run.py

This script provides a convenient way to start the application in a local
development environment. It sets sensible default environment variables for
development and uses uvicorn with hot reloading.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def print_startup_info(settings):
    """Prints a formatted banner with useful development information."""
    host_display = "localhost" if settings.host == "0.0.0.0" else settings.host

    print("🚀 MLOps Sentiment Analysis Service")
    print("=" * 50)
    print(f"📁 Project Directory: {project_root}")
    print(f"🐍 Python: {sys.executable}")
    print(f"🔧 Mode: {'Debug' if settings.debug else 'Production'}")
    print(f"📝 Log Level: {settings.log_level}")
    print("📊 Metrics: Enabled")
    print()
    print("📚 Available Endpoints:")
    print(f"  • API Docs (Swagger): http://{host_display}:{settings.port}/docs")
    print(f"  • API Docs (ReDoc):   http://{host_display}:{settings.port}/redoc")
    print(f"  • Health Check:       http://{host_display}:{settings.port}/health")
    print(f"  • Metrics:            http://{host_display}:{settings.port}/metrics")
    print(f"  • Predict:            http://{host_display}:{settings.port}/api/v1/predict")
    print()
    print("=" * 50)


if __name__ == "__main__":
    # Set development environment variables
    # Note: pydantic parses 'true' (case-insensitive) as True for boolean fields
    os.environ.setdefault("MLOPS_DEBUG", "true")
    os.environ.setdefault("MLOPS_LOG_LEVEL", "INFO")
    os.environ.setdefault("MLOPS_ENABLE_METRICS", "true")
    # Force development profile if not set - this script is explicitly for dev usage
    os.environ.setdefault("MLOPS_PROFILE", "development")

    try:
        import uvicorn

        from app.core.config import get_settings

        settings = get_settings()
        print_startup_info(settings)

        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            reload=settings.debug,
            reload_dirs=[str(project_root / "app")],
        )

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you've installed dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
