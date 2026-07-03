"""
Configuration management for the MLOps sentiment analysis service.

This module provides backward-compatible access to the refactored configuration
system. The monolithic Settings class has been split into domain-specific
configuration classes in app/core/config/.

For new code, prefer importing from app.core.config directly:
    from app.core.config import Settings, get_settings

This module re-exports all configuration classes for backward compatibility.
"""

# Re-export all configuration classes
from app.core.config.data_lake import DataLakeConfig
from app.core.config.kafka import KafkaConfig
from app.core.config.mlops import MLOpsConfig
from app.core.config.model import ModelConfig
from app.core.config.monitoring import MonitoringConfig
from app.core.config.performance import PerformanceConfig
from app.core.config.redis import RedisConfig
from app.core.config.security import SecurityConfig
from app.core.config.server import ServerConfig
from app.core.config.settings import Settings, get_settings, settings
from app.core.config.vault import VaultConfig

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "ServerConfig",
    "ModelConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "KafkaConfig",
    "RedisConfig",
    "VaultConfig",
    "DataLakeConfig",
    "MonitoringConfig",
    "MLOpsConfig",
]
