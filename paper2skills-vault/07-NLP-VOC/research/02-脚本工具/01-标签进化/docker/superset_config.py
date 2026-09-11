"""Phase 7 D2 — Superset config (mounted at /app/pythonpath/superset_config.py).

Minimal dev config. Disables CSRF for local dev convenience and sets a stable
secret key. NOT for production.
"""
import os

SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "voc_bi_dev_secret_key_not_for_production")

APP_NAME = "路特VOC自助工作台"

WTF_CSRF_ENABLED = False

FEATURE_FLAGS = {
    "DASHBOARD_RBAC": True,
    "ENABLE_TEMPLATE_PROCESSING": True,
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    "ALERT_REPORTS": False,
}

CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_URL": "redis://superset-redis:6379/0",
}

DATA_CACHE_CONFIG = CACHE_CONFIG

LANGUAGES = {
    "en": {"flag": "us", "name": "English"},
    "zh": {"flag": "cn", "name": "Chinese"},
}
BABEL_DEFAULT_LOCALE = "zh"

ROW_LIMIT = 100000
QUERY_SEARCH_LIMIT = 1000
SQL_MAX_ROW = 100000
DEFAULT_RELATIVE_START_TIME = "today"

SUPERSET_LOAD_EXAMPLES = False
