from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-6.1-terra"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dimensions: int = 1536
    min_similarity: float = field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY", "0.30")))
    max_age_hours: int = field(default_factory=lambda: int(os.getenv("MAX_DATA_AGE_HOURS", "24")))
    raw_s3_bucket: str = field(default_factory=lambda: os.getenv("RAW_S3_BUCKET", ""))
    max_date_range_days: int = field(default_factory=lambda: int(os.getenv("MAX_DATE_RANGE_DAYS", "31")))
    meal_refresh_hours: int = field(default_factory=lambda: int(os.getenv("MEAL_REFRESH_HOURS", "6")))
    meal_empty_refresh_minutes: int = field(default_factory=lambda: int(os.getenv("MEAL_EMPTY_REFRESH_MINUTES", "60")))
    refresh_retry_minutes: int = field(default_factory=lambda: int(os.getenv("REFRESH_RETRY_MINUTES", "5")))
    schedule_enabled: bool = field(default_factory=lambda: _env_bool("SCHEDULE_ENABLED", True))
    schedule_interval_minutes: int = field(default_factory=lambda: int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "60")))
    schedule_start_offset_days: int = field(default_factory=lambda: int(os.getenv("SCHEDULE_START_OFFSET_DAYS", "0")))
    schedule_days: int = field(default_factory=lambda: int(os.getenv("SCHEDULE_DAYS", "14")))
    agentic_enabled: bool = field(default_factory=lambda: _env_bool("AGENTIC_ENABLED", True))
    agentic_max_turns: int = field(default_factory=lambda: int(os.getenv("AGENTIC_MAX_TURNS", "5")))

    def __post_init__(self):
        for name in ("max_date_range_days", "schedule_interval_minutes", "schedule_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("meal_refresh_hours", "meal_empty_refresh_minutes", "refresh_retry_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.schedule_start_offset_days,bool) or not isinstance(self.schedule_start_offset_days,int):
            raise ValueError("schedule_start_offset_days must be an integer")
        if not isinstance(self.schedule_enabled, bool):
            raise ValueError("schedule_enabled must be a boolean")
        if not isinstance(self.agentic_enabled, bool):
            raise ValueError("agentic_enabled must be a boolean")
        if isinstance(self.agentic_max_turns,bool) or not isinstance(self.agentic_max_turns,int) or not 1<=self.agentic_max_turns<=8:
            raise ValueError("agentic_max_turns must be an integer from 1 to 8")
        if self.schedule_days > self.max_date_range_days:
            raise ValueError("schedule_days must not exceed max_date_range_days")
        self.data_dir.mkdir(parents=True, exist_ok=True)
