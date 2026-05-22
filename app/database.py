"""
Database models and async session management.
Uses SQLAlchemy 2.x async with asyncpg driver (Railway PostgreSQL).
"""

import os
import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Railway may set DATABASE_URL as either postgres:// or postgresql://
# asyncpg requires the postgresql+asyncpg:// scheme — normalize all variants
_raw_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./dev.db")

if _raw_url.startswith("postgres://"):
    # Old Heroku/Railway style: postgres://user:pass@host/db
    DATABASE_URL = "postgresql+asyncpg://" + _raw_url[len("postgres://"):]
elif _raw_url.startswith("postgresql://") and "+asyncpg" not in _raw_url:
    # New Railway style: postgresql://user:pass@host/db
    DATABASE_URL = "postgresql+asyncpg://" + _raw_url[len("postgresql://"):]
else:
    # Already correct (postgresql+asyncpg://) or local SQLite fallback
    DATABASE_URL = _raw_url

print(f"[DB] Connecting with scheme: {DATABASE_URL.split('://')[0]}")

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class User(Base):
    """
    Represents one connected Garmin user.
    No passwords stored — only encrypted garth OAuth tokens.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # The token embedded in the user's personal MCP URL path
    access_token: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # Fernet-encrypted garth.dumps() base64 string
    garth_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)

    # Garmin display name (informational only)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Garmin email — stored so we can look up users for revocation
    garmin_email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    token_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Soft-delete for revocation — revoked tokens are rejected by tool calls
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ---------------------------------------------------------------------------
    # Strava OAuth tokens (optional — only set after user connects Strava)
    # ---------------------------------------------------------------------------
    strava_athlete_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    strava_athlete_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    strava_access_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    strava_refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    strava_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


async def get_db():
    """FastAPI/Starlette dependency for async DB sessions."""
    async with SessionLocal() as session:
        yield session


async def create_tables():
    """Create all tables if they don't exist (called on startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
