"""
Multi-user Strava adapter.

Mirrors the pattern of garmin_adapter.py:
  1. Look up the User record by the MCP access token
  2. Check whether Strava is connected (tokens exist)
  3. Refresh the access token if it expires within 5 minutes
  4. Persist refreshed tokens back to the DB
  5. Return a ready-to-use StravaApiClient

Usage (in MCP tools):
    client = await get_strava_client(access_token)
    data = await loop.run_in_executor(None, client.get_athlete)
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.auth_manager import decrypt_token, encrypt_token
from app.database import SessionLocal, User
from app.strava_client import StravaApiClient, refresh_access_token

log = logging.getLogger(__name__)

# Refresh if the token expires within this many seconds
_REFRESH_BUFFER_SECS = 300


async def get_strava_client(access_token: str) -> StravaApiClient:
    """
    Load a user's Strava tokens from the DB and return an authenticated
    StravaApiClient, automatically refreshing the access token if needed.

    Raises:
        RuntimeError: If the user doesn't exist/is revoked, or hasn't
                      connected their Strava account yet.
    """
    async with SessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.access_token == access_token,
                User.revoked == False,  # noqa: E712
            )
        )
        user = result.scalar_one_or_none()

    if not user:
        raise RuntimeError("User not found or revoked.")

    if not user.strava_access_token_encrypted:
        raise RuntimeError(
            "Strava not connected. "
            "Visit the setup page and use the 'Connect Strava' button after signing in, "
            "or navigate to /strava/connect?token=YOUR_TOKEN."
        )

    strava_access = decrypt_token(user.strava_access_token_encrypted)
    strava_refresh = decrypt_token(user.strava_refresh_token_encrypted)
    expires_at: Optional[datetime] = user.strava_token_expires_at
    athlete_id = int(user.strava_athlete_id) if user.strava_athlete_id else None

    # Refresh if expiring soon
    now = datetime.utcnow()
    if expires_at is None or (expires_at - now).total_seconds() < _REFRESH_BUFFER_SECS:
        log.info("Refreshing Strava access token for user %s", access_token[:8])
        loop = asyncio.get_event_loop()
        try:
            token_data = await loop.run_in_executor(
                None, refresh_access_token, strava_refresh
            )
        except Exception as exc:
            log.warning("Strava token refresh failed: %s", exc)
            # Fall through — the existing token may still work for a bit
        else:
            strava_access = token_data["access_token"]
            new_refresh = token_data.get("refresh_token", strava_refresh)
            new_expires = datetime.utcfromtimestamp(token_data["expires_at"])

            enc_access = encrypt_token(strava_access)
            enc_refresh = encrypt_token(new_refresh)

            try:
                async with SessionLocal() as db:
                    result = await db.execute(
                        select(User).where(User.access_token == access_token)
                    )
                    u = result.scalar_one_or_none()
                    if u:
                        u.strava_access_token_encrypted = enc_access
                        u.strava_refresh_token_encrypted = enc_refresh
                        u.strava_token_expires_at = new_expires
                        await db.commit()
            except Exception as exc:
                log.warning("Failed to persist refreshed Strava tokens: %s", exc)

    return StravaApiClient(access_token=strava_access, athlete_id=athlete_id)
