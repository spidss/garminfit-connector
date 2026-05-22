"""
Strava OAuth 2.0 client and API wrapper.

OAuth flow:
  1. Redirect user to Strava authorization URL (build_auth_url)
  2. Strava POSTs back to /strava/callback with ?code=...&state=...
  3. Exchange the code for tokens (exchange_code)
  4. Tokens are stored encrypted in the User row
  5. On each MCP call, tokens are refreshed if within 5 minutes of expiry

Required environment variables:
  STRAVA_CLIENT_ID      — numeric ID from strava.com/settings/api
  STRAVA_CLIENT_SECRET  — client secret from strava.com/settings/api

The redirect URI registered in your Strava app must match:
  {APP_BASE_URL}/strava/callback
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"

# Scopes needed for full activity + profile access
_OAUTH_SCOPE = "activity:read_all,profile:read_all"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def strava_configured() -> bool:
    """Return True if Strava client credentials are set."""
    return bool(
        os.environ.get("STRAVA_CLIENT_ID", "").strip()
        and os.environ.get("STRAVA_CLIENT_SECRET", "").strip()
    )


def _get_credentials() -> tuple[str, str]:
    client_id = os.environ.get("STRAVA_CLIENT_ID", "").strip()
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET environment variables are required. "
            "Create a Strava API application at https://www.strava.com/settings/api"
        )
    return client_id, client_secret


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def build_auth_url(redirect_uri: str, state: str) -> str:
    """Build the Strava OAuth authorization URL."""
    client_id, _ = _get_credentials()
    params = urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": _OAUTH_SCOPE,
        "state": state,
    })
    return f"{STRAVA_AUTH_URL}?{params}"


def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """
    Exchange an authorization code for access/refresh tokens.
    Returns the full Strava token response dict, including an 'athlete' key.
    """
    client_id, client_secret = _get_credentials()
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
        resp.raise_for_status()
        return resp.json()


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """
    Refresh an expired Strava access token.
    Returns the updated token response dict.
    """
    client_id, client_secret = _get_credentials()
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class StravaApiClient:
    """
    Thin synchronous Strava API client.

    Designed to be called from asyncio via run_in_executor, matching the
    pattern used for the Garmin client throughout this codebase.
    """

    def __init__(self, access_token: str, athlete_id: Optional[int] = None) -> None:
        self._token = access_token
        self.athlete_id = athlete_id

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _get(self, path: str, **params) -> Any:
        url = f"{STRAVA_API_BASE}{path}"
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                params={k: v for k, v in params.items() if v is not None},
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Athlete
    # ------------------------------------------------------------------

    def get_athlete(self) -> Dict[str, Any]:
        """Return the authenticated athlete's profile."""
        return self._get("/athlete")

    def get_athlete_stats(self, athlete_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Return lifetime and recent totals for runs, rides, and swims.
        Uses the stored athlete_id if none is provided.
        """
        aid = athlete_id or self.athlete_id
        if not aid:
            raise RuntimeError("athlete_id required for get_athlete_stats")
        return self._get(f"/athletes/{aid}/stats")

    def get_athlete_zones(self) -> Dict[str, Any]:
        """Return the athlete's heart rate and power zones."""
        return self._get("/athlete/zones")

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    def get_activities(
        self,
        after: Optional[int] = None,
        before: Optional[int] = None,
        page: int = 1,
        per_page: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of the athlete's activities.

        Args:
            after:    Unix timestamp — only return activities after this time.
            before:   Unix timestamp — only return activities before this time.
            page:     Page number for pagination.
            per_page: Activities per page (max 200).
        """
        return self._get(
            "/athlete/activities",
            after=after,
            before=before,
            page=page,
            per_page=min(per_page, 200),
        )

    def get_activity(self, activity_id: int) -> Dict[str, Any]:
        """Return detailed data for a single activity, including segment efforts."""
        return self._get(f"/activities/{activity_id}", include_all_efforts=True)

    def get_activity_laps(self, activity_id: int) -> List[Dict[str, Any]]:
        """Return lap data for a single activity."""
        return self._get(f"/activities/{activity_id}/laps")

    def get_activity_zones(self, activity_id: int) -> List[Dict[str, Any]]:
        """Return HR and power zone distribution for a single activity."""
        return self._get(f"/activities/{activity_id}/zones")

    def get_starred_segments(
        self, page: int = 1, per_page: int = 30
    ) -> List[Dict[str, Any]]:
        """Return the athlete's starred segments."""
        return self._get("/segments/starred", page=page, per_page=per_page)
