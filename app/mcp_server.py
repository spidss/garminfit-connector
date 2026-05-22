"""
FastMCP server with 38 Garmin data tools.

All tools:
  - Read the user's access token from a contextvars.ContextVar set by GarminMCPRouter
  - Run blocking garminconnect calls in a thread executor (never block the event loop)
  - Default to today's date when no date is provided
  - Return formatted strings suitable for Claude to read and summarize

Tool groupings:
  Group 1  — Daily Overview:          get_health_snapshot, get_today_summary, get_sleep_summary, get_activities
  Group 2  — Recovery & Wellness:     get_body_battery, get_stress_summary, get_hrv_status, get_heart_rate
  Group 3  — Training Performance:    get_training_status, get_training_readiness, get_intensity_minutes
  Group 4  — Nutrition & Hydration:   get_nutrition_log, get_hydration
  Group 5  — Comprehensive & Range:   get_body_metrics, get_spo2_and_respiration, get_activities_by_date_range
  Group 6  — Gear Tracking:           get_gear, get_gear_stats, get_gear_activities, get_activity_gear, get_gear_defaults
  Group 7  — Activity Details:        get_activity_details, get_activity_splits, get_activity_hr_zones,
                                       get_activity_power_zones, get_activity_exercise_sets, get_activity_weather
  Group 8  — Advanced Performance:    get_race_predictions, get_endurance_score, get_hill_score,
                                       get_lactate_threshold, get_cycling_ftp, get_running_tolerance, get_fitness_age
  Group 9  — Body & Health:           get_resting_heart_rate, get_body_battery_events, get_weigh_ins, get_blood_pressure
  Group 10 — Weekly Trends:           get_weekly_step_trends, get_weekly_stress_trends, get_weekly_intensity_trends
  Group 11 — Goals & Achievements:    get_personal_records, get_earned_badges
  Group 12 — Devices:                 get_devices
  Group 13 — Nutrition Details:       get_nutrition_meals
"""

import asyncio
import contextvars
from datetime import datetime, timedelta
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.garmin_adapter import get_garmin_handler, save_refreshed_tokens

# ---------------------------------------------------------------------------
# ContextVar — set by GarminMCPRouter for each request
# ---------------------------------------------------------------------------

# This variable holds the current user's access token during a request.
# It is set in app/main.py's GarminMCPRouter before delegating to the MCP app.
user_access_token_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "user_access_token"
)


def _get_token() -> str:
    """Retrieve the current user's access token from the context variable."""
    try:
        return user_access_token_var.get()
    except LookupError:
        raise RuntimeError(
            "No user access token found in request context. "
            "Make sure you are connecting via your personal MCP URL."
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Core helper: get handler + run blocking call in thread pool
# ---------------------------------------------------------------------------

async def _call(method_name: str, *args, **kwargs) -> str:
    """
    Get an authenticated MultiUserGarminHandler for the current user,
    run the named method in a thread executor, return the result as a string.
    """
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    def _run():
        method = getattr(handler, method_name)
        return method(*args, **kwargs)

    result = await loop.run_in_executor(None, _run)

    # Persist any token refresh that occurred
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if result is None:
        return "No data available for this metric."
    return str(result) if not isinstance(result, str) else result


async def _format(data_type: str, activity_limit: int = 5) -> str:
    """Call format_data_for_context() which wraps multiple Garmin sub-calls."""
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None, handler.format_data_for_context, data_type, activity_limit
    )

    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if not result:
        return "No data available for this metric."
    return str(result)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Garmin Fitness",
    json_response=True,     # Plain JSON responses, no SSE streaming
    stateless_http=True,    # Fresh server per request — no session state needed
    # Disable MCP SDK's DNS-rebinding check: we're already on HTTPS (TLS prevents
    # DNS rebinding at the network layer), and allowing arbitrary hosts is required
    # for a public connector that Claude.ai connects to from external servers.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    instructions=(
        "Tools for querying the connected user's Garmin Connect fitness data. "
        "All date parameters use YYYY-MM-DD format. "
        "When no date is specified, today's date is used automatically. "
        "Data availability depends on the user's Garmin device model — "
        "some metrics (e.g. nutrition, hydration, body composition) require "
        "manual logging or specific Garmin hardware. "
        "Start with get_health_snapshot for a comprehensive daily overview."
    ),
)


# ===========================================================================
# GROUP 1 — Daily Overview
# ===========================================================================

@mcp.tool()
async def get_health_snapshot(date: Optional[str] = None) -> str:
    """
    Returns a comprehensive health data snapshot for one day.
    Includes steps, calories, sleep, body battery, stress, HRV,
    heart rate, and training status — all in one response.

    This is the best tool for a complete daily health overview.
    Use this instead of calling multiple individual tools.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Formatted text summary of all available health metrics for that day.
    """
    if not date:
        date = _today()
    return await _format("comprehensive")


@mcp.tool()
async def get_today_summary(date: Optional[str] = None) -> str:
    """
    Returns a daily activity summary: step count, calories burned,
    distance, active minutes, and goal progress.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Formatted summary of daily activity totals.
    """
    if not date:
        date = _today()
    return await _format("summary")


@mcp.tool()
async def get_sleep_summary(date: Optional[str] = None) -> str:
    """
    Returns last night's sleep data: total sleep duration, time in each
    sleep stage (deep, light, REM, awake), and sleep score if available.

    Note: Sleep data is recorded for the PREVIOUS night. Pass yesterday's
    date (or no date) to get the most recent sleep data.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to yesterday if not provided
              (since sleep is recorded for the prior night).

    Returns:
        Formatted sleep breakdown in hours and minutes.
    """
    if not date:
        date = _yesterday()
    return await _format("sleep")


@mcp.tool()
async def get_activities(
    limit: Optional[int] = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns recent workout and activity records: activity type (running,
    cycling, swimming, etc.), duration, distance, pace, calories, and
    heart rate zones for each activity.

    Args:
        limit: Number of activities to return (1-20). Defaults to 5.
        start_date: Optional start date filter in YYYY-MM-DD format.
        end_date: Optional end date filter in YYYY-MM-DD format.

    Returns:
        List of recent activities with key stats for each.
    """
    limit = max(1, min(20, limit or 5))

    if start_date and end_date:
        return await _call("get_activities_by_date", start_date, end_date)

    return await _format("activities", limit)


# ===========================================================================
# GROUP 2 — Recovery & Wellness
# ===========================================================================

@mcp.tool()
async def get_body_battery(date: Optional[str] = None) -> str:
    """
    Returns Garmin Body Battery energy level data: current level (0-100),
    daily high and low, and charge/drain events throughout the day.

    Body Battery reflects overall energy reserves combining sleep quality,
    stress, and activity. High values indicate good recovery.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Body Battery levels and charge/drain events for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_body_battery, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No Body Battery data available for this date."
    return await _format("body_battery")


@mcp.tool()
async def get_stress_summary(date: Optional[str] = None) -> str:
    """
    Returns stress level data: average stress score, max stress,
    and time spent in each stress category (low, medium, high, rest).

    Garmin stress is measured via heart rate variability (0-100 scale).
    Lower scores indicate less physiological stress.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Stress scores and time breakdown across stress categories.
    """
    if not date:
        date = _today()
    return await _format("stress")


@mcp.tool()
async def get_hrv_status(date: Optional[str] = None) -> str:
    """
    Returns Heart Rate Variability (HRV) data: last night's HRV average,
    5-day baseline, and HRV status (balanced, unbalanced, poor).

    HRV is a key recovery indicator — higher values generally indicate
    better nervous system recovery. Best viewed over time as a trend.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        HRV measurements and status classification.
    """
    if not date:
        date = _today()
    return await _format("hrv")


@mcp.tool()
async def get_heart_rate(date: Optional[str] = None) -> str:
    """
    Returns heart rate data for the day: resting heart rate,
    maximum heart rate recorded, and average heart rate.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Resting, average, and max heart rate for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_heart_rate_data, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No heart rate data available for this date."
    import json
    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# GROUP 3 — Training Performance
# ===========================================================================

@mcp.tool()
async def get_training_status() -> str:
    """
    Returns training performance metrics: VO2 Max estimate, fitness age,
    training load (acute vs. chronic), and current training status
    (productive, maintaining, recovery, overreaching, detraining).

    These metrics update after activities and reflect fitness trends
    over the past 4 weeks.

    Returns:
        VO2 Max, fitness age, training load, and training status classification.
    """
    return await _format("training")


@mcp.tool()
async def get_training_readiness(date: Optional[str] = None) -> str:
    """
    Returns today's training readiness score (0-100) and the contributing
    factors: sleep quality, HRV status, recovery time, body battery,
    and recent training load.

    Higher scores indicate your body is well-recovered and ready for
    hard training. Below 50 suggests a lighter day is advisable.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Training readiness score and factor breakdown.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_training_readiness, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No training readiness data available. This metric requires a compatible Garmin device."
    import json
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_intensity_minutes(date: Optional[str] = None) -> str:
    """
    Returns weekly moderate and vigorous intensity activity minutes,
    compared to the WHO-recommended 150 minutes of moderate activity per week.

    Vigorous activity counts double toward the weekly goal.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Weekly intensity minutes breakdown and goal progress percentage.
    """
    if not date:
        date = _today()
    return await _format("intensity")


# ===========================================================================
# GROUP 4 — Nutrition & Hydration
# ===========================================================================

@mcp.tool()
async def get_nutrition_log(date: Optional[str] = None) -> str:
    """
    Returns nutrition data logged in Garmin Connect: total calories consumed,
    macronutrients (protein, carbs, fat, fiber, sugar), and individual
    meal/food entries if logged.

    Note: This data is only populated if the user manually logs food in the
    Garmin Connect app or a connected food tracking service.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Calories consumed, macros breakdown, and food log entries.
    """
    if not date:
        date = _today()
    return await _format("nutrition")


@mcp.tool()
async def get_hydration(date: Optional[str] = None) -> str:
    """
    Returns water intake logged in Garmin Connect for the day,
    in both milliliters and US cups.

    Note: This data is only populated if the user logs water intake
    in the Garmin Connect app.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Total water intake in ml and cups for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_hydration_data, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No hydration data logged for this date. Log water intake in the Garmin Connect app."
    import json
    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# GROUP 5 — Comprehensive & Date-Range
# ===========================================================================

@mcp.tool()
async def get_body_metrics(date: Optional[str] = None) -> str:
    """
    Returns body composition metrics from a compatible Garmin scale:
    weight, BMI, body fat percentage, muscle mass, and bone mass.

    Note: Requires a Garmin Index smart scale or manual body composition
    entry in Garmin Connect.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Body composition measurements for the given date.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_body_composition, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No body composition data for this date. Requires a Garmin Index scale."
    import json
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_spo2_and_respiration(date: Optional[str] = None) -> str:
    """
    Returns blood oxygen saturation (SpO2) and breathing/respiration rate data.

    SpO2 measures the percentage of oxygen in the blood — normal is 95-100%.
    Respiration rate is breaths per minute, which is often lower during sleep.

    Note: Requires a Garmin device with pulse oximeter (e.g. Fenix, Forerunner 945+).

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        SpO2 percentage and respiration rate data for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    spo2, resp = await asyncio.gather(
        loop.run_in_executor(None, handler.get_spo2_data, date),
        loop.run_in_executor(None, handler.get_respiration_data, date),
    )

    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    import json
    return json.dumps(
        {"spo2": spo2, "respiration": resp},
        indent=2, default=str
    )


@mcp.tool()
async def get_activities_by_date_range(start_date: str, end_date: str) -> str:
    """
    Returns all workout activities between two dates — useful for weekly
    or monthly summaries. Each activity includes type, duration, distance,
    pace, calories, and average heart rate.

    Args:
        start_date: Start of date range in YYYY-MM-DD format (e.g. "2026-03-01").
        end_date: End of date range in YYYY-MM-DD format (e.g. "2026-03-13").

    Returns:
        All activities within the specified date range.
    """
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, handler.get_activities_by_date, start_date, end_date
    )
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if not result:
        return f"No activities found between {start_date} and {end_date}."
    import json
    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# GROUP 6 — Gear Tracking
# ===========================================================================

@mcp.tool()
async def get_gear() -> str:
    """
    Returns all gear items registered in Garmin Connect: shoes, bikes,
    and other equipment. Each item shows the gear name, type, status
    (active/inactive), total distance logged, and number of activities.

    Use this to see what equipment is tracked and to get gear UUIDs
    needed for get_gear_stats and get_gear_activities.

    Returns:
        List of all gear with name, type, distance, and activity count.
    """
    import json
    result = await _call("get_gear")
    return result


@mcp.tool()
async def get_gear_stats(gear_uuid: str) -> str:
    """
    Returns usage statistics for a specific piece of gear: total distance,
    total time, and number of activities.

    Useful for tracking shoe mileage (when to replace) or bike maintenance.

    Args:
        gear_uuid: The UUID of the gear item. Get this from get_gear.

    Returns:
        Total distance, time, and activity count for the gear item.
    """
    import json
    result = await _call("get_gear_stats", gear_uuid)
    return result


@mcp.tool()
async def get_gear_activities(gear_uuid: str, limit: Optional[int] = 20) -> str:
    """
    Returns recent activities that used a specific piece of gear.

    Args:
        gear_uuid: The UUID of the gear item. Get this from get_gear.
        limit: Maximum number of activities to return (default 20).

    Returns:
        List of activities using this gear, with date, type, and distance.
    """
    limit = max(1, min(100, limit or 20))
    import json
    result = await _call("get_gear_activities", gear_uuid, limit)
    return result


@mcp.tool()
async def get_activity_gear(activity_id: str) -> str:
    """
    Returns the gear (shoes, bike, etc.) used for a specific activity.

    Args:
        activity_id: The Garmin activity ID. Get this from get_activities
                     or get_activities_by_date_range (the "activityId" field).

    Returns:
        Gear details for the activity, or a message if no gear was logged.
    """
    import json
    result = await _call("get_activity_gear", activity_id)
    return result


@mcp.tool()
async def get_gear_defaults() -> str:
    """
    Returns default gear assignments by activity type — for example,
    which shoes are automatically logged for running activities, or
    which bike is used for cycling.

    Returns:
        List of default gear rules per activity type.
    """
    import json
    result = await _call("get_gear_defaults")
    return result


# ===========================================================================
# GROUP 7 — Activity Details
# ===========================================================================

@mcp.tool()
async def get_activity_details(activity_id: str) -> str:
    """
    Returns the full GPS track and per-second/per-lap metric timeseries
    for a single activity: pace, heart rate, elevation, cadence, power, etc.

    This is the most detailed view of a single activity. Use for post-activity
    analysis or when the user asks about a specific run, ride, or workout.

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Detailed activity metrics and GPS data.
    """
    import json
    result = await _call("get_activity_details", activity_id)
    return result


@mcp.tool()
async def get_activity_splits(activity_id: str) -> str:
    """
    Returns lap/split data for a single activity: distance, duration,
    average pace, average heart rate, and elevation for each lap.

    Useful for analysing pacing strategy or comparing effort across laps.

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Per-lap split metrics for the activity.
    """
    import json
    result = await _call("get_activity_splits", activity_id)
    return result


@mcp.tool()
async def get_activity_hr_zones(activity_id: str) -> str:
    """
    Returns time spent in each heart rate training zone for a single activity.

    Zones are typically: Zone 1 (easy/recovery), Zone 2 (aerobic base),
    Zone 3 (tempo), Zone 4 (threshold), Zone 5 (max/VO2 max).

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Duration in each HR zone for the activity.
    """
    import json
    result = await _call("get_activity_hr_zones", activity_id)
    return result


@mcp.tool()
async def get_activity_power_zones(activity_id: str) -> str:
    """
    Returns time spent in each power training zone for a single cycling activity.

    Power zones are relative to the athlete's FTP (Functional Threshold Power).
    Requires a power meter on the bike.

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Duration in each power zone for the activity, or a message if no power data.
    """
    import json
    result = await _call("get_activity_power_zones", activity_id)
    return result


@mcp.tool()
async def get_activity_exercise_sets(activity_id: str) -> str:
    """
    Returns exercise sets and reps for a strength training activity.

    Shows each exercise performed, number of sets, reps per set,
    and weight used. Only populated for strength/gym activities.

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Exercise sets, reps, and weights for a strength training activity.
    """
    import json
    result = await _call("get_activity_exercise_sets", activity_id)
    return result


@mcp.tool()
async def get_activity_weather(activity_id: str) -> str:
    """
    Returns weather conditions recorded at the start of an activity:
    temperature, humidity, wind speed, and weather description.

    Args:
        activity_id: Garmin activity ID (the "activityId" field from get_activities).

    Returns:
        Weather conditions during the activity.
    """
    import json
    result = await _call("get_activity_weather", activity_id)
    return result


# ===========================================================================
# GROUP 8 — Advanced Performance Metrics
# ===========================================================================

@mcp.tool()
async def get_race_predictions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns Garmin's predicted finish times for common race distances:
    5K, 10K, half marathon, and full marathon.

    Predictions are based on recent training, VO2 Max, and running economy.
    Requires sufficient recent running activity for accuracy.

    Args:
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format.

    Returns:
        Predicted finish times for 5K, 10K, half marathon, and marathon.
    """
    import json
    result = await _call("get_race_predictions", start_date, end_date)
    return result


@mcp.tool()
async def get_endurance_score(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns Garmin's endurance score — a measure of aerobic base fitness
    built up from sustained moderate-to-hard training over time.

    Higher scores indicate greater capacity for long-duration efforts.
    Requires a compatible Garmin device.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 28 days ago.
        end_date: Optional end date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Endurance score and trend data.
    """
    import json
    result = await _call("get_endurance_score", start_date, end_date)
    return result


@mcp.tool()
async def get_hill_score(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns Garmin's hill score — a metric reflecting climbing fitness
    based on power output and heart rate during uphill efforts.

    Higher scores indicate better ability to handle elevation gain.
    Requires a compatible Garmin device.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 28 days ago.
        end_date: Optional end date in YYYY-MM-DD format.

    Returns:
        Hill score and trend data.
    """
    import json
    result = await _call("get_hill_score", start_date, end_date)
    return result


@mcp.tool()
async def get_lactate_threshold(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns lactate threshold data: the heart rate and pace at which
    lactate begins to accumulate rapidly in the blood.

    Training at or near lactate threshold improves race pace.
    Requires a compatible Garmin device with running dynamics or
    a recent Lactate Threshold test activity.

    Args:
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format.

    Returns:
        Lactate threshold heart rate and pace estimate.
    """
    import json
    result = await _call("get_lactate_threshold", start_date, end_date)
    return result


@mcp.tool()
async def get_cycling_ftp() -> str:
    """
    Returns the athlete's Functional Threshold Power (FTP) for cycling —
    the maximum power (in watts) a cyclist can sustain for approximately 1 hour.

    FTP is used to set training zones for power-based cycling workouts.
    Requires a power meter and a compatible Garmin device.

    Returns:
        Current FTP value in watts.
    """
    import json
    result = await _call("get_cycling_ftp")
    return result


@mcp.tool()
async def get_running_tolerance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns running load tolerance data — how well the body is adapting
    to recent running training volume and intensity.

    Helps identify if training load is increasing too quickly (injury risk)
    or if there's room to increase volume.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 28 days ago.
        end_date: End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Running tolerance metrics and weekly aggregation.
    """
    import json
    result = await _call("get_running_tolerance", start_date, end_date)
    return result


@mcp.tool()
async def get_fitness_age(date: Optional[str] = None) -> str:
    """
    Returns Garmin's fitness age estimate — the biological age that
    corresponds to the user's current fitness level.

    A fitness age lower than chronological age indicates above-average
    cardiovascular fitness for that age group.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Fitness age estimate and contributing factors.
    """
    if not date:
        date = _today()
    import json
    result = await _call("get_fitness_age", date)
    return result


# ===========================================================================
# GROUP 9 — Body & Health Tracking
# ===========================================================================

@mcp.tool()
async def get_resting_heart_rate(date: Optional[str] = None) -> str:
    """
    Returns the resting heart rate (RHR) for a specific date.

    RHR is measured during sleep or rest and is a key cardiovascular
    health indicator — lower values generally indicate better fitness.
    Trending RHR upward can be an early sign of illness or overtraining.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Resting heart rate in beats per minute for the day.
    """
    if not date:
        date = _today()
    import json
    result = await _call("get_resting_heart_rate", date)
    return result


@mcp.tool()
async def get_body_battery_events(date: Optional[str] = None) -> str:
    """
    Returns body battery charge and drain events throughout the day:
    what activities or rest periods caused the biggest changes.

    Useful for understanding which activities are most taxing and
    how effectively sleep is recharging energy reserves.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        List of body battery charge/drain events with timestamps and amounts.
    """
    if not date:
        date = _today()
    import json
    result = await _call("get_body_battery_events", date)
    return result


@mcp.tool()
async def get_weigh_ins(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns weight log entries over a date range from a Garmin Index scale
    or manual entries in Garmin Connect.

    Useful for tracking weight trends over time.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 30 days ago.
        end_date: End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Weight entries with date, weight, and optional body composition data.
    """
    if not end_date:
        end_date = _today()
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    import json
    result = await _call("get_weigh_ins", start_date, end_date)
    return result


@mcp.tool()
async def get_blood_pressure(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns blood pressure readings logged in Garmin Connect.

    Note: Requires manual entry in the Garmin Connect app or a compatible
    blood pressure monitor. Not automatically measured by Garmin wearables.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 30 days ago.
        end_date: Optional end date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Blood pressure readings (systolic/diastolic) with timestamps.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    import json
    result = await _call("get_blood_pressure", start_date, end_date)
    return result


# ===========================================================================
# GROUP 10 — Weekly Trends
# ===========================================================================

@mcp.tool()
async def get_weekly_step_trends(
    end_date: Optional[str] = None,
    weeks: Optional[int] = 12,
) -> str:
    """
    Returns weekly step totals over the past N weeks.

    Useful for identifying activity trends, seasonal patterns, or the
    impact of life changes on daily movement.

    Args:
        end_date: Last date of the range in YYYY-MM-DD format. Defaults to today.
        weeks: Number of weeks of history to return (default 12, max 52).

    Returns:
        Weekly step totals and averages.
    """
    if not end_date:
        end_date = _today()
    weeks = max(1, min(52, weeks or 12))
    import json
    result = await _call("get_weekly_steps", end_date, weeks)
    return result


@mcp.tool()
async def get_weekly_stress_trends(
    end_date: Optional[str] = None,
    weeks: Optional[int] = 12,
) -> str:
    """
    Returns weekly average stress scores over the past N weeks.

    Useful for identifying high-stress periods and correlating stress
    with training load, sleep quality, or life events.

    Args:
        end_date: Last date of the range in YYYY-MM-DD format. Defaults to today.
        weeks: Number of weeks of history to return (default 12, max 52).

    Returns:
        Weekly average and peak stress scores.
    """
    if not end_date:
        end_date = _today()
    weeks = max(1, min(52, weeks or 12))
    import json
    result = await _call("get_weekly_stress", end_date, weeks)
    return result


@mcp.tool()
async def get_weekly_intensity_trends(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns weekly moderate and vigorous intensity minutes over a date range.

    Compare against the WHO-recommended 150 min/week moderate or
    75 min/week vigorous activity guideline. Vigorous counts double.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to 12 weeks ago.
        end_date: End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Weekly moderate and vigorous intensity minute totals.
    """
    if not end_date:
        end_date = _today()
    if not start_date:
        start_date = (datetime.now() - timedelta(weeks=12)).strftime("%Y-%m-%d")
    import json
    result = await _call("get_weekly_intensity_minutes", start_date, end_date)
    return result


# ===========================================================================
# GROUP 11 — Goals & Achievements
# ===========================================================================

@mcp.tool()
async def get_personal_records() -> str:
    """
    Returns the athlete's all-time personal records (PRs) across activity types:
    fastest 1K, 5K, 10K, half marathon, marathon, longest run, etc.

    PRs are automatically detected and updated by Garmin Connect after activities.

    Returns:
        All-time personal records with dates achieved.
    """
    import json
    result = await _call("get_personal_records")
    return result


@mcp.tool()
async def get_earned_badges() -> str:
    """
    Returns badges and challenges the user has completed in Garmin Connect:
    step milestones, distance achievements, workout streaks, and more.

    Returns:
        List of earned badges with name, category, and date earned.
    """
    import json
    result = await _call("get_earned_badges")
    return result


# ===========================================================================
# GROUP 12 — Devices
# ===========================================================================

@mcp.tool()
async def get_devices() -> str:
    """
    Returns all Garmin devices connected to the account: watch model,
    firmware version, battery status, and last sync time.

    Useful for understanding which hardware capabilities are available
    (e.g. whether the device supports SpO2, power, solar charging).

    Returns:
        List of connected Garmin devices with model and firmware details.
    """
    import json
    result = await _call("get_devices")
    return result


# ===========================================================================
# GROUP 13 — Nutrition Details
# ===========================================================================

@mcp.tool()
async def get_nutrition_meals(date: Optional[str] = None) -> str:
    """
    Returns a per-meal nutrition breakdown for the day: each meal (breakfast,
    lunch, dinner, snacks) with calories and macros for that meal.

    More granular than get_nutrition_log, which only shows daily totals.

    Note: Requires food logging in the Garmin Connect app.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Per-meal calorie and macro breakdown.
    """
    if not date:
        date = _today()
    import json
    result = await _call("get_nutrition_meals", date)
    return result


# ===========================================================================
# GROUP 14 — Strava Integration
# ===========================================================================

async def _strava_call(method_name: str, *args, **kwargs) -> str:
    """
    Load the current user's StravaApiClient and run the named method in a
    thread executor, matching the pattern of _call() for Garmin.

    Returns a JSON string or a plain text error message.
    """
    import json as _json
    from app.strava_adapter import get_strava_client

    token = _get_token()
    try:
        client = await get_strava_client(token)
    except RuntimeError as exc:
        return str(exc)

    loop = asyncio.get_event_loop()
    method = getattr(client, method_name)
    result = await loop.run_in_executor(None, method, *args, **kwargs)

    if result is None:
        return "No data available."
    if isinstance(result, str):
        return result
    return _json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_strava_athlete() -> str:
    """
    Returns the connected Strava athlete's profile: name, location,
    follower/following counts, weight, measurement preference, and
    sport type breakdown.

    Use this to confirm the Strava connection is active and to get
    the athlete's basic biographical information.

    Returns:
        Athlete profile fields from the Strava API.
    """
    return await _strava_call("get_athlete")


@mcp.tool()
async def get_strava_recent_activities(days: Optional[int] = 14) -> str:
    """
    Returns recent Strava activities (runs, rides, swims, etc.) from
    the past N days, with sport type, date, distance, moving time,
    elapsed time, elevation gain, average heart rate, and average speed/pace.

    Args:
        days: How many days of history to retrieve (default 14, max 90).

    Returns:
        List of recent activities with summary metrics.
    """
    import time as _time
    days = max(1, min(90, days or 14))
    after = int(_time.time()) - days * 86400
    return await _strava_call("get_activities", after=after, per_page=30)


@mcp.tool()
async def get_strava_activity_detail(activity_id: int) -> str:
    """
    Returns detailed data for a single Strava activity: full stats,
    gear used, segment efforts, achievement count, and perceived exertion.

    Args:
        activity_id: Strava activity ID. Get this from get_strava_recent_activities
                     (the "id" field of each activity).

    Returns:
        Complete activity detail including segments and gear.
    """
    return await _strava_call("get_activity", activity_id)


@mcp.tool()
async def get_strava_activity_laps(activity_id: int) -> str:
    """
    Returns lap-by-lap breakdown for a single Strava activity:
    lap number, distance, moving time, average pace, average heart rate,
    and max heart rate for each lap.

    Args:
        activity_id: Strava activity ID.

    Returns:
        Per-lap metrics for the activity.
    """
    return await _strava_call("get_activity_laps", activity_id)


@mcp.tool()
async def get_strava_activity_zones(activity_id: int) -> str:
    """
    Returns heart rate and power zone distribution for a single Strava activity:
    time spent in each zone, distribution percentages, and zone boundaries.

    Args:
        activity_id: Strava activity ID.

    Returns:
        HR and power zone distribution for the activity.
    """
    return await _strava_call("get_activity_zones", activity_id)


@mcp.tool()
async def get_strava_stats() -> str:
    """
    Returns the athlete's all-time and recent (4-week) training statistics
    from Strava, broken down by sport: running, cycling, and swimming.

    Includes total distance, total elevation gain, total moving time,
    and activity count for each period and sport.

    Returns:
        Lifetime and recent totals across all sport types.
    """
    import json as _json
    from app.strava_adapter import get_strava_client

    token = _get_token()
    try:
        client = await get_strava_client(token)
    except RuntimeError as exc:
        return str(exc)

    if not client.athlete_id:
        return (
            "Athlete ID not available. "
            "Please disconnect and reconnect your Strava account."
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, client.get_athlete_stats, client.athlete_id
    )
    if result is None:
        return "No stats available."
    return _json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_strava_zones() -> str:
    """
    Returns the athlete's configured heart rate and power training zones in Strava.

    Zone boundaries are set by the athlete in Strava settings based on
    lactate threshold heart rate or FTP (for power zones).

    Returns:
        Heart rate and power zone boundaries and names.
    """
    return await _strava_call("get_athlete_zones")

