"""
Context-aware dynamic crowd risk assessment.
Uses Physical Density (people/m2) as the priority metric if calibrated.
Falls back to Place-Specific absolute count thresholds if uncalibrated.
"""

from __future__ import annotations

# Polus Level-of-Service (LOS) density thresholds (people/m2)
DENSITY_THRESHOLDS = {
    "VERY LOW": 0.60,
    "LOW": 1.00,
    "MODERATE": 1.80,
    "HIGH": 2.50,
}

# Absolute count limits for fallback uncalibrated mode
PLACE_LIMITS = {
    "school":   {"low": 80,  "medium": 150, "high": 250},
    "station":  {"low": 150, "medium": 400, "high": 800},
    "mall":     {"low": 200, "medium": 500, "high": 1000},
    "hospital": {"low": 50,  "medium": 120, "high": 250},
    "stadium":  {"low": 500, "medium": 2000, "high": 5000},
    "office":   {"low": 15,  "medium": 30,  "high": 50},
}

_current_place: str = ""

def set_place(place: str) -> None:
    """Set active place type for uncalibrated fallback."""
    global _current_place
    if place.lower() in PLACE_LIMITS:
        _current_place = place.lower()

def get_place() -> str:
    return _current_place

def get_context_risk(person_count: int, physical_density: float = 0.0, is_calibrated: bool = False) -> str:
    """
    Return risk level based on physical density (Priority) or place counts (Fallback).
    Returns one of: 'VERY LOW', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'.
    """
    if is_calibrated and physical_density > 0:
        # Priority: Area-aware physical density
        if physical_density < DENSITY_THRESHOLDS["VERY LOW"]:
            return "VERY LOW"
        elif physical_density < DENSITY_THRESHOLDS["LOW"]:
            return "LOW"
        elif physical_density < DENSITY_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        elif physical_density < DENSITY_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"
            
    else:
        # Fallback: Absolute counts (if place is set)
        if not _current_place:
            return "" # Cannot compute risk without place or calibration
            
        limits = PLACE_LIMITS[_current_place]
        if person_count < limits["low"] * 0.5:
            return "VERY LOW"
        elif person_count < limits["low"]:
            return "LOW"
        elif person_count < limits["medium"]:
            return "MODERATE"
        elif person_count < limits["high"]:
            return "HIGH"
        else:
            return "CRITICAL"

def get_risk_color_bgr(risk: str) -> tuple[int, int, int]:
    """Map risk string to BGR color tuple for drawing."""
    return {
        "VERY LOW": (80,  220, 100),
        "LOW":      (0,   220, 255),
        "MODERATE": (0,   200, 255),
        "HIGH":     (0,   140, 255),
        "CRITICAL": (60,  60,  255),
    }.get(risk, (200, 200, 200))
