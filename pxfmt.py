"""Price formatting: 3 decimals as before, more for sub-1 prices (e.g. crypto like SHIB 0.0000123)."""
import math


def _dec(v):
    try:
        a = abs(float(v))
    except (TypeError, ValueError):
        return 3
    if not math.isfinite(a) or a == 0 or a >= 0.1:
        return 3
    return min(10, max(3, -math.floor(math.log10(a)) + 2))


def fp(v, comma=False):
    """Format a price: 3 decimals, or enough decimals to show small prices."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(f):
        return "N/A"
    return f"{f:{',' if comma else ''}.{_dec(f)}f}"


def pxf(v):
    """printf-style format for st.column_config number columns, sized for a typical value v."""
    return f"%.{_dec(v)}f"
