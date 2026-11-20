"""Small user preferences that should survive closing the browser: watchlist, favourites and saved screens.

Saved on the app server (data/prefs.json) and mirrored to this browser's local storage, so they come back
after the server restarts (Streamlit Cloud wipes server files on restart/redeploy)."""
import json
import os

import streamlit as st

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prefs.json")
LS_KEY = "ssa_prefs_v1"
DEFAULT_WATCH = ["1155.KL", "5347.KL", "5398.KL", "1023.KL", "AAPL", "NVDA", "TSLA", "NFLX"]


def _snapshot():
    return {"watchlist": list(st.session_state.get("term_stocks", [])),
            "favourites": sorted(set(st.session_state.get("term_fav", []))),
            "screens": dict(st.session_state.get("scr_saved", {}))}


def _apply(data):
    wl = [t for t in data.get("watchlist", []) if isinstance(t, str)]
    st.session_state["term_stocks"] = wl or DEFAULT_WATCH.copy()
    st.session_state["term_fav"] = [t for t in data.get("favourites", []) if isinstance(t, str)]
    st.session_state["scr_saved"] = data.get("screens", {}) if isinstance(data.get("screens"), dict) else {}
    st.session_state["_prefs_saved"] = json.dumps(_snapshot())


def load():
    """Call once per run, before any page uses the watchlist."""
    if "term_stocks" in st.session_state and "term_fav" in st.session_state:
        return
    try:
        with open(STORE) as f:
            _apply(json.load(f))
        st.session_state["_prefs_from_disk"] = True
    except Exception:
        st.session_state.setdefault("term_stocks", DEFAULT_WATCH.copy())
        st.session_state.setdefault("term_fav", [])
        st.session_state["_prefs_saved"] = json.dumps(_snapshot())
        st.session_state["_prefs_from_disk"] = False


def sync():
    """Call at the end of each run: save changes, and restore from the browser if the server copy was lost."""
    snap = json.dumps(_snapshot())
    if streamlit_js_eval is not None and not st.session_state.get("_prefs_from_disk") \
            and not st.session_state.get("_prefs_ls_done"):
        raw = streamlit_js_eval(js_expressions=f"localStorage.getItem('{LS_KEY}') || ''", key="prefs_ls_get")
        if raw is not None:
            st.session_state["_prefs_ls_done"] = True
            try:
                saved = json.loads(raw) if raw else None
            except Exception:
                saved = None
            untouched = snap == st.session_state.get("_prefs_saved")
            if isinstance(saved, dict) and saved.get("watchlist") and untouched:
                _apply(saved)
                _write(json.dumps(_snapshot()))
                st.rerun()
    if snap != st.session_state.get("_prefs_saved"):
        _write(snap)
        st.session_state["_prefs_saved"] = snap
        st.session_state["_prefs_from_disk"] = True
        if streamlit_js_eval is not None:
            streamlit_js_eval(js_expressions=f"localStorage.setItem('{LS_KEY}', {json.dumps(snap)})",
                              key=f"prefs_ls_set_{abs(hash(snap)) % 10**9}")


def _write(snap):
    try:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        tmp = STORE + ".tmp"
        with open(tmp, "w") as f:
            f.write(snap)
        os.replace(tmp, STORE)
    except Exception:
        pass
