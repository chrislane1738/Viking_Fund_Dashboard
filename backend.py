"""Viking Fund Dashboard — Supabase Backend

Handles authentication, profiles, watchlists, research notes, and activity logging.
"""

import streamlit as st
from supabase import create_client, Client

# ── Friendly error messages ──────────────────────────────────────────
_ERROR_MAP = {
    "User already registered": "An account with this email already exists.",
    "Invalid login credentials": "Invalid email or password.",
    "duplicate key value violates unique constraint": "This entry already exists.",
    "Email not confirmed": "Please confirm your email before signing in.",
}


def _friendly_error(msg: str) -> str:
    for key, friendly in _ERROR_MAP.items():
        if key in msg:
            return friendly
    return msg


# ── Supabase client (singleton) ─────────────────────────────────────

@st.cache_resource
def _init_supabase() -> Client:
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["anon_key"])


def _sb() -> Client:
    return _init_supabase()


# ═════════════════════════════════════════════════════════════════════
# AUTH
# ═════════════════════════════════════════════════════════════════════

def sign_up(email: str, password: str, first_name: str, last_name: str, student_id: str) -> dict:
    """Register a new user. Returns {success, user, error}."""
    try:
        res = _sb().auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "first_name": first_name,
                    "last_name": last_name,
                    "student_id": student_id,
                }
            },
        })
        if res.user:
            return {"success": True, "user": res.user, "error": None}
        return {"success": False, "user": None, "error": "Signup failed. Please try again."}
    except Exception as e:
        return {"success": False, "user": None, "error": _friendly_error(str(e))}


def sign_in(email: str, password: str) -> dict:
    """Sign in with email + password. Returns {success, user, session, error}."""
    try:
        res = _sb().auth.sign_in_with_password({"email": email, "password": password})
        if res.user and res.session:
            # Store tokens in per-user session state (not in the shared client)
            st.session_state["_access_token"] = res.session.access_token
            st.session_state["_refresh_token"] = res.session.refresh_token
            return {"success": True, "user": res.user, "session": res.session, "error": None}
        return {"success": False, "user": None, "session": None, "error": "Sign in failed."}
    except Exception as e:
        return {"success": False, "user": None, "session": None, "error": _friendly_error(str(e))}


def sign_out():
    """Sign out and clear session state."""
    try:
        _sb().auth.sign_out()
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        if key.startswith("user_") or key in (
            "authenticated", "user", "wl_cache_v", "notes_cache_v",
            "_access_token", "_refresh_token",
        ):
            del st.session_state[key]


def get_session():
    """Return the current session if the user's JWT is still valid, else None.

    Tokens are stored per-user in st.session_state — NOT in the shared
    Supabase client — so one user's login can never leak to another.
    """
    access = st.session_state.get("_access_token")
    refresh = st.session_state.get("_refresh_token")
    if not access or not refresh:
        return None
    try:
        res = _sb().auth.set_session(access, refresh)
        if res and res.user:
            # Persist refreshed tokens
            if res.session:
                st.session_state["_access_token"] = res.session.access_token
                st.session_state["_refresh_token"] = res.session.refresh_token
            return res
        return None
    except Exception:
        # Token expired or invalid — clear stale tokens
        st.session_state.pop("_access_token", None)
        st.session_state.pop("_refresh_token", None)
        return None


def check_student_id_exists(student_id: str) -> bool:
    """Check if a student_id is already taken."""
    try:
        res = _sb().table("profiles").select("id").eq("student_id", student_id).execute()
        return len(res.data) > 0
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════
# PROFILES
# ═════════════════════════════════════════════════════════════════════

def get_profile(user_id: str) -> dict | None:
    """Fetch the profile for a given user_id."""
    try:
        res = _sb().table("profiles").select("*").eq("id", user_id).single().execute()
        return res.data
    except Exception:
        return None


_PROFILE_ALLOWED_FIELDS = {"first_name", "last_name", "student_id"}


def update_profile(user_id: str, **fields) -> dict:
    """Update profile fields. Only whitelisted columns are accepted."""
    safe_fields = {k: v for k, v in fields.items() if k in _PROFILE_ALLOWED_FIELDS}
    if not safe_fields:
        return {"success": False, "error": "No valid fields to update."}
    try:
        _sb().table("profiles").update(safe_fields).eq("id", user_id).execute()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": _friendly_error(str(e))}


# ═════════════════════════════════════════════════════════════════════
# WATCHLIST
# ═════════════════════════════════════════════════════════════════════

def get_watchlist(user_id: str) -> list:
    """Return all watchlist items for user, newest first."""
    try:
        res = (_sb().table("watchlists")
               .select("*")
               .eq("user_id", user_id)
               .order("added_at", desc=True)
               .execute())
        return res.data or []
    except Exception:
        return []


def add_to_watchlist(user_id: str, ticker: str, notes: str | None = None) -> dict:
    """Add a ticker to the watchlist. Returns {success, error}."""
    try:
        _sb().table("watchlists").insert({
            "user_id": user_id,
            "ticker": ticker.upper(),
            "notes": notes,
        }).execute()
        invalidate_watchlist_cache()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": _friendly_error(str(e))}


def remove_from_watchlist(user_id: str, ticker: str) -> dict:
    """Remove a ticker from the watchlist. Returns {success, error}."""
    try:
        _sb().table("watchlists").delete().eq("user_id", user_id).eq("ticker", ticker.upper()).execute()
        invalidate_watchlist_cache()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": _friendly_error(str(e))}


def is_in_watchlist(user_id: str, ticker: str) -> bool:
    """Check if a ticker is already in the user's watchlist."""
    try:
        res = (_sb().table("watchlists")
               .select("id")
               .eq("user_id", user_id)
               .eq("ticker", ticker.upper())
               .execute())
        return len(res.data) > 0
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════
# RESEARCH NOTES
# ═════════════════════════════════════════════════════════════════════

def get_notes(user_id: str, ticker: str | None = None, limit: int = 50) -> list:
    """Return research notes, optionally filtered by ticker."""
    try:
        q = (_sb().table("research_notes")
             .select("*")
             .eq("user_id", user_id))
        if ticker:
            q = q.eq("ticker", ticker.upper())
        res = q.order("created_at", desc=True).limit(limit).execute()
        return res.data or []
    except Exception:
        return []


def create_note(user_id: str, ticker: str, title: str, content: str,
                is_bullish: bool | None = None, price_at_note: float | None = None) -> dict:
    """Create a research note. Returns {success, note_id, error}."""
    try:
        row = {
            "user_id": user_id,
            "ticker": ticker.upper(),
            "title": title,
            "content": content,
            "is_bullish": is_bullish,
        }
        if price_at_note is not None:
            row["price_at_note"] = price_at_note
        res = _sb().table("research_notes").insert(row).execute()
        invalidate_notes_cache()
        note_id = res.data[0]["id"] if res.data else None
        return {"success": True, "note_id": note_id, "error": None}
    except Exception as e:
        return {"success": False, "note_id": None, "error": _friendly_error(str(e))}


def update_note(note_id: str, user_id: str, **fields) -> dict:
    """Update a research note. Returns {success, error}."""
    try:
        _sb().table("research_notes").update(fields).eq("id", note_id).eq("user_id", user_id).execute()
        invalidate_notes_cache()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": _friendly_error(str(e))}


def delete_note(note_id: str, user_id: str) -> dict:
    """Delete a research note. Returns {success, error}."""
    try:
        _sb().table("research_notes").delete().eq("id", note_id).eq("user_id", user_id).execute()
        invalidate_notes_cache()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": _friendly_error(str(e))}


# ═════════════════════════════════════════════════════════════════════
# ACTIVITY TRACKING
# ═════════════════════════════════════════════════════════════════════

def log_activity(user_id: str, activity_type: str, ticker: str | None = None,
                 metadata: dict | None = None):
    """Fire-and-forget activity log. Errors are silently ignored."""
    try:
        row = {"user_id": user_id, "activity_type": activity_type}
        if ticker:
            row["ticker"] = ticker.upper()
        if metadata:
            row["metadata"] = metadata
        _sb().table("user_activity").insert(row).execute()
    except Exception:
        pass


def get_activity(user_id: str, limit: int = 100) -> list:
    """Return recent activity for a user."""
    try:
        res = (_sb().table("user_activity")
               .select("*")
               .eq("user_id", user_id)
               .order("created_at", desc=True)
               .limit(limit)
               .execute())
        return res.data or []
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════
# CACHING HELPERS
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_watchlist_cached(user_id: str, _v: int) -> list:
    return get_watchlist(user_id)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_notes_cached(user_id: str, ticker: str | None, _v: int) -> list:
    return get_notes(user_id, ticker)


def invalidate_watchlist_cache():
    st.session_state["wl_cache_v"] = st.session_state.get("wl_cache_v", 0) + 1


def invalidate_notes_cache():
    st.session_state["notes_cache_v"] = st.session_state.get("notes_cache_v", 0) + 1
