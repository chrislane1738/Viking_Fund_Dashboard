-- ============================================================
-- Viking Fund Dashboard — Supabase Schema Migration
-- Run this in the Supabase SQL Editor (https://app.supabase.com)
-- ============================================================

-- ── 1. Profiles ─────────────────────────────────────────────
CREATE TABLE public.profiles (
    id         UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email      TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name  TEXT NOT NULL,
    student_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

-- ── 2. Watchlists ───────────────────────────────────────────
CREATE TABLE public.watchlists (
    id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id  UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker   TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes    TEXT,
    UNIQUE(user_id, ticker)
);

ALTER TABLE public.watchlists ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own watchlist"
    ON public.watchlists FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own watchlist"
    ON public.watchlists FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own watchlist"
    ON public.watchlists FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own watchlist"
    ON public.watchlists FOR DELETE
    USING (auth.uid() = user_id);

CREATE INDEX idx_watchlists_user_id ON public.watchlists(user_id);
CREATE INDEX idx_watchlists_ticker ON public.watchlists(ticker);

-- ── 3. Research Notes ───────────────────────────────────────
CREATE TABLE public.research_notes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    title         TEXT NOT NULL,
    content       TEXT,
    is_bullish    BOOLEAN,        -- TRUE = bullish, FALSE = bearish, NULL = neutral
    price_at_note DECIMAL(12,2),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.research_notes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own notes"
    ON public.research_notes FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own notes"
    ON public.research_notes FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own notes"
    ON public.research_notes FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own notes"
    ON public.research_notes FOR DELETE
    USING (auth.uid() = user_id);

CREATE INDEX idx_research_notes_user_id ON public.research_notes(user_id);
CREATE INDEX idx_research_notes_ticker ON public.research_notes(ticker);
CREATE INDEX idx_research_notes_created ON public.research_notes(created_at DESC);

-- ── 4. User Activity ────────────────────────────────────────
CREATE TABLE public.user_activity (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL,
    ticker        TEXT,
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.user_activity ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own activity"
    ON public.user_activity FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own activity"
    ON public.user_activity FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE INDEX idx_user_activity_user_id ON public.user_activity(user_id);
CREATE INDEX idx_user_activity_created ON public.user_activity(created_at DESC);

-- ── 5. Auto-create profile on signup ────────────────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, first_name, last_name, student_id)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data ->> 'first_name', ''),
        COALESCE(NEW.raw_user_meta_data ->> 'last_name', ''),
        COALESCE(NEW.raw_user_meta_data ->> 'student_id', '')
    );
    RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ── 6. Auto-update updated_at columns ───────────────────────
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER set_research_notes_updated_at
    BEFORE UPDATE ON public.research_notes
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ============================================================
-- MIGRATION: Watchlist Groups (run after initial schema)
-- ============================================================

-- ── 7. Watchlist Groups ───────────────────────────────────────
CREATE TABLE public.watchlist_groups (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id   UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name      TEXT NOT NULL DEFAULT 'General',
    position  SMALLINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, name),
    UNIQUE(user_id, position),
    CONSTRAINT position_range CHECK (position >= 0 AND position <= 2)
);

ALTER TABLE public.watchlist_groups ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own groups"
    ON public.watchlist_groups FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own groups"
    ON public.watchlist_groups FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own groups"
    ON public.watchlist_groups FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own groups"
    ON public.watchlist_groups FOR DELETE
    USING (auth.uid() = user_id);

CREATE INDEX idx_watchlist_groups_user_id ON public.watchlist_groups(user_id);

-- ── 8. Migrate watchlists to use groups ───────────────────────

-- 8a. Add nullable group_id column
ALTER TABLE public.watchlists
    ADD COLUMN group_id UUID REFERENCES public.watchlist_groups(id) ON DELETE CASCADE;

-- 8b. Create a "General" group (position 0) for each user with existing watchlist entries
INSERT INTO public.watchlist_groups (user_id, name, position)
SELECT DISTINCT user_id, 'General', 0
FROM public.watchlists
ON CONFLICT DO NOTHING;

-- 8c. Backfill existing watchlist rows with their user's "General" group
UPDATE public.watchlists w
SET group_id = g.id
FROM public.watchlist_groups g
WHERE g.user_id = w.user_id
  AND g.name = 'General'
  AND w.group_id IS NULL;

-- 8d. Make group_id NOT NULL
ALTER TABLE public.watchlists
    ALTER COLUMN group_id SET NOT NULL;

-- 8e. Drop old unique constraint, add new one scoped to group
ALTER TABLE public.watchlists
    DROP CONSTRAINT watchlists_user_id_ticker_key;

ALTER TABLE public.watchlists
    ADD CONSTRAINT watchlists_group_id_ticker_key UNIQUE(group_id, ticker);

-- 8f. Index on group_id
CREATE INDEX idx_watchlists_group_id ON public.watchlists(group_id);

-- ============================================================
-- 9. Comparison Groups (valuation multiples saved sets)
-- ============================================================
CREATE TABLE public.comparison_groups (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    tickers    TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, name)
);

ALTER TABLE public.comparison_groups ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own comparison groups"
    ON public.comparison_groups FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own comparison groups"
    ON public.comparison_groups FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own comparison groups"
    ON public.comparison_groups FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own comparison groups"
    ON public.comparison_groups FOR DELETE
    USING (auth.uid() = user_id);

CREATE INDEX idx_comparison_groups_user_id ON public.comparison_groups(user_id);
