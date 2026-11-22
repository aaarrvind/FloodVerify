"""Tweet statistics and example tweets for the demo. Owner: member 5.

Run in notebook 08 after the tweets are scored (the `scored` DataFrame and the settings THETA, ACTIONABLE):

    import sys; sys.path.insert(0, "<repo>/demo")
    from export_tweet_stats import export_tweet_stats
    export_tweet_stats(scored, ACTIONABLE, THETA, f"{OUT}/tweet_stats.json")

Then pass that file to build_bundle.py with --tweet-stats. The file holds counts per event, per day and per class
only, so it carries no account names, links or tweet text.

For the tweet text itself (private Space only), also run:

    from export_tweet_stats import export_tweet_examples
    export_tweet_examples(scored, THETA, f"{OUT}/tweet_examples.json")

and pass that file to build_bundle.py with --tweet-examples. Account names and links are removed from the text.
"""
import json
import re


def export_tweet_stats(scored, actionable, theta, out_path):
    """scored: DataFrame with event, predicted_class, score, posted (UTC datetime). Writes a JSON file."""
    events = {}
    for event, d in scored.groupby("event"):
        d = d.dropna(subset=["posted"]).copy()
        d["day"] = d["posted"].dt.strftime("%Y-%m-%d")
        d["flagged"] = d["score"] >= theta
        daily = {day: [int(len(g)), int(g["flagged"].sum())] for day, g in d.groupby("day")}
        events[event] = {
            "n": int(len(d)), "flagged": int(d["flagged"].sum()),
            "first": min(daily), "last": max(daily), "daily": daily,
            "classes": {str(k): int(v) for k, v in d["predicted_class"].value_counts().items()},
        }
    payload = {"theta": float(theta), "actionable": list(actionable), "events": events}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"{len(events)} events written to {out_path}")
    return payload


def clean_tweet(text):
    """Remove links and @names so the example carries no account."""
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def export_tweet_examples(scored, theta, out_path, per_event=60, text_col=None):
    """Highest-scoring tweets per event, with their text. scored needs event, predicted_class, score, posted and a
    text column (text_col, or the first of tweet_text / text / full_text)."""
    text_col = text_col or next(c for c in ("tweet_text", "text", "full_text") if c in scored.columns)
    out = {}
    for event, d in scored.groupby("event"):
        d = d[d["score"] >= theta].sort_values("score", ascending=False)
        rows, seen = [], set()
        for _, r in d.iterrows():
            t = clean_tweet(r[text_col])
            if len(t) < 25 or t.lower() in seen:
                continue
            seen.add(t.lower())
            day = r["posted"].strftime("%Y-%m-%d") if hasattr(r["posted"], "strftime") else ""
            rows.append({"text": t, "class": str(r["predicted_class"]), "score": round(float(r["score"]), 3), "day": day})
            if len(rows) >= per_event:
                break
        out[event] = rows
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{sum(len(v) for v in out.values())} example tweets for {len(out)} events written to {out_path}")
    return out
