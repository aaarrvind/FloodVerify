"""Create (or update) the private FloodVerify Hugging Face Space. Run on your own computer from the repo root.

    pip install -U huggingface_hub
    hf auth login                      # paste a token with "write" access (huggingface.co/settings/tokens)
    python demo/deploy_space.py --space-id YOUR_HF_USERNAME/floodverify

What it does
    1. Copies app.py, the small part of the floodsitrep package the app imports, and demo/data/ into a clean folder.
    2. Writes the Space README (title, SDK, version) and requirements.txt.
    3. Creates the Space as PRIVATE (unless you pass --public) and uploads the folder.

The data folder can hold xBD imagery (CC BY-NC-SA 4.0), tweet text and guidance passages. These must stay in a
private Space, so the script refuses --public when any of them is present.

--dry-run only builds the folder (default: ./space_build) and prints what would be uploaded.
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"
PKG = REPO / "src" / "floodsitrep"
# The app imports only these modules (plus their __init__ files).
PKG_FILES = ["__init__.py", "common/__init__.py", "common/interfaces.py",
             "nlp/__init__.py", "nlp/schema.py", "nlp/claims.py", "nlp/checker.py"]
PRIVATE_ONLY = ["images", "tweet_examples.json", "advisories.json"]  # data that must not be public


def gradio_version():
    try:
        import gradio
        return gradio.__version__
    except Exception:
        return "6.32.0"


def readme(version):
    return f"""---
title: FloodVerify
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {version}
app_file: app.py
pinned: false
short_description: Fact-check flood damage reports against building-level evidence
---

# FloodVerify demo

CPU-only Gradio app. Private: it can contain xBD imagery (CC BY-NC-SA 4.0), tweet text and guidance passages.
Prototype only, not for operational use.
"""


def stage(out):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(DEMO / "app.py", out / "app.py")
    shutil.copy2(DEMO / "land.json", out / "land.json")
    for rel in PKG_FILES:
        src = PKG / rel
        if not src.exists():
            sys.exit(f"Missing {src}. Run this from a checkout that has src/floodsitrep.")
        dst = out / "floodsitrep" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    data = DEMO / "data"
    if data.exists():
        shutil.copytree(data, out / "data", ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    else:
        print("Warning: demo/data/ not found. The app will start without data (counts only).")
    (out / "README.md").write_text(readme(gradio_version()))
    (out / "requirements.txt").write_text("numpy\npillow\n")  # gradio comes from sdk_version
    return out


def private_items(out):
    return [n for n in PRIVATE_ONLY if (out / "data" / n).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space-id", help="owner/name, for example antaun/floodverify")
    ap.add_argument("--public", action="store_true", help="make the Space public (refused if private data is present)")
    ap.add_argument("--dry-run", action="store_true", help="build the folder only, do not upload")
    ap.add_argument("--stage-dir", default="space_build", help="folder to build (default: space_build)")
    a = ap.parse_args()

    out = stage(Path(a.stage_dir).resolve())
    files = sorted(p for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files) / 1e6
    held = private_items(out)
    print(f"Built {out}: {len(files)} files, {size:.1f} MB")
    if held:
        print("Private data present:", ", ".join(held))
    if a.dry_run:
        print("Dry run: nothing uploaded.")
        return
    if not a.space_id or "/" not in a.space_id:
        sys.exit("Give --space-id owner/name (for example antaun/floodverify).")
    if a.public and held:
        sys.exit("Refusing to make a public Space: it holds " + ", ".join(held) + ". Drop --public.")

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]
    print("Signed in as", who)
    api.create_repo(a.space_id, repo_type="space", space_sdk="gradio", private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=a.space_id, repo_type="space",
                      commit_message="Update FloodVerify demo")
    info = api.space_info(a.space_id)
    print("Space:", f"https://huggingface.co/spaces/{a.space_id}", "| private:", info.private)
    print("It builds for a few minutes. Open the link, then Settings to share it with named people if needed.")


if __name__ == "__main__":
    main()
