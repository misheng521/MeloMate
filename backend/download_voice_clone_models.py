from pathlib import Path
import argparse
import sys


BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from open_llm_vtuber.utils.model_downloads import ensure_voice_clone_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only verify an already extracted local model cache",
    )
    args = parser.parse_args()
    snapshots = ensure_voice_clone_models(download_missing=not args.local_only)
    for repo_id, snapshot in snapshots.items():
        print(f"Verified {repo_id}: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
