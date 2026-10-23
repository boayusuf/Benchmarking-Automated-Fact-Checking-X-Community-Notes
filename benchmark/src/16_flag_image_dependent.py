"""
Step 16: Use Claude Haiku 4.5 to flag image-dependent claims in
extracted_claims.csv. Switched from gpt-4o-mini -> Haiku on 2026-05-01 to
avoid the OpenAI mini RPD wall.

Image-dependent claims are ones that can only be evaluated by inspecting an
attached image / video / photo / chart — text-based AFC systems cannot verify
them. Flagging lets us recompute metrics with image-grounded claims excluded
(step 17).

- Reads `data/filtered/extracted_claims.csv`
- Adds boolean `image_dependent` column (and the model's reason).
- Saves back to `data/filtered/extracted_claims.csv`.
- Resume-aware: skips noteIds already flagged on rerun.
"""

from pathlib import Path
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr); raise SystemExit(1)
client = anthropic.Anthropic()

MODEL_NAME = "claude-haiku-4-5"
SLEEP_BETWEEN = 0.2
CHECKPOINT_EVERY = 50
QUOTA_BAIL_THRESHOLD = 10
MAX_OUTPUT_TOKENS = 200

PROMPT = """A claim is "image-dependent" if its truth value can only be determined by looking at an attached image, video, photo, screenshot, chart, or other visual media. Examples:
  IMAGE-DEPENDENT: "The image shows a dolphin swimming." / "This video is from the 2020 election." / "The chart depicts US GDP growth."
  NOT IMAGE-DEPENDENT: "The Empire State Building is in New York." / "Vaccines do not cause autism." / "Russia invaded Ukraine in 2022."

Claim: "{claim}"

Reply with image_dependent=true or false and a one-sentence reason."""


class ImageDependentResult(BaseModel):
    image_dependent: bool
    reason: str


RETRYABLE = (anthropic.RateLimitError, anthropic.APIConnectionError,
             anthropic.APITimeoutError, anthropic.InternalServerError)


def classify_with_retry(claim: str, max_attempts: int = 5) -> dict:
    for attempt in range(max_attempts):
        try:
            resp = client.messages.parse(
                model=MODEL_NAME,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": PROMPT.format(claim=claim)}],
                output_format=ImageDependentResult,
            )
            r = resp.parsed_output
            return {"image_dependent": r.image_dependent, "reason": r.reason}
        except RETRYABLE:
            wait = (2 ** attempt) * 5
            time.sleep(wait)
        except Exception as e:
            return {"image_dependent": None, "reason": f"error: {type(e).__name__}: {str(e)[:100]}"}
    return {"image_dependent": None, "reason": "exhausted retries"}


def main() -> int:
    src_path = PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv"
    df = pd.read_csv(src_path, dtype={"noteId": str, "tweetId": str})
    print(f"  loaded {len(df)} claims", flush=True)

    if "image_dependent" not in df.columns:
        df["image_dependent"] = pd.NA
        df["image_dependent_reason"] = pd.NA
    todo = df[df["image_dependent"].isna()].index
    print(f"  to flag: {len(todo)} (resume mode)", flush=True)

    pbar = tqdm(total=len(todo), desc="flag-img", file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0

    for idx in todo:
        claim = str(df.at[idx, "extracted_claim"])
        result = classify_with_retry(claim)
        if result["image_dependent"] is None:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        df.at[idx, "image_dependent"] = result["image_dependent"]
        df.at[idx, "image_dependent_reason"] = result["reason"]
        n_since_save += 1
        pbar.update(1)
        if n_since_save >= CHECKPOINT_EVERY:
            df.to_csv(src_path, index=False)
            n_since_save = 0
        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print("\n  BAIL: too many consecutive failures. Saving and exiting.", flush=True)
            df.to_csv(src_path, index=False)
            pbar.close()
            raise SystemExit(2)
        time.sleep(SLEEP_BETWEEN)

    df.to_csv(src_path, index=False)
    pbar.close()
    n_img = (df["image_dependent"] == True).sum()
    n_text = (df["image_dependent"] == False).sum()
    n_unknown = df["image_dependent"].isna().sum()
    print(f"\n  flagged: image-dependent={n_img}, text-only={n_text}, unknown={n_unknown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
