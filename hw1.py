#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    # The prompt asks only for values printed on the paper, never for a total the
    # model worked out, so the arithmetic happens in Python where it can be
    # checked. Braces are doubled because this template uses f-string syntax.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a meticulous receipt auditor. You are shown exactly one \
supermarket receipt photo, which may be rotated; turn it in your head and read it if needed.

Transcribe the paper evidence into this JSON object and reply with the JSON object ONLY - \
no markdown code fence, no commentary, no text before or after it:

{{
  "subtotal_after_discounts_before_rounding": <number>,
  "discount_lines": [{{"label": "<printed text of the line>", "amount": <positive number>}}],
  "discount_total": <number>,
  "amount_paid_after_rounding": <number>,
  "amount_without_discounts": <number>
}}

Field rules:
- "subtotal_after_discounts_before_rounding": copy the printed subtotal line (Chinese 小計, \
or SUBTOTAL / 總額). This is the amount AFTER the discounts and BEFORE the ROUNDING line.
- "discount_lines": EVERY printed line that reduces the bill, each with its discount amount \
written as a POSITIVE number, in the order printed. This covers packaging-damage lines \
(包裝變形), coupons (COUPON / 優惠券), member prices (MB PRICE / MB $Xoff / 會員), app offers \
(APP upgrade / App Upgrad$), bundle promotions (Buy 2 Save $, Buy 3 Save $), and percentage \
discounts (5% OFF, 10% OFF). A discount printed as a negative number such as -12.40 or \
-$9.99 is reported as positive.
- "discount_total": add up your own "discount_lines" amounts and put that total here.
- "amount_paid_after_rounding": the final amount actually charged, i.e. the printed OCTOPUS \
/ VISA / payment line or the 餘額 line, read after the ROUNDING line has been applied.
- "amount_without_discounts": the bill as it would have been with no discounts at all, which \
is "subtotal_after_discounts_before_rounding" plus "discount_total". The ROUNDING line is \
part of the discount calculation only in the sense that it is already inside the subtotal; \
do not add rounding on top of this figure.

Important exclusions - these are NOT discounts:
- the ROUNDING line itself,
- any plastic-bag surcharge (PLASTIC BAG CHARGIN, e.g. $1.00),
- money paid, change (找續/CHANGE), points balances, or card numbers.

The most important rule - do not copy numbers out of label text:
- A promotion LABEL embeds a number that is NOT the discount amount. Labels such as \
"Buy 2 Save $5", "Buy 3 Save $9.8" or "App Upgrad$30->$20" only describe the offer.
- The discount is the amount printed in the MONEY COLUMN on the far right of that line.
- Read the money column first, then the label. Never copy the label number into the amount.
- Example of the trap: the line "Buy 2 Save $5" can carry a money column amount of -$6.00. \
The correct answer for that line is 6.00, not 5.00.
- A COUPON line may print $0.00 in the money column. That is not an error: report 0.00 when \
that is what the money column really shows, and never invent an amount for it.

Rules:
1. Copy every digit character by character from the image. Never estimate and never round.
2. Only "discount_total" and "amount_without_discounts" may be worked out by adding up the \
values above; every other field is copied from the paper exactly as printed.
3. If a value is genuinely absent from the receipt, use 0 for numbers and [] for lists.
4. Reply with the JSON object and nothing else.""",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Audit receipt {receipt_label} and return its JSON record.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=3,
        timeout=120,
    )
    return prompt | llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.

    The chain returns the four figures of one receipt in the same shape as
    ``ground_truth.json``, so this function only has to add them across receipts:
    ``amount_paid_after_rounding`` answers the first query and
    ``amount_without_discounts`` answers the second.
    """
    amounts: dict[str, Decimal] = {QUERY_1: Decimal("0.00"), QUERY_2: Decimal("0.00")}
    unreadable: list[str] = []
    required = (
        "subtotal_after_discounts_before_rounding",
        "discount_total",
        "amount_paid_after_rounding",
        "amount_without_discounts",
    )

    payloads = [
        {
            "image_url": image_data_url(path),
            "receipt_label": f"{index} of {len(images)} ({path.name})",
        }
        for index, path in enumerate(images, start=1)
    ]
    for path, reply in zip(images, chain.batch(payloads)):
        text = response_text(reply)
        start = text.find("{")
        record = None
        if start != -1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                record = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                record = None
        if record is None:
            unreadable.append(f"{path.name} (reply was not JSON)")
            continue

        # A figure only counts if it really is a number; anything else would
        # silently poison the sum.
        missing = [key for key in required if record.get(key) is None]
        if missing:
            unreadable.append(f"{path.name} (missing {', '.join(missing)})")
            continue
        values = [Decimal(str(record[key])).quantize(Decimal("0.01")) for key in required]
        subtotal, discount_total, paid, without = values

        # These two lines describe every receipt in this format, so a record
        # that breaks them was misread and is not safe to add up.
        issues = []
        if subtotal <= 0:
            issues.append("subtotal is not positive")
        if paid <= 0:
            issues.append("amount_paid_after_rounding is not positive")
        elif (subtotal - paid).copy_abs() > Decimal("1.00"):
            issues.append("subtotal and payment differ by more than rounding")
        if discount_total < 0:
            issues.append("discount_total is negative")
        if without < subtotal and without < paid:
            issues.append("amount_without_discounts is below both other figures")

        if issues:
            unreadable.append(f"{path.name} ({'; '.join(issues)})")
            continue

        amounts[QUERY_1] += paid
        amounts[QUERY_2] += without

    if unreadable:
        # A partial sum would be scored as one wrong amount anyway, so report the
        # failure instead of a number that is quietly missing receipts.
        note = "unreadable receipt(s): " + ", ".join(unreadable)
        return {QUERY_1: note, QUERY_2: note}

    # The response must parse to exactly one amount, so it is the bare figure:
    # extra prose, or even a trailing period, would hide it from the scorer.
    return {QUERY_1: f"HK${amounts[QUERY_1]:.2f}", QUERY_2: f"HK${amounts[QUERY_2]:.2f}"}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
