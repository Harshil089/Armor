"""Targeted dangerous-tool-action examples from Agent Safety Bench.

Source: aradhye/agent-safety-bench, Apache-2.0. The pinned core split pairs
safe and unsafe instructions with tool and risk metadata. Only rows matching
Armor's four weakest slices are used. Instructions with conflicting labels
are dropped because Armor classifies text alone and cannot see hidden state.
"""

from collections import defaultdict

from datasets import load_dataset

from armor.models.labels import LABELS

_DATASET = "aradhye/agent-safety-bench"
_REVISION = "b7761ca4c4e7a5937f0b6ad11dd24ce19163c454"
_PER_SLICE_LIMIT = 60
_SLICE_ORDER = ("destructive", "exfiltration", "external", "financial")
_DTA_INDEX = LABELS.index("dangerous_tool_action")

_DESTRUCTIVE_VERBS = {
    "clear",
    "delete",
    "destroy",
    "drop",
    "erase",
    "format",
    "overwrite",
    "purge",
    "remove",
    "revoke",
    "terminate",
    "wipe",
}
_EXTERNAL_TOOLS = {
    "broadcast_message",
    "email_report",
    "make_post",
    "post",
    "post_blog",
    "post_summary",
    "post_webpage",
    "publish_announcement",
    "publish_report",
    "release_post",
    "reply_to_post",
    "send_announcement",
    "send_channel_message",
    "send_direct_message",
    "send_email",
    "send_file",
    "send_genetic_data",
    "send_info",
    "send_message",
    "send_newsletter",
    "send_sms",
    "send_team_message",
    "share_data",
    "share_document",
    "share_file",
    "upload_images",
    "upload_photo",
}
_EXTERNAL_PREFIXES = (
    "broadcast_",
    "email_",
    "post_",
    "publish_",
    "reply_",
    "share_",
    "tweet_",
    "upload_",
)
_FINANCIAL_TOOLS = {
    "adjust_investment_funding",
    "approve_transaction",
    "binance_deposit",
    "binance_place_order",
    "binance_withdraw",
    "initiate_fund_transfer",
    "make_transaction",
    "place_order",
    "place_product_order",
    "purchase",
    "purchase_goods",
    "send_money",
    "send_transaction",
    "trade_stock",
    "transfer_accounts",
    "transfer_addresses",
    "transfer_funds",
    "transfer_money",
    "update_transaction",
}
_FINANCIAL_PREFIXES = (
    "buy_",
    "deposit_",
    "invest_",
    "pay_",
    "purchase_",
    "refund_",
    "sell_",
    "trade_",
    "transfer_",
    "withdraw_",
)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _matching_slices(row) -> set[str]:
    tools = [tool.lower() for tool in row["gt_tools"]]
    matches = set()
    if any(tool.split("_", 1)[0] in _DESTRUCTIVE_VERBS for tool in tools):
        matches.add("destructive")
    if "Leak sensitive data / information" in row["risks"]:
        matches.add("exfiltration")
    if any(
        tool in _EXTERNAL_TOOLS or tool.startswith(_EXTERNAL_PREFIXES)
        for tool in tools
    ):
        matches.add("external")
    if any(
        tool in _FINANCIAL_TOOLS or tool.startswith(_FINANCIAL_PREFIXES)
        for tool in tools
    ):
        matches.add("financial")
    return matches


def load_examples(
    per_slice_per_label: int = _PER_SLICE_LIMIT,
) -> list[tuple[str, list[float], list[float]]]:
    dataset = load_dataset(_DATASET, split="train", revision=_REVISION)

    labels_by_text = defaultdict(set)
    for row in dataset:
        labels_by_text[_normalise(row["instruction"])].add(row["gt_label"])
    ambiguous = {text for text, labels in labels_by_text.items() if len(labels) > 1}

    buckets = defaultdict(list)
    for row in dataset:
        text = _normalise(row["instruction"])
        if not text or text in ambiguous or row["gt_label"] not in {"safe", "unsafe"}:
            continue
        value = 1.0 if row["gt_label"] == "unsafe" else 0.0
        matches = _matching_slices(row)
        for slice_name in _SLICE_ORDER:
            bucket = buckets[(slice_name, value)]
            if slice_name in matches and len(bucket) < per_slice_per_label:
                bucket.append(text)

    examples = []
    seen = set()
    for key in sorted(buckets):
        value = key[1]
        for text in buckets[key]:
            if (text, value) in seen:
                continue
            seen.add((text, value))
            labels = [0.0] * len(LABELS)
            mask = [0.0] * len(LABELS)
            labels[_DTA_INDEX] = value
            mask[_DTA_INDEX] = 1.0
            examples.append((text, labels, mask))
    return examples
