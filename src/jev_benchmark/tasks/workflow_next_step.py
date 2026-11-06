"""S2 - Next-step decision in a workflow (ABCD, Action-Based Conversations Dataset).

A customer-service agent of an online clothing retailer follows a standard operating procedure
(SOP) for the current workflow (e.g. "Order Issue > Status Mystery Fee"). At the moments where the
human agent clicked a system button, the system under test receives the SOP of that workflow and the
conversation so far (including the system actions already taken) and must choose the next action
among the 30 buttons of the ABCD ontology. The gold label is the button the agent actually clicked.

This is a realistic "workflow engine" setting: the workflow is known (it would be selected upstream,
e.g. by the S1 router), the decision is *where we are* in the procedure and which branch applies.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict

from jev_benchmark.config import RAW_DIR
from jev_benchmark.tasks.base import Example, Task, TaskKind, download

ABCD_URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
GUIDELINES_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json"

# Descriptions start with the button name used in the SOP text, so both systems can link them.
ACTIONS: dict[str, str] = {
    "pull-up-account": "[Pull up Account] Load the customer's account from their full name or account ID; "
    "usually the first step of account- or order-related workflows.",
    "verify-identity": "[Verify Identity] Verify the customer's identity with the required fields "
    "(e.g. full name, account ID, order ID, zip code) before sharing or changing protected data.",
    "validate-purchase": "[Validate Purchase] Confirm that a specific purchase is valid using username, "
    "email address and order ID.",
    "ask-the-oracle": "[Ask the Oracle] Query the backend, which answers Yes/No, to check whether the "
    "customer's claim is valid (e.g. a company error, a missing credit, a delivery problem).",
    "membership": "[Membership Privileges] Enter the customer's membership level (guest, bronze, "
    "silver, gold) to find out which remedies or policies apply.",
    "enter-details": "[Enter Details] Enter additional information required by the procedure "
    "(e.g. full address, account ID, username, email, PIN, security answer, amount).",
    "record-reason": "[Record Reason] Record the reason or key detail of the request (e.g. refund "
    "method, item to return, how the customer learned about the issue, competitor price).",
    "offer-refund": "[Offer Refund] Refund a dollar amount to the customer.",
    "update-order": "[Update Order] Change an existing order (e.g. shipping method or address, items, "
    "payment method, fee removal, how a return is processed).",
    "update-account": "[Update Account] Change the customer's account profile or services (e.g. "
    "address, name, phone, payment method, add or remove a service).",
    "make-purchase": "[Make Purchase] Place an order for an item on the customer's behalf.",
    "shipping-status": "[Shipping Status] Check the shipping status of the order (e.g. whether the "
    "item has already shipped).",
    "subscription-status": "[Subscription Status] Check the status of the customer's premium "
    "subscription (active, amount due, due date).",
    "promo-code": "[Promo Code] Generate a promo code to give to the customer.",
    "make-password": "[Make Password] Generate a new password for the customer.",
    "send-link": "[Send Link] Send the customer a link (e.g. to their account page so they can "
    "check or manage things themselves).",
    "notify-team": "[Notify Internal Team] Report the issue to an internal team (e.g. website team, "
    "purchasing department).",
    "log-out-in": "[Log Out/In] Instruct the customer to log out of their account and log back in.",
    "try-again": "[Try Again] Ask the customer to retry the failed operation (e.g. re-enter the "
    "card, search again, reload).",
    "instructions": "[Instructions] Give the customer troubleshooting instructions (e.g. clear "
    "cookies or cache, close other tabs).",
    "search-faq": "[Search FAQ] Open the FAQ view because the customer's question can be answered "
    "from the FAQ pages.",
    "search-boots": "[Boots] In the FAQ view, open the topic about boots.",
    "search-shirt": "[Shirt] In the FAQ view, open the topic about shirts.",
    "search-jeans": "[Jeans] In the FAQ view, open the topic about jeans.",
    "search-jacket": "[Jacket] In the FAQ view, open the topic about jackets.",
    "search-pricing": "[Pricing] In the FAQ view, open the store-wide topic about pricing.",
    "search-timing": "[Timing] In the FAQ view, open the store-wide topic about timing "
    "(dates, deadlines, opening times).",
    "search-membership": "[Membership] In the FAQ view, open the store-wide topic about the "
    "membership program.",
    "search-policy": "[Policy] In the FAQ view, open the store-wide topic about store policies.",
    "select-faq": "[Select Answer] Select the FAQ answer matching the customer's question, after "
    "the FAQ topic has been opened.",
}

TASK = Task(
    name="workflow_next_step",
    title="S2 - Decisione del passo successivo in un workflow",
    kind=TaskKind.CHOICE,
    instructions=(
        "You are the workflow engine assisting a customer-service agent of an online clothing "
        "retailer. The agent must follow the procedure of the current workflow. Given the procedure "
        "and the conversation so far (including the system actions already taken), which system "
        "action should the agent take next?"
    ),
    labels=ACTIONS,
    dataset="ABCD v1.1 (ASAPP, Chen et al. NAACL 2021), split test (+dev for rare actions), MIT",
)

FLOW_NAMES = {
    "account_access": "Account Access",
    "manage_account": "Manage Account",
    "order_issue": "Order Issue",
    "product_defect": "Product Defect",
    "purchase_dispute": "Purchase Dispute",
    "shipping_issue": "Shipping Issue",
    "single_item_query": "Single-Item Query",
    "storewide_query": "Storewide Query",
    "subscription_inquiry": "Subscription Inquiry",
    "troubleshoot_site": "Troubleshoot Site",
}

# Data subflow -> guideline subflow name (the FAQ flows are handled by `guideline_key`).
SUBFLOW_NAMES = {
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
    "reset_2fa": "Reset Two-Factor Auth",
    "status_service_added": "Status Service Added",
    "status_service_removed": "Status Service Removed",
    "status_shipping_question": "Status Shipping Question",
    "status_credit_missing": "Status Credit Missing",
    "manage_change_address": "Manage Change Address",
    "manage_change_name": "Manage Change Name",
    "manage_change_phone": "Manage Change Phone",
    "manage_payment_method": "Manage Payment Method",
    "status_mystery_fee": "Status Mystery Fee",
    "status_delivery_time": "Status Delivery Time",
    "status_delivery_date": "Status Delivery Time",
    "status_payment_method": "Status Payment Method",
    "status_quantity": "Status Quantity",
    "manage_upgrade": "Manage Upgrade",
    "manage_downgrade": "Manage Downgrade",
    "manage_create": "Manage Create",
    "manage_cancel": "Manage Cancel",
    "refund_initiate": "Initiate Refund",
    "refund_update": "Update Refund",
    "refund_status": "Refund Status",
    "return_stain": "Return Due to Stain",
    "return_color": "Return Due to Color",
    "return_size": "Return Due to Size",
    "bad_price_competitor": "Bad Price Competitor",
    "bad_price_yesterday": "Bad Price Yesterday",
    "out_of_stock_general": "Out-of-Stock General",
    "out_of_stock_one_item": "Out-of-Stock One Item",
    "promo_code_invalid": "Promo Code Invalid",
    "promo_code_out_of_date": "Promo Code Out of Date",
    "mistimed_billing_already_returned": "Mistimed Billing Already Returned",
    "mistimed_billing_never_bought": "Mistimed Billing Never Bought",
    "status": "Shipping Status",
    "manage": "Manage Shipping",
    "missing": "Missing Item",
    "cost": "Shipping Cost",
    "status_active": "Status Active",
    "status_questions": "Status Active",
    "status_due_amount": "Status Due Amount",
    "status_due_date": "Status Due Date",
    "manage_pay_bill": "Manage Pay Bill",
    "manage_extension": "Manage Extension",
    "manage_dispute_bill": "Manage Dispute Bill",
    "credit_card": "Invalid Credit Card",
    "shopping_cart": "Cart Not Updating",
    "search_results": "Search Not Working",
    "slow_speed": "Website Too Slow",
}

SPEAKERS = {"agent": "Agent", "customer": "Customer", "action": "System"}


def guideline_key(flow: str, subflow: str) -> tuple[str, str]:
    """Map a data (flow, subflow) pair to the (flow, subflow) names used in guidelines.json."""
    if flow in ("single_item_query", "storewide_query"):  # e.g. "boots_how_1", "timing_4"
        return FLOW_NAMES[flow], f"{subflow.split('_')[0].capitalize()} FAQ"
    return FLOW_NAMES[flow], SUBFLOW_NAMES[subflow]


def _clean(text: str) -> str:
    return " ".join(text.replace("�", "'").split())


def render_procedure(guidelines: dict, flow: str, subflow: str) -> str:
    """Render the SOP of one workflow as plain text."""
    flow_name, sub_name = guideline_key(flow, subflow)
    spec = guidelines[flow_name]["subflows"][sub_name]
    lines = [f"Workflow: {flow_name} > {sub_name} ({_clean(guidelines[flow_name]['description'])})"]
    lines += [f"Note: {_clean(t)}" for t in spec.get("instructions", []) if _clean(t)]
    lines.append("Steps:")
    for n, step in enumerate(spec["actions"], start=1):
        button = step["button"]
        head = "[no system action]" if button == "N/A" else f"[{button}]"
        text = _clean(step["text"])
        lines.append(f"{n}. {head} {text if text.lower() != 'x' else ''}".rstrip())
        lines += [f"   - {_clean(s)}" for s in step["subtext"] if _clean(s) and _clean(s).lower() != "x"]
    return "\n".join(lines)


def render_conversation(turns: list[list[str]]) -> str:
    return "\n".join(f"{SPEAKERS[speaker]}: {_clean(text)}" for speaker, text in turns)


def _decision_points(convo: dict) -> list[tuple[int, str]]:
    """(turn index, gold action) for every action turn of a conversation."""
    original, delexed = convo["original"], convo["delexed"]
    if len(original) != len(delexed):
        return []
    points = []
    for i, (orig, delex) in enumerate(zip(original, delexed, strict=True)):
        if orig[0] == "action" and delex["speaker"] == "action":
            action = delex["targets"][2]
            if action in ACTIONS:
                points.append((i, action))
    return points


def prepare(per_class: int = 10, seed: int = 42) -> list[Example]:
    data_path = download(ABCD_URL, RAW_DIR / "abcd" / "abcd_v1.1.json.gz")
    guide_path = download(GUIDELINES_URL, RAW_DIR / "abcd" / "guidelines.json")
    with gzip.open(data_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    guidelines = json.loads(guide_path.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    # Candidates per action, test split first; dev is used only to fill rare actions.
    by_action: dict[str, dict[str, list]] = defaultdict(lambda: {"test": [], "dev": []})
    for split in ("test", "dev"):
        for convo in data[split]:
            for idx, action in _decision_points(convo):
                by_action[action][split].append((split, convo, idx))

    picked, used_convos = [], set()
    for action in ACTIONS:  # deterministic order
        chosen = []
        for split in ("test", "dev"):
            pool = by_action[action][split]
            rng.shuffle(pool)
            for split_name, convo, idx in pool:
                if len(chosen) == per_class:
                    break
                if convo["convo_id"] in used_convos:  # one decision point per conversation
                    continue
                chosen.append((split_name, convo, idx, action))
                used_convos.add(convo["convo_id"])
        picked.extend(chosen)
    rng.shuffle(picked)

    examples = []
    for split, convo, idx, action in picked:
        scenario = convo["scenario"]
        examples.append(
            Example(
                id=f"abcd-{split}-{convo['convo_id']}-t{idx:02d}",
                label=action,
                state={
                    "procedure": render_procedure(guidelines, scenario["flow"], scenario["subflow"]),
                    "conversation_so_far": render_conversation(convo["original"][:idx]),
                },
                meta={
                    "split": split,
                    "convo_id": convo["convo_id"],
                    "turn": idx,
                    "flow": scenario["flow"],
                    "subflow": scenario["subflow"],
                },
            )
        )
    return examples
