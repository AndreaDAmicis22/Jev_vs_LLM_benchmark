"""S1 - Incoming request classification (Banking77).

A customer writes to the support chat of a digital bank; the system must route the message to one of
77 fine-grained intents. Descriptions were written from the *train* split (never from the test
split used for the benchmark) and follow the actual semantics of the dataset labels, e.g.
`get_physical_card` is mostly about the PIN of the newly received physical card.
"""

from __future__ import annotations

import csv
from collections import Counter

from jev_benchmark.config import RAW_DIR
from jev_benchmark.tasks.base import (
    Example,
    Task,
    TaskKind,
    balanced_quotas,
    download,
    stratified_sample,
)

BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)

# Order follows `categories.json` of the original dataset.
INTENTS: dict[str, str] = {
    "card_arrival": "The ordered card has not arrived yet; the customer wants to track it.",
    "card_linking": "Link, connect or re-activate an existing card in the app.",
    "exchange_rate": "How exchange rates are determined or which rate will be applied.",
    "card_payment_wrong_exchange_rate": "A card payment in a foreign currency used a wrong exchange rate.",
    "extra_charge_on_statement": "An unexpected extra fee or charge appears on the statement.",
    "pending_cash_withdrawal": "An ATM cash withdrawal is still shown as pending.",
    "fiat_currency_support": "Which fiat currencies can be held or used with the account.",
    "card_delivery_estimate": "How long card delivery takes, express or scheduled delivery.",
    "automatic_top_up": "The automatic top-up feature: how it works, enabling or disabling it.",
    "card_not_working": "The physical card does not work (generic failure).",
    "exchange_via_app": "Exchanging between currencies inside the app.",
    "lost_or_stolen_card": "The card was lost or stolen.",
    "age_limit": "Minimum age or requirements to open an account (e.g. for a child).",
    "pin_blocked": "The PIN is blocked after too many wrong attempts.",
    "contactless_not_working": "Contactless payments are not working.",
    "top_up_by_bank_transfer_charge": "Fees for topping up or transferring money by bank transfer.",
    "pending_top_up": "A top-up has been made but is still pending or not in the balance.",
    "cancel_transfer": "Cancel or reverse a transfer that was just sent (e.g. to the wrong account).",
    "top_up_limits": "Minimum or maximum amount allowed for a top-up.",
    "wrong_amount_of_cash_received": "The ATM dispensed less or different cash than requested.",
    "card_payment_fee_charged": "A fee was charged on a card payment.",
    "transfer_not_received_by_recipient": "A transfer was sent but the recipient says it has not arrived.",
    "supported_cards_and_currencies": "Which cards and currencies are supported (e.g. to top up).",
    "getting_virtual_card": "How to get or where to find a virtual card.",
    "card_acceptance": "Where the card is accepted or refused (shops, merchants, places).",
    "top_up_reverted": "A top-up was reverted, cancelled or returned.",
    "balance_not_updated_after_cheque_or_cash_deposit": "A cheque or cash deposit is not reflected in the balance.",
    "card_payment_not_recognised": "A card payment on the statement is not recognised by the customer.",
    "edit_personal_details": "Change personal details such as name or address.",
    "why_verify_identity": "Why identity verification is required, or whether it can be skipped.",
    "unable_to_verify_identity": "The customer is trying but fails to verify their identity.",
    "get_physical_card": "Information about the newly received physical card, mainly where to find its PIN.",
    "visa_or_mastercard": "Whether the card is Visa or Mastercard, or a request for a specific network.",
    "topping_up_by_card": "Topping up with a card, including someone else topping up or money missing after it.",
    "disposable_card_limits": "Limits and restrictions of disposable virtual cards.",
    "compromised_card": "Card details may have been compromised or used fraudulently.",
    "atm_support": "Which ATMs can be used to withdraw money with the card.",
    "direct_debit_payment_not_recognised": "An unknown direct debit charge appears on the statement.",
    "passcode_forgotten": "The app passcode or password was forgotten and must be reset.",
    "declined_cash_withdrawal": "An ATM cash withdrawal was declined.",
    "pending_card_payment": "A card payment is still pending.",
    "lost_or_stolen_phone": "The phone with the banking app was lost or stolen.",
    "request_refund": "The customer wants to cancel a purchase or get a refund from a merchant.",
    "declined_transfer": "A transfer was declined.",
    "Refund_not_showing_up": "A refund from a merchant has not shown up in the account yet.",
    "declined_card_payment": "A card payment was declined.",
    "pending_transfer": "An outgoing transfer is still pending.",
    "terminate_account": "Close or delete the account.",
    "card_swallowed": "An ATM retained (swallowed) the card.",
    "transaction_charged_twice": "The same transaction was charged twice.",
    "verify_source_of_funds": "Verification of the source of funds / where the money comes from.",
    "transfer_timing": "How long a transfer takes to arrive.",
    "reverted_card_payment?": "A card payment was reverted or cancelled and the money came back.",
    "change_pin": "Change the card PIN.",
    "beneficiary_not_allowed": "A transfer to a specific beneficiary or account is not allowed.",
    "transfer_fee_charged": "A fee was charged on a transfer.",
    "receiving_money": "Receiving money or salary into the account, including other currencies.",
    "failed_transfer": "A transfer attempt failed.",
    "transfer_into_account": "Move money into this account from another bank account.",
    "verify_top_up": "Verification step or code requested for a top-up.",
    "getting_spare_card": "Order additional or spare physical cards and their cost.",
    "top_up_by_cash_or_cheque": "Top up or deposit money with cash or cheque.",
    "order_physical_card": "Order a physical card and how or where it will be received.",
    "virtual_card_not_working": "The virtual card does not work.",
    "wrong_exchange_rate_for_cash_withdrawal": "Exchange rate applied to a cash withdrawal in a foreign currency.",
    "get_disposable_virtual_card": "What disposable virtual cards are and how to get one.",
    "top_up_failed": "A top-up attempt failed or was rejected.",
    "balance_not_updated_after_bank_transfer": "An incoming bank transfer is not yet reflected in the balance.",
    "cash_withdrawal_not_recognised": "A cash withdrawal on the statement was not made by the customer.",
    "exchange_charge": "Fees or discounts for currency exchange.",
    "top_up_by_card_charge": "Fees for topping up the account by card.",
    "activate_my_card": "Activate a newly received card.",
    "cash_withdrawal_charge": "A fee was charged for a cash withdrawal.",
    "card_about_to_expire": "The card is about to expire: replacement, cost and timing.",
    "apple_pay_or_google_pay": "Using Apple Pay or Google Pay, including topping up through them.",
    "verify_my_identity": "How to verify identity and which documents are needed.",
    "country_support": "Which countries are supported for accounts and cards.",
}

TASK = Task(
    name="intent_routing",
    title="S1 - Classificazione delle richieste in ingresso",
    kind=TaskKind.CHOICE,
    instructions=(
        "A customer sent this message to the support chat of a digital bank. "
        "Which intent best describes what the customer needs?"
    ),
    labels=INTENTS,
    dataset="Banking77 (PolyAI, Casanueva et al. 2020), split test, CC BY 4.0",
)


def prepare(total: int = 500, seed: int = 42) -> list[Example]:
    """`total` test messages, 6-7 per intent (v1 used 4 per intent: those 308 are all included)."""
    path = download(BANKING77_TEST_URL, RAW_DIR / "banking77" / "test.csv")
    with path.open(encoding="utf-8") as fh:
        rows = [(i, row) for i, row in enumerate(csv.DictReader(fh))]
    quotas = balanced_quotas(Counter(row["category"] for _, row in rows), total, seed)
    sampled = stratified_sample(rows, key=lambda r: r[1]["category"], per_class=quotas, seed=seed)
    return [
        Example(
            id=f"banking77-test-{i:04d}",
            label=row["category"],
            state={"customer_message": row["text"].strip()},
            meta={"split": "test", "row": i},
        )
        for i, row in sampled
    ]
