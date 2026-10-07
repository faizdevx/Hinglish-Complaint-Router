"""Intent -> operational destination.

This is a PROJECT-LEVEL taxonomy written by the author of this repo, not a label that
exists in any dataset and not any real company's org chart. The destination names are
re-used from the department options that appear in the synthetic_enterprise routing
questions (billing, technical_support, account_management, loans_and_credit,
fraud_and_disputes, general_inquiry). The assignment of each Banking77 intent to a
destination is a judgement call and has NOT been validated against ground truth
(no dataset provides intent+route pairs). Routing is therefore a deterministic function
of the predicted intent, not an independently learned task.
"""
from __future__ import annotations

_GROUPS = {
    "fraud_and_disputes": """compromised_card lost_or_stolen_card lost_or_stolen_phone card_payment_not_recognised
        cash_withdrawal_not_recognised direct_debit_payment_not_recognised transaction_charged_twice
        wrong_amount_of_cash_received reverted_card_payment?""",
    "billing": """Refund_not_showing_up request_refund card_payment_fee_charged cash_withdrawal_charge exchange_charge
        extra_charge_on_statement top_up_by_bank_transfer_charge top_up_by_card_charge transfer_fee_charged
        wrong_exchange_rate_for_cash_withdrawal card_payment_wrong_exchange_rate""",
    "technical_support": """activate_my_card apple_pay_or_google_pay atm_support automatic_top_up
        balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit
        beneficiary_not_allowed card_linking card_not_working card_swallowed contactless_not_working
        declined_card_payment declined_cash_withdrawal declined_transfer failed_transfer pending_card_payment
        pending_cash_withdrawal pending_top_up pending_transfer top_up_failed top_up_reverted
        transfer_not_received_by_recipient virtual_card_not_working cancel_transfer""",
    "account_management": """change_pin pin_blocked passcode_forgotten edit_personal_details terminate_account
        verify_my_identity unable_to_verify_identity why_verify_identity verify_source_of_funds verify_top_up
        disposable_card_limits top_up_limits get_disposable_virtual_card get_physical_card getting_spare_card
        getting_virtual_card order_physical_card card_about_to_expire card_arrival""",
    "general_inquiry": """age_limit card_acceptance card_delivery_estimate country_support exchange_rate
        exchange_via_app fiat_currency_support receiving_money supported_cards_and_currencies top_up_by_cash_or_cheque
        topping_up_by_card transfer_into_account transfer_timing visa_or_mastercard""",
}
INTENT_TO_ROUTE: dict[str, str] = {i: route for route, ids in _GROUPS.items() for i in ids.split()}
ROUTES = sorted(_GROUPS)
UNMAPPED = "unmapped"


def route_for(intent: str) -> str:
    """Destination for a predicted intent label; ``unmapped`` if the label is not in the taxonomy."""
    return INTENT_TO_ROUTE.get(intent, UNMAPPED)
