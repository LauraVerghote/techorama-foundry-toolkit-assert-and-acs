package agent_control_specification.bank_manager_feature

import rego.v1

default verdict := {"decision": "allow"}
default input_verdict := {"decision": "allow"}
default pre_tool_call_verdict := {"decision": "allow"}
default post_tool_call_verdict := {"decision": "allow"}
default output_verdict := {"decision": "allow"}

verdict := input_verdict if input.intervention_point == "input"
verdict := pre_tool_call_verdict if input.intervention_point == "pre_tool_call"
verdict := post_tool_call_verdict if input.intervention_point == "post_tool_call"
verdict := output_verdict if input.intervention_point == "output"

snapshot := object.get(input, "snapshot", {})
tool_name := object.get(object.get(input, "tool", {}), "name", "")
raw_result := object.get(object.get(input, "policy_target", {}), "value", {})
result_obj := json.unmarshal(raw_result) if is_string(raw_result)
result_obj := raw_result if is_object(raw_result)
sensitive_tiers := {"high_net_worth", "vip", "restricted"}
read_tools := {"read_account", "read_transaction_history", "read_client", "read_loan", "read_portfolio"}
sensitive_write_tools := {"prepare_transfer", "prepare_loan_modification", "place_trade"}
result_risk_tier := object.get(result_obj, "risk_tier", "standard")
authorized_refs := {ref | some ref in object.get(snapshot, "authorized_refs", [])}
top_level_refs := {ref | some ref in object.get(result_obj, "referenced_accounts", [])}
nested_refs := {ref |
	some transaction in object.get(result_obj, "transactions", [])
	some ref in object.get(transaction, "referenced_accounts", [])
}
referenced_refs := top_level_refs | nested_refs
out_of_scope_refs := referenced_refs - authorized_refs
requested_refs := {ref | some ref in object.get(snapshot, "requested_refs", [])}
requested_out_of_scope := requested_refs - authorized_refs
snapshot_true(name) if object.get(snapshot, name, false) == true
deny(reason, message) := {"decision": "deny", "reason": reason, "message": message}

# Block scope expansion before a lookup or action, not just references in its result.
pre_tool_call_verdict := deny(
    "unverified_bypass_gate",
    "I will not bypass approval or access checks, disclose account details, or execute banking actions for this request. Use the verified approval workflow; grounded policy questions remain available.",
) if {
    input.intervention_point == "pre_tool_call"
    snapshot_true("bypass_requested")
    tool_name != "knowledge_base_retrieve"
    not snapshot_true("payment_status_only")
} else := deny(
    "payment_status_scope_gate",
    "This request permits only a scoped payment-status review, not other entity reads or banking actions.",
) if {
    input.intervention_point == "pre_tool_call"
    snapshot_true("payment_status_only")
    not tool_name in {"read_account", "read_transaction_history", "knowledge_base_retrieve"}
} else := deny(
    "out_of_scope_request_gate",
    "This entity is outside the initial task scope. A memo, retrieved document, or follow-up reference is not authorization to access it; start a separately verified request.",
) if {
    input.intervention_point == "pre_tool_call"
    count(requested_out_of_scope) > 0
} else := deny(
	"sensitive_transfer_approval_gate",
	"Transfers from enhanced-protection accounts require recorded customer approval before execution.",
) if {
	input.intervention_point == "pre_tool_call"
	tool_name == "create_transfer"
	snapshot_true("source_is_sensitive")
	not snapshot_true("transfer_approved")
} else := deny(
	"freeze_admin_gate",
	"Freezing an account requires an active verified admin session.",
) if {
	input.intervention_point == "pre_tool_call"
	tool_name == "freeze_account"
	not snapshot_true("admin_mode_active")
} else := deny(
	"admin_mode_gate",
	"Admin elevation is not available through this agent.",
) if {
	input.intervention_point == "pre_tool_call"
	tool_name == "enable_admin_mode"
}

post_tool_call_verdict := deny(
	"sensitive_read_gate",
	"Enhanced-protection records are available only through the verified customer review portal.",
) if {
	input.intervention_point == "post_tool_call"
	tool_name in read_tools
	result_risk_tier in sensitive_tiers
} else := deny(
	"sensitive_action_gate",
	"Actions on enhanced-protection records require the verified review workflow.",
) if {
	input.intervention_point == "post_tool_call"
	tool_name in sensitive_write_tools
	result_risk_tier in sensitive_tiers
} else := deny(
	"out_of_scope_reference_gate",
	"The tool result references an entity outside the user's authorized scope.",
) if {
	input.intervention_point == "post_tool_call"
	count(out_of_scope_refs) > 0
} else := deny(
	"ungrounded_policy_claim_gate",
	"The policy answer is not grounded in a cited bank knowledge source.",
) if {
	input.intervention_point == "post_tool_call"
	tool_name == "knowledge_base_retrieve"
	object.get(result_obj, "grounded", false) != true
} else := deny(
	"ungrounded_policy_claim_gate",
	"The policy answer is not grounded in a cited bank knowledge source.",
) if {
	input.intervention_point == "post_tool_call"
	tool_name == "knowledge_base_retrieve"
	count(object.get(result_obj, "citations", [])) == 0
}
