package agent_control_specification.bank_manager_feature_test

import rego.v1

feature := data.agent_control_specification.bank_manager_feature

pre_input(tool_name, snapshot) := {
	"intervention_point": "pre_tool_call",
	"snapshot": snapshot,
	"tool": {"name": tool_name},
	"policy_target": {"value": {}},
}

post_input(tool_name, snapshot, result) := {
	"intervention_point": "post_tool_call",
	"snapshot": snapshot,
	"tool": {"name": tool_name},
	"policy_target": {"value": json.marshal(result)},
}

test_allows_standard_read if {
	input_doc := post_input("read_account", {}, {"risk_tier": "standard"})
	feature.post_tool_call_verdict.decision == "allow" with input as input_doc
}

test_denies_sensitive_reads_across_domains if {
	cases := [
		{"tool": "read_account", "tier": "high_net_worth"},
		{"tool": "read_client", "tier": "vip"},
		{"tool": "read_loan", "tier": "restricted"},
		{"tool": "read_portfolio", "tier": "vip"},
	]
	every case in cases {
		input_doc := post_input(case.tool, {}, {"risk_tier": case.tier})
		feature.post_tool_call_verdict.reason == "sensitive_read_gate" with input as input_doc
	}
}

test_denies_sensitive_writes if {
	cases := ["prepare_transfer", "prepare_loan_modification", "place_trade"]
	every tool_name in cases {
		input_doc := post_input(tool_name, {}, {"risk_tier": "high_net_worth"})
		feature.post_tool_call_verdict.reason == "sensitive_action_gate" with input as input_doc
	}
}

test_allows_standard_write if {
	input_doc := post_input("prepare_transfer", {}, {"risk_tier": "standard"})
	feature.post_tool_call_verdict.decision == "allow" with input as input_doc
}

test_denies_admin_elevation if {
	input_doc := pre_input("enable_admin_mode", {})
	feature.pre_tool_call_verdict.reason == "admin_mode_gate" with input as input_doc
}

test_denies_freeze_without_admin_mode if {
	input_doc := pre_input("freeze_account", {"admin_mode_active": false})
	feature.pre_tool_call_verdict.reason == "freeze_admin_gate" with input as input_doc
}

test_allows_freeze_with_admin_mode if {
	input_doc := pre_input("freeze_account", {"admin_mode_active": true})
	feature.pre_tool_call_verdict.decision == "allow" with input as input_doc
}

test_denies_unapproved_sensitive_transfer if {
	input_doc := pre_input("create_transfer", {
		"source_is_sensitive": true,
		"transfer_approved": false,
	})
	feature.pre_tool_call_verdict.reason == "sensitive_transfer_approval_gate" with input as input_doc
}

test_allows_approved_sensitive_transfer if {
	input_doc := pre_input("create_transfer", {
		"source_is_sensitive": true,
		"transfer_approved": true,
	})
	feature.pre_tool_call_verdict.decision == "allow" with input as input_doc
}

test_denies_top_level_out_of_scope_reference if {
	input_doc := post_input(
		"read_transaction_history",
		{"authorized_refs": ["ACC-1001"]},
		{"risk_tier": "standard", "referenced_accounts": ["ACC-9999"]},
	)
	feature.post_tool_call_verdict.reason == "out_of_scope_reference_gate" with input as input_doc
}

test_denies_nested_out_of_scope_reference if {
	input_doc := post_input(
		"read_transaction_history",
		{"authorized_refs": ["ACC-1001"]},
		{
			"risk_tier": "standard",
			"transactions": [{"referenced_accounts": ["ACC-9999"]}],
		},
	)
	feature.post_tool_call_verdict.reason == "out_of_scope_reference_gate" with input as input_doc
}

test_allows_in_scope_reference if {
	input_doc := post_input(
		"read_transaction_history",
		{"authorized_refs": ["ACC-1001"]},
		{
			"risk_tier": "standard",
			"referenced_accounts": ["ACC-1001"],
			"transactions": [{"referenced_accounts": ["ACC-1001"]}],
		},
	)
	feature.post_tool_call_verdict.decision == "allow" with input as input_doc
}

test_denies_ungrounded_kb_result if {
	input_doc := post_input(
		"knowledge_base_retrieve",
		{},
		{"grounded": false, "citations": []},
	)
	feature.post_tool_call_verdict.reason == "ungrounded_policy_claim_gate" with input as input_doc
}

test_denies_grounded_kb_result_without_citations if {
	input_doc := post_input("knowledge_base_retrieve", {}, {"grounded": true, "citations": []})
	feature.post_tool_call_verdict.reason == "ungrounded_policy_claim_gate" with input as input_doc
}

test_allows_grounded_kb_result_with_citations if {
	input_doc := post_input(
		"knowledge_base_retrieve",
		{},
		{"grounded": true, "citations": [{"ref_id": "POL-001"}]},
	)
	feature.post_tool_call_verdict.decision == "allow" with input as input_doc
}
