# Bank policy knowledge base (the KB corpus)

These seven Markdown files **are the company policy** that the demo agent
retrieves and grounds against through the local BM25 knowledge-base MCP server.

If you were looking for *"where is the policy that says a VIP / high-net-worth
client needs verified approval"* — it is right here:

| Policy fact shown in the demo | File | Section |
|---|---|---|
| Accounts/clients with `risk_tier` **high_net_worth** or **vip** are on the **enhanced-protection list**; state-changing actions require verified approval | [`sensitive-account-access.md`](sensitive-account-access.md) | *Enhanced-protection accounts* |
| **All** VIP / high-net-worth transfers require recorded customer approval through the verified review portal, **regardless of amount** | [`transfers-and-approvals.md`](transfers-and-approvals.md) | *Transfer approval thresholds* |
| Jumbo refinance LTV cap + exception approver | [`home-loan-policy.md`](home-loan-policy.md) | — |
| Brokerage margin-call review | [`brokerage-policy.md`](brokerage-policy.md) | — |
| KYC / AML obligations | [`kyc-aml.md`](kyc-aml.md) | — |
| Fees and dispute handling | [`fees-and-disputes.md`](fees-and-disputes.md) | — |
| Vendor onboarding notes (off-topic control) | [`vendor-onboarding-notes.md`](vendor-onboarding-notes.md) | — |

## How it's consumed

The backend returns `answer`, `citations`, and `grounded`, so the agent and ACS
controls can use typed grounding signals (see [`../kb_backend.py`](../kb_backend.py)):

- **`mock`** (default, no Azure) — pure-Python BM25 over these `*.md` files.
  Set `KB_BACKEND=mock`. Optional `KB_CORPUS_DIR` overrides this directory.

## Grounding is the control signal

The `grounded` flag and the presence of `citations` are **typed signals**, not
answer-text pattern matching. The ACS grounding gate keys on them: a claim with
no supporting citation is `ungrounded_policy_claim` and can be blocked. Try the
`What is the capital of France?` preset — no content overlap with this corpus →
`grounded: false` → the gate can deny an ungrounded "policy" answer.
