---
type: tier_profile
name: tier_profile_bronze
version: 0.5
inputs: as_spec_v0_2
tags: [tier]

gating:
  # Recognition: zero spenders; prefer Bronze if they've failed to pay repeatedly.
  ltv_min_cents: 0
  ltv_max_cents: 0
  min_paid_ppvs: 0
  prefer_if_unpaid_strikes_gte: 3   # optional tie-breaker if you track strikes

goal:
  summary: "Convert first payment (micro) and filter timewasters."

escalate_when:
  ltv_cents_gte: 1
  paid_ppvs_gte: 1