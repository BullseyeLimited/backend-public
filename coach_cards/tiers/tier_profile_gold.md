---
type: tier_profile
name: tier_profile_gold
version: 0.5
inputs: as_spec_v0_2
tags: [tier]

gating:
  # Recognition: has paid, not yet whale.
  ltv_min_cents: 1
  ltv_max_cents: 49999          # <$500 lifetime
  min_paid_ppvs: 1

goal:
  summary: "Stabilize repeat buys and grow average ticket."

escalate_when:
  ltv_cents_gte: 50000          # promote to Diamond at $500+
  paid_ppvs_gte: 3