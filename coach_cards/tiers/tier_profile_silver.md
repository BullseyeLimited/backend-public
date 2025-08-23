---
type: tier_profile
name: tier_profile_silver
version: 0.5
inputs: as_spec_v0_2
tags: [tier]

gating:
  # Recognition: brand-new or free-only, not flagged as Bronze.
  ltv_min_cents: 0
  ltv_max_cents: 0
  min_paid_ppvs: 0
  prefer_if_unpaid_strikes_gte: 0

goal:
  summary: "Warm up and win the FIRST paid action."

escalate_when:
  ltv_cents_gte: 1
  paid_ppvs_gte: 1