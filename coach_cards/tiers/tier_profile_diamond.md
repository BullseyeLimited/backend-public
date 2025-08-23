---
type: tier_profile
name: tier_profile_emerald
version: 0.5
inputs: as_spec_v0_2
tags: [tier]

gating:
  # Recognition: “girlfriend/VIP+”.
  ltv_min_cents: 150000         # $1,500+
  ltv_max_cents: null
  min_paid_ppvs: 5

goal:
  summary: "Sustain VIP experience and long-term retention."

escalate_when:
  ltv_cents_gte: null           # top tier; no further escalation
  paid_ppvs_gte: null