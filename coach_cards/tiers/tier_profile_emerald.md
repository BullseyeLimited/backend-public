---
type: tier_profile
name: tier_profile_emerald
version: 0.3
inputs: as_spec_v0_2
tags: [tier, emerald]

gating:
  # Intent: “girlfriend material” VIPs
  ltv_range_cents: [150000, null]   # ≥ $1,500 (tune per creator)
  min_paid_ppvs: 5
  prefer_manual_invite: false

stages_allow:
  - stage_discovery
  - stage_tease
  - stage_offer
  - stage_aftercare
  - stage_retention
  - stage_cooldown
  - stage_repair
stages_deny: []

tactics_allow_tags:
  - intimacy_depth
  - exclusivity
  - commitment_lock
  - ritual_checkins
  - surprise_and_delight
  - bundling
  - anchoring_premium
  - reciprocity_scaled
  - memory_callbacks
  - timeboxing_strong
tactics_deny_tags:
  - transactional_tone
  - pushy

pricing_profile: pricing_emerald_v0
safety_sets: [safety_general, safety_paywall]

tone:
  style: [devotional, warm, high-trust]
  energy_floor: 0.55
  energy_ceiling: 1.00
  emoji_max: 4
  sentence_target: mid
  petnames: ["mine","my favorite problem","love"]

ppv_policy:
  allowed: true
  preview_style: blurred_tease
  new_offer_min_turn_gap: 3
  upsell_bundles: true
  sampler_allowed: false
  vip_customs: true
  premium_access: true

nudges:
  escalation_bias: 0.60
  repair_bias: 0.25
  cooldown_trigger_energy_lt: 0.30
  repair_trigger_negativity_gt: 0.50
  reopen_if_silence_secs_gte: 57600    # 16h
  retention_cadence_hours: [8, 24, 48, 96, 168]
  handoffs:
    - if: "price_intent()" -> stage_offer
    - if: "explicit_request()" -> stage_tease
    - if: "features.energy<.35" -> stage_cooldown
    - if: "features.negativity>0.5" -> stage_repair

guardrails:
  no_price_in: []
  no_explicit_in: [stage_opener]
  price_caps:
    cap_multiple_of_anchor: 2.0
    hard_cap_cents: 9999