---
type: pricing_profile
name: pricing_gold_v0
version: 0.1
tags: [pricing, gold]

ppv_floor_cents: 799
anchor_cents: 1299
step_up_cents: 400
bundles:
  - { items: 2, price_cents: 1999 }
  - { items: 3, price_cents: 2799 }
discount_rules:
  - { if: "db.ppv_unpaid_count >= 2", percent_off: 10 }