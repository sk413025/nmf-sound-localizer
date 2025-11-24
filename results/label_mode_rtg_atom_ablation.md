# DTMin ablation: label_mode × RTG × atom loss (speech260, M=50, K=5)

Shards: `results/speech260_dtmin_full_w_ranges_mix*` (full+low+mid+high+weak_0_50+weak_140_170+strong_edges). Training: epochs=80–120, batch_size=32, lr=3e-4. Device: MPS. No early-stop. Teacher OMP unchanged (voted full≈0.324).

| label_mode | RTG | atom loss | voted_acc (best/final) | joint_acc (best/final) | Notes |
|------------|-----|-----------|------------------------|------------------------|-------|
| teacher    | ON  | ON        | 0.325 / 0.325          | 0.949 / 0.949          | Baseline (mix) |
| teacher    | OFF | ON        | 0.325 / 0.325          | 0.949 / 0.949          | RTG removal negligible |
| teacher    | ON  | OFF       | 0.376 / 0.325          | 0.029 / 0.021          | Atom loss essential; trajectory imitation collapses |
| angle (GT) | ON  | ON        | 0.949 / 0.949          | 0.966 / 0.959          | Leaks true angle; not tied to OMP quality |
| angle (GT) | OFF | ON        | 0.949 / 0.949          | 0.966 / 0.959          | Same as above; RTG removal negligible when labels are GT |

Logs:
- Baseline (teacher, RTG ON, atom ON): `results/speech260_dtmin_full_w_ranges_mix_training.log/json`
- RTG OFF (teacher): `results/speech260_dtmin_full_w_ranges_mix_nortg_training.log/json`
- Atom OFF (teacher): `results/speech260_dtmin_full_w_ranges_mix_noatom_training.log/json`
- Angle mode RTG ON: `results/speech260_dtmin_full_w_ranges_mix_angle_training.log/json`
- Angle mode RTG OFF: `results/speech260_dtmin_full_w_ranges_mix_nortg_angle_training.log/json`

Takeaways:
- RTG projection is optional for DTMin imitation in this setup (no measurable change).
- Atom loss is critical; disabling it collapses joint/trajectory accuracy even if voted stays ~0.33.
- label_mode='angle' gives a high upper bound (~0.95 voted) by leaking GT; not representative of OMP+DTMin.
