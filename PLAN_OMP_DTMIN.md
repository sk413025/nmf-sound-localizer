# OMP teacher → DTMin strengthening plan

Goals: keep the OMP teacher + DTMin architecture intact while improving teacher quality and reducing trajectory error accumulation on speech260.

## Teacher-side actions
- Preserve physical alignment: fs=16k, n_fft=2048, band 300–3000 Hz, ensure Y.F == H.F == W.F, keep normalize_w/d.
- Add energy-based early stop (still OMP greedy): stop when relative residual energy < eps; enforce min_steps=1, max_steps=K; log actual steps per sample.
- Keep full W (M=50) for speech; if compression is needed, prefer SVD over k-means (but avoid for speech unless forced).
- Per-angle diagnostics: log first_step_acc/voted_acc per angle to identify weak angles; avoid using configs with mean voted < 0.3.
- Optional score temperature (τ≈1) only if needed for stability (start with τ=1 no change).

## Test sequence (current run)
1) Implement energy early-stop in generator `_run_omp` + logging of actual steps.
2) Regenerate speech260 shards (full, low, mid, high) with M=50, K=5, early-stop on.
3) Train DTMin with domain-rand shards (same training command as before) and compare teacher/student voted/joint to the previous runs.
4) If teacher still weak per angle, consider targeted no-compression for bad angles or revisiting W training (separate task).

Artifacts to capture per run
- Generation logs, summaries with per-angle metrics and step histograms.
- Training logs (DTMin) with voted/expert/atom/joint curves.
