# Runtime Substrate Package

This package metadata exists only to expose the maintained runtime substrate for the manuscript worktree.

The actively maintained package surface is intentionally small:

- `nmf_localizer.config`
- `nmf_localizer.core.data_processor`
- `nmf_localizer.core.stft_unified_processor`
- `nmf_localizer.core.transfer_functions`
- `nmf_localizer.core.usm_trainer`
- `nmf_localizer.utils.audio_utils`
- `doa_rl.data`
- `doa_rl.omp.soft_omp`

This is not the branch identity and it is not the primary workflow entrypoint. Use the branch constitution, governance contracts, and paper-facing commands first.

Legacy pipeline, oracle, DT, reconstruction, and visualization paths have been quarantined under `legacy/runtime/`.
