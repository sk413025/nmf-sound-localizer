# Quarantined Runtime

This directory stores runtime modules and scripts that are no longer part of the active TF + USM + soft-OMP support layer.

Typical reasons for quarantine:

- broken imports or stale package assumptions
- DT, oracle, or teacher-alignment paths outside the current paper workflow
- heavy reconstruction, visualization, or analysis code with no active caller
- older high-level pipeline surfaces that would recreate a code-first branch identity

Do not route branch entrypoints or active contracts through this directory.
