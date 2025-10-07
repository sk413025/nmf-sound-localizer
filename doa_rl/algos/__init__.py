"""High-level runners wrapping PPO/GRPO with tokenizer+Transformer.

These runners accept prompts and per-sample extras (Y, s_hat, H, dirs)
from doa_rl.env.dataset.build_dataset and train a Transformer policy.
"""

