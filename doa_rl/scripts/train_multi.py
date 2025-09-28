import argparse
import os
import logging
import numpy as np
import torch
from transformers import set_seed

from doa_rl.env.dataset import build_dataset
from doa_rl.algos.ppo_runner import PPORunner
from doa_rl.algos.grpo_runner import GRPORunner
from doa_rl.advantage import AdvantageComputer
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config.defaults import NMFConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=str, required=True)
    ap.add_argument("--W", type=str, default=None)
    ap.add_argument("--J", type=int, default=2)
    ap.add_argument("--algo", type=str, choices=["ppo", "grpo"], default="grpo")
    ap.add_argument("--feature", type=str, choices=["patch", "leaf", "scatter", "nmf"], default="patch")
    ap.add_argument("--add_dir_tokens", type=int, default=1)
    ap.add_argument("--num_wavs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s_mode", type=str, choices=["S1", "S2"], default="S1")
    ap.add_argument("--nmf_iter", type=int, default=50)
    ap.add_argument("--nmf_l1", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args(); set_seed(args.seed)

    log_level = logging.DEBUG if args.debug or os.environ.get("RL_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")

    H_store = np.load(args.H)
    H_np = H_store["H"]
    if H_np.shape[0] < H_np.shape[1]:
        H_np_FD = H_np.T.astype(np.float32)
    else:
        H_np_FD = H_np.astype(np.float32)
    W = np.load(args.W)["W"].astype(np.float32) if args.W else None

    ds, exs, dir_vocab = build_dataset(H_np, [None]*args.num_wavs, feature=args.feature,
                                       add_dir_tokens=bool(args.add_dir_tokens), J=args.J,
                                       W=W, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)

    if W is not None:
        H_t = torch.from_numpy(H_np_FD)
        W_t = torch.from_numpy(W)
        ncfg = NMFConfig(beta=0.0, lambda_group=5.0, gamma_sparse=0.1, max_iter=100, device="cpu")
        loc = NMFSoundLocalizer(ncfg)
        loc.load_source_dictionary(W_t)
        loc.load_transfer_functions(H_t)
        adv = AdvantageComputer(loc, W_t, H_t, s_mode=args.s_mode, nmf_iter=args.nmf_iter,
                                nmf_l1=args.nmf_l1, require_s_hat=True)
    else:
        adv = None

    if args.algo=="ppo":
        PPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=args.J, adv=adv)
    else:
        GRPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=args.J, G=6, adv=adv)


if __name__=="__main__":
    main()
