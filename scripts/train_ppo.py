import argparse
from pathlib import Path

import torch

from rl.config import RLConfig
from rl.assets import load_H, load_W
from rl.data import DoADataset, create_dataloader
from rl.policy import DirectionPolicy
from rl.transformer_policy import TransformerPolicy
from rl.text import Vocab, pad_sequences
from features.tokenizers import PatchTokenizer, NMFTokenizer, direction_projection_tokens
from rl.advantage import AdvantageComputer
from rl.buffer import OnPolicyBuffer
from rl.ppo_trainer import PPOTrainer

from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config.defaults import NMFConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-path', type=str, default=RLConfig.tf_path)
    parser.add_argument('--w-path', type=str, default=RLConfig.w_path)
    parser.add_argument('--test-root', type=str, default=RLConfig.test_root)
    parser.add_argument('--epochs', type=int, default=RLConfig.ppo_epochs)
    parser.add_argument('--lr', type=float, default=RLConfig.ppo_lr)
    parser.add_argument('--clip-eps', type=float, default=RLConfig.ppo_clip_eps)
    parser.add_argument('--target-kl', type=float, default=RLConfig.ppo_target_kl)
    parser.add_argument('--entropy-coef', type=float, default=RLConfig.ppo_entropy_coef)
    parser.add_argument('--out-dir', type=str, default='rl_runs')
    parser.add_argument('--s-mode', type=str, default=RLConfig.s_mode, choices=['S1','S2'])
    parser.add_argument('--nmf-iter', type=int, default=RLConfig.nmf_iter)
    parser.add_argument('--nmf-l1', type=float, default=RLConfig.nmf_l1)
    parser.add_argument('--phys', action='store_true')
    parser.add_argument('--feature', type=str, choices=['patch','nmf'], default='patch')
    parser.add_argument('--add-dir-tokens', action='store_true')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--layers', type=int, default=2)
    args = parser.parse_args()

    cfg = RLConfig()
    H, angles = load_H(args.tf_path)
    W = load_W(args.w_path)

    # Sanity checks
    assert H.shape[0] == W.shape[0] == 346, f"Expected F=346, got H {H.shape}, W {W.shape}"
    assert H.shape[1] == 17, f"Expected D=17, got {H.shape[1]}"

    # Localizer setup
    ncfg = NMFConfig(beta=0.0, lambda_group=cfg.lambda_group, gamma_sparse=cfg.gamma_sparse,
                     max_iter=cfg.max_iter, device=cfg.device)
    loc = NMFSoundLocalizer(ncfg)
    loc.load_source_dictionary(W)
    loc.load_transfer_functions(H, angles)

    # Data
    ds = DoADataset(args.test_root, angles.tolist(), fs=cfg.fs, n_fft=cfg.n_fft,
                    window=cfg.window, freq_min=cfg.freq_min, freq_max=cfg.freq_max)
    dl = create_dataloader(ds, batch_size=1, shuffle=True)

    # Advantage computer
    advcomp = AdvantageComputer(loc, W, H, s_mode=args.s_mode,
                                nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1,
                                add_phys_feature=args.phys or cfg.add_phys_feature)

    # Build tokenizer/vocab
    import numpy as np
    H_np = H.numpy(); W_np = W.numpy()
    if args.feature == 'nmf':
        fea = NMFTokenizer(W=W_np, n_iter=args.nmf_iter, mode=args.s_mode, l1=args.nmf_l1)
    else:
        fea = PatchTokenizer()
    token_lists = []
    precomputed = []
    for item in dl:
        Y = item['Y'].squeeze(0).numpy()
        if args.feature == 'nmf':
            toks, _ = fea(Y, H=H_np)
        else:
            toks = fea(Y)
        if args.add_dir_tokens:
            toks = toks + direction_projection_tokens(Y.mean(axis=1), H_np)
        token_lists.append(toks)
        precomputed.append({"tokens": toks, "path": item['path'][0], "Y": item['Y'].squeeze(0)})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocab(); vocab.build(token_lists)
    # Save vocab for reproducible evaluation
    import json
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab.itos, f)

    # Policy (Transformer)
    policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H.shape[1], d_model=args.d_model,
                               nhead=args.nhead, num_layers=args.layers)
    trainer = PPOTrainer(policy, lr=args.lr, clip_eps=args.clip_eps,
                         target_kl=args.target_kl, entropy_coef=args.entropy_coef,
                         device=cfg.device)

    # Training
    buf = OnPolicyBuffer()
    policy.train()
    for epoch in range(args.epochs):
        buf.clear()
        for rec in precomputed:
            ids = vocab.encode(rec["tokens"], add_cls=True)
            input_ids, attn = pad_sequences([ids], vocab.pad_id)
            Y = rec['Y']
            out = advcomp(Y)
            A = out['A']
            with torch.no_grad():
                logits = policy(input_ids, attn)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                adv = A[action.item()].view(1)
            buf.add(logits=logits.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                    advantage=adv.squeeze(0), input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))

        batch_t = buf.to_tensors()
        info = trainer.update(batch_t, epochs=cfg.ppo_epochs)
        print({"epoch": epoch, **info})

    save_path = out_dir / "ppo_policy.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"PPO training finished. Policy saved to {save_path}")


if __name__ == '__main__':
    main()
