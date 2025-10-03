import argparse
import torch

from rl.config import RLConfig
from rl.assets import load_H, load_W
from rl.data import DoADataset, create_dataloader
from rl.advantage import AdvantageComputer
from rl.buffer import OnPolicyBuffer
from rl.ppo_trainer import PPOTrainer
from rl.transformer_policy import TransformerPolicy
from rl.text import Vocab, pad_sequences
from doa_rl.features import PatchTokenizer, direction_projection_tokens

from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config.defaults import NMFConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-path", type=str, default=RLConfig.tf_path)
    ap.add_argument("--w-path", type=str, default=RLConfig.w_path)
    ap.add_argument("--test-root", type=str, default=RLConfig.test_root)
    ap.add_argument("--epochs", type=int, default=RLConfig.ppo_epochs)
    ap.add_argument("--feature", type=str, choices=["patch"], default="patch")
    ap.add_argument("--add-dir-tokens", action="store_true")
    ap.add_argument("--s-mode", type=str, choices=["S1", "S2"], default=RLConfig.s_mode)
    ap.add_argument("--nmf-iter", type=int, default=RLConfig.nmf_iter)
    ap.add_argument("--nmf-l1", type=float, default=RLConfig.nmf_l1)
    args = ap.parse_args()

    cfg = RLConfig()
    H, angles = load_H(args.tf_path)
    W = load_W(args.w_path)

    ncfg = NMFConfig(beta=0.0, lambda_group=cfg.lambda_group, gamma_sparse=cfg.gamma_sparse,
                     max_iter=cfg.max_iter, device=cfg.device)
    loc = NMFSoundLocalizer(ncfg)
    loc.load_source_dictionary(W)
    loc.load_transfer_functions(H, angles)

    ds = DoADataset(args.test_root, angles.tolist(), fs=cfg.fs, n_fft=cfg.n_fft,
                    window=cfg.window, freq_min=cfg.freq_min, freq_max=cfg.freq_max)
    dl = create_dataloader(ds, batch_size=1, shuffle=True)

    advcomp = AdvantageComputer(loc, W, H, s_mode=args.s_mode,
                                nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)

    # Build tokens
    import numpy as np
    H_np = H.numpy(); W_np = W.numpy()
    fea = PatchTokenizer()
    token_lists = []
    precomputed = []
    for item in dl:
        Y = item['Y'].squeeze(0).numpy()
        toks = fea(Y)
        s_hat_np = None
        if args.add_dir_tokens:
            toks = toks + direction_projection_tokens(Y.mean(axis=1), H_np)
        token_lists.append(toks)
        rec = {"tokens": toks, "Y": item['Y'].squeeze(0)}
        if s_hat_np is not None:
            rec["s_hat"] = torch.from_numpy(s_hat_np.astype(np.float32))
        precomputed.append(rec)
    vocab = Vocab(); vocab.build(token_lists)

    policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H.shape[1])
    trainer = PPOTrainer(policy, lr=cfg.ppo_lr, clip_eps=cfg.ppo_clip_eps,
                         target_kl=cfg.ppo_target_kl, entropy_coef=cfg.ppo_entropy_coef,
                         device=cfg.device)

    buf = OnPolicyBuffer()
    for epoch in range(args.epochs):
        buf.clear()
        for rec in precomputed:
            ids = vocab.encode(rec["tokens"], add_cls=True)
            input_ids, attn = pad_sequences([ids], vocab.pad_id)
            # If we already computed ŝ with W, provide it to AdvantageComputer
            s_in = rec.get('s_hat')
            out = advcomp(rec['Y'], s_hat=s_in)
            A = out['A']
            with torch.no_grad():
                logits = policy(input_ids, attn)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                adv = A[action.item()].view(1)
            buf.add(logits=logits.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                    advantage=adv.squeeze(0), input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))
        batch = buf.to_tensors()
        info = trainer.update(batch, epochs=cfg.ppo_epochs)
        print({"epoch": epoch, **info})


if __name__ == "__main__":
    main()
