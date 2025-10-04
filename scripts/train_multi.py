import argparse
import torch

from doa_rl.assets import load_H, load_W
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.advantage import AdvantageComputer
from doa_rl.training.buffer import GroupOnPolicyBuffer
from doa_rl.training.grpo_trainer import GRPOTrainer
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from doa_rl.features import PatchTokenizer, direction_projection_tokens

from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config.defaults import NMFConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-path", type=str, required=True, help="Path to transfer function .pth file")
    ap.add_argument("--w-path", type=str, required=True, help="Path to source dictionary .pth file")
    ap.add_argument("--test-root", type=str, required=True, help="Path to test data root directory")
    ap.add_argument("--feature", type=str, choices=["patch"], default="patch")
    ap.add_argument("--add-dir-tokens", action="store_true")
    ap.add_argument("--group-size", type=int, default=4, help="GRPO group size")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--s-mode", type=str, choices=["S1", "S2"], default="S1")
    ap.add_argument("--nmf-iter", type=int, default=50)
    ap.add_argument("--nmf-l1", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--beta-kl", type=float, default=0.01, help="KL divergence penalty")
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--fs", type=int, default=48000, help="Sampling rate")
    ap.add_argument("--n-fft", type=int, default=2048, help="FFT size")
    ap.add_argument("--window", type=str, default="hann", help="Window function")
    ap.add_argument("--freq-min", type=float, default=300.0, help="Min frequency")
    ap.add_argument("--freq-max", type=float, default=3000.0, help="Max frequency")
    args = ap.parse_args()

    # Select device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    H, angles = load_H(args.tf_path)
    W = load_W(args.w_path)

    # Device for localizer (use CPU if MPS for stability)
    loc_device = "cpu" if str(device) == "mps" else str(device)
    
    ncfg = NMFConfig(beta=0.0, lambda_group=5.0, gamma_sparse=0.1,
                     max_iter=100, device=loc_device)
    loc = NMFSoundLocalizer(ncfg)
    loc.load_source_dictionary(W)
    loc.load_transfer_functions(H, angles)

    ds = DoADataset(args.test_root, angles.tolist(), fs=args.fs, n_fft=args.n_fft,
                    window=args.window, freq_min=args.freq_min, freq_max=args.freq_max)
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
        rec = {"tokens": toks, "Y": item['Y'].squeeze(0), "path": item['path'][0]}
        if s_hat_np is not None:
            import torch
            rec["s_hat"] = torch.from_numpy(s_hat_np.astype(np.float32))
        precomputed.append(rec)
    vocab = Vocab(); vocab.build(token_lists)

    policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H.shape[1])
    trainer = GRPOTrainer(policy, lr=args.lr, clip_eps=args.clip_eps, beta_kl=args.beta_kl,
                          device=device)

    buf = GroupOnPolicyBuffer()
    for epoch in range(args.epochs):
        buf.clear()
        for rec in precomputed:
            ids = vocab.encode(rec["tokens"], add_cls=True)
            input_ids, attn = pad_sequences([ids], vocab.pad_id)
            s_in = rec.get('s_hat')
            out = advcomp(rec['Y'], s_hat=s_in)
            A = out['A']
            logits_old = None
            for _ in range(args.group_size):
                with torch.no_grad():
                    logits = policy(input_ids, attn)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    adv = A[action.item()].view(1)
                if logits_old is None:
                    logits_old = logits
                buf.add(logits=logits_old.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                        advantage_raw=adv.squeeze(0), group_id=rec['path'],
                        input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))
        batch = buf.to_tensors()
        info = trainer.update(batch, epochs=4)  # Inner update epochs
        print({"epoch": epoch, **info})


if __name__ == "__main__":
    main()
