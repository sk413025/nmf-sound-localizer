import argparse
from pathlib import Path
import json
import numpy as np
import torch
from transformers import set_seed
import logging

from doa_rl.env.dataset import build_dataset
from doa_rl.algos.ppo_runner import PPORunner
from doa_rl.algos.grpo_runner import GRPORunner
from doa_rl.advantage import AdvantageComputer
from doa_rl.assets import load_H, load_W
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from features.tokenizers import PatchTokenizer, NMFTokenizer, direction_projection_tokens
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config.defaults import NMFConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=str, required=False)
    ap.add_argument("--W", type=str, default=None)
    ap.add_argument("--tf-path", type=str, default=None)
    ap.add_argument("--w-path", type=str, default=None)
    ap.add_argument("--test-root", type=str, default=None)
    ap.add_argument("--algo", type=str, choices=["ppo", "grpo"], default="ppo")
    ap.add_argument("--feature", type=str, choices=["patch", "leaf", "scatter", "nmf"], default="patch")
    ap.add_argument("--add_dir_tokens", type=int, default=1)
    ap.add_argument("--num_wavs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s_mode", type=str, choices=["S1", "S2"], default="S1")
    ap.add_argument("--nmf_iter", type=int, default=50)
    ap.add_argument("--nmf_l1", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

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

    if args.tf_path and args.w_path and args.test_root:
        # Real-data mode: direct .pth assets + dataset root
        H_t, angles_t = load_H(args.tf_path)
        W_t = load_W(args.w_path)
        logger.info("Assets loaded: H=%s, W=%s, test_root=%s", tuple(H_t.shape), tuple(W_t.shape), args.test_root)
        # For stability, keep localizer on CPU if device is MPS
        loc_device = "cpu" if str(device) == "mps" else str(device)
        ncfg = NMFConfig(beta=0.0, lambda_group=5.0, gamma_sparse=0.1, max_iter=100, device=loc_device)
        loc = NMFSoundLocalizer(ncfg)
        loc.load_source_dictionary(W_t)
        loc.load_transfer_functions(H_t, angles_t)
        adv = AdvantageComputer(loc, W_t, H_t, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)

        ds_real = DoADataset(args.test_root, angles_t.tolist())
        dl = create_dataloader(ds_real, batch_size=1, shuffle=True)
        logger.info("Dataset size: %d clips; device=%s (localizer on %s)", len(ds_real), device, loc_device)

        H_np = H_t.numpy(); W_np = W_t.numpy()
        fea = NMFTokenizer(W=W_np, n_iter=args.nmf_iter, mode=args.s_mode, l1=args.nmf_l1) if args.feature == 'nmf' else PatchTokenizer()
        token_lists = []; precomputed = []
        for item in dl:
            Y = item['Y'].squeeze(0).numpy()
            toks = (fea(Y, H=H_np)[0] if args.feature == 'nmf' else fea(Y))
            if args.add_dir_tokens:
                toks = toks + direction_projection_tokens(Y.mean(axis=1), H_np)
            token_lists.append(toks)
            ai = item['angle_index']
            try:
                gt = int(ai)
            except Exception:
                gt = int(ai[0])
            precomputed.append({"tokens": toks, "Y": item['Y'].squeeze(0), "path": item['path'][0], "gt": gt})
        vocab = Vocab(); vocab.build(token_lists)
        policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H_t.shape[1])
        from doa_rl.training.ppo_trainer import PPOTrainer
        from doa_rl.training.buffer import OnPolicyBuffer
        trainer = PPOTrainer(policy, lr=1e-3, clip_eps=0.2, target_kl=0.02, entropy_coef=0.0, device=str(device))
        logs = []
        for ep in range(args.epochs):
            buf = OnPolicyBuffer(); buf.clear()
            for rec in precomputed:
                ids = vocab.encode(rec["tokens"], add_cls=True)
                input_ids, attn = pad_sequences([ids], vocab.pad_id)
                out = adv(rec['Y'])
                A = out['A']
                with torch.no_grad():
                    logits = policy(input_ids.to(device), attn.to(device))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample(); logp = dist.log_prob(action)
                    adv_val = A[action.item()].view(1)
                buf.add(logits=logits.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                        advantage=adv_val.squeeze(0), input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))
            info = trainer.update(buf.to_tensors(), epochs=1)
            row = {"mode": "real", "epoch": ep, **info}
            logs.append(row)
            print(row)
            logger.info("Epoch %d summary: %s", ep, row)
        out_dir = Path("rl_runs"); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab.itos, f)
        torch.save(policy.state_dict(), out_dir / "ppo_policy.pt")
        # Save training curve
        with open(out_dir / "ppo_train_log.json", "w") as f:
            json.dump(logs, f)
        try:
            import matplotlib.pyplot as plt
            xs = [r["epoch"] for r in logs]
            ys = [r["policy_loss"] for r in logs]
            yk = [r.get("kl", 0.0) for r in logs]
            fig, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(xs, ys, '-o', label='policy_loss')
            ax1.set_xlabel('epoch'); ax1.set_ylabel('policy_loss')
            ax2 = ax1.twinx(); ax2.plot(xs, yk, '-x', color='orange', label='kl')
            ax2.set_ylabel('kl')
            fig.tight_layout(); fig.savefig(out_dir / 'ppo_convergence.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            print({"warn": "matplotlib unavailable", "detail": str(e)})
        # Evaluate top-1 accuracy after training
        try:
            policy.eval()
            correct = 0; n = 0
            with torch.no_grad():
                for rec in precomputed:
                    ids = vocab.encode(rec["tokens"], add_cls=True)
                    input_ids, attn = pad_sequences([ids], vocab.pad_id)
                    logits = policy(input_ids.to(device), attn.to(device))
                    pred = int(torch.argmax(logits, dim=-1).item())
                    gt = int(rec["gt"]) if "gt" in rec else -1
                    if gt >= 0:
                        correct += int(pred == gt); n += 1
            acc = 100.0 * correct / max(n, 1)
            print({"eval_top1_acc": acc, "n": n})
            with open(out_dir / 'ppo_eval.json', 'w') as f:
                json.dump({"acc_top1": acc, "n": n}, f)
        except Exception as e:
            print({"warn": "eval_failed", "detail": str(e)})
    else:
        H_store = np.load(args.H)
        H_np = H_store["H"]
        if H_np.shape[0] < H_np.shape[1]:
            H_np_FD = H_np.T.astype(np.float32)
        else:
            H_np_FD = H_np.astype(np.float32)
        W_np = np.load(args.W)["W"].astype(np.float32) if args.W else None
        ds, exs, dir_vocab = build_dataset(H_np, [None] * args.num_wavs, feature=args.feature,
                                           add_dir_tokens=bool(args.add_dir_tokens), J=1,
                                           W=W_np, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)
        if W_np is not None:
            H_t = torch.from_numpy(H_np_FD); W_t = torch.from_numpy(W_np)
            ncfg = NMFConfig(beta=0.0, lambda_group=5.0, gamma_sparse=0.1, max_iter=100, device=str(device))
            loc = NMFSoundLocalizer(ncfg)
            loc.load_source_dictionary(W_t)
            loc.load_transfer_functions(H_t)
            adv = AdvantageComputer(loc, W_t, H_t, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)
        else:
            adv = None
        if args.algo == "ppo":
            PPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=1, adv=adv)
        else:
            GRPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=1, G=4, adv=adv)


if __name__ == "__main__":
    main()
