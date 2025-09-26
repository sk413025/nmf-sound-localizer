import argparse
from pathlib import Path
import json
from typing import Any, Dict, List
import numpy as np
import torch
from transformers import set_seed
import logging
import time


def _compute_patch_reconstruction(Y_np: np.ndarray, s_hat: np.ndarray,
                                  H_np: np.ndarray, dir_idx: int,
                                  eps: float = 1e-12) -> Dict[str, Any]:
    """Return reconstruction vector and metrics for a given clip."""
    s_hat = np.asarray(s_hat).astype(np.float32)
    H_safe = np.maximum(H_np, eps)
    Hs = H_safe * s_hat[:, None]  # (F, D)
    pi = np.zeros(H_np.shape[1], dtype=np.float32)
    if 0 <= dir_idx < pi.shape[0]:
        pi[dir_idx] = 1.0
    Y_mix = Hs @ pi  # (F,)
    Y_mix = np.maximum(Y_mix, eps)

    Y_mean = np.maximum(Y_np.mean(axis=1), eps)
    mse = float(np.mean((Y_mean - Y_mix) ** 2))
    mae = float(np.mean(np.abs(Y_mean - Y_mix)))
    ratio = np.maximum(Y_mean, eps) / Y_mix
    is_div = float(np.sum(ratio - np.log(ratio) - 1.0))

    metrics = {
        "mse": mse,
        "mae": mae,
        "is_div": is_div,
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
    }
    return {"Y_mix": Y_mix, "metrics": metrics}


def _visualize_reconstructions(samples: List[Dict[str, Any]], output_dir: Path,
                               logger: logging.Logger, eps: float = 1e-12) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Visualization skipped: matplotlib unavailable (%s)", exc)
        return

    import matplotlib.pyplot as plt

    for sample in samples:
        idx = sample["index"]
        Y = sample["Y"]
        Y_mix = sample["Y_mix"]
        path = sample.get("path", f"sample_{idx:02d}")
        metrics = sample.get("metrics", {})

        freq_axis = np.arange(Y.shape[0])
        Y_mean = np.maximum(Y.mean(axis=1), eps)
        Y_mix = np.maximum(Y_mix, eps)
        log_Y = np.log10(Y_mean)
        log_Ymix = np.log10(Y_mix)
        diff = log_Y - log_Ymix

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(freq_axis, log_Y, label='log10 |Y̅|', color='tab:blue', linewidth=1.2)
        axes[0].plot(freq_axis, log_Ymix, label='log10 |Ŷ̅|', color='tab:orange', linewidth=1.2)
        axes[0].set_ylabel('Amplitude (log10)')
        axes[0].set_title('Average magnitude per frequency bin')
        axes[0].legend(loc='best', fontsize=9)

        axes[1].plot(freq_axis, diff, label='log10 |Y̅| - log10 |Ŷ̅|', color='tab:green', linewidth=1.0)
        axes[1].axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].set_xlabel('Frequency bin index')
        axes[1].set_ylabel('Difference (log10)')
        caption = (f"MSE={metrics.get('mse', 0):.3e}, MAE={metrics.get('mae', 0):.3e}\n"
                   f"IS={metrics.get('is_div', 0):.3e}, ratio[min,max]="
                   f"({metrics.get('ratio_min', 0):.3g}, {metrics.get('ratio_max', 0):.3g})")
        axes[1].text(0.01, 0.95, caption, transform=axes[1].transAxes,
                     fontsize=8, color='black', verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

        fig.suptitle(f'Patch Reconstruction — sample {idx}')
        fig.tight_layout()
        out_path = output_dir / f"patch_recon_{idx:02d}_{Path(path).stem}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved patch reconstruction visualization: %s", out_path)

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
    # Tokenizer controls (increase sequence)
    ap.add_argument("--patch-fp", type=int, default=16, help="PatchTokenizer: freq patch size")
    ap.add_argument("--patch-np", type=int, default=10, help="PatchTokenizer: time patch size")
    ap.add_argument("--nmf-topk", type=int, default=12, help="NMFTokenizer: top-k atoms (larger → longer sequence)")
    ap.add_argument("--dir-topm", type=int, default=None, help="Direction tokens: keep top-M (default all)")
    ap.add_argument("--dir-alphas", type=str, default="", help="Comma list of extra alpha values to add more direction tokens, e.g., '0.5,2.0'")
    ap.add_argument("--add_dir_tokens", type=int, default=1)
    ap.add_argument("--num_wavs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s_mode", type=str, choices=["S1", "S2"], default="S1")
    ap.add_argument("--nmf_iter", type=int, default=50)
    ap.add_argument("--nmf_l1", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--policy-batch", type=int, default=8, help="Batch size for Transformer policy forward")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--viz-samples", type=int, default=3,
                    help="Number of reconstruction visualizations to save (patch mode)")
    args = ap.parse_args()
    # Log to console and file under logs/
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"doa_rl_run_{ts}.log"
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')])
    logger = logging.getLogger(__name__)
    # Silence overly verbose nmf_localizer logs for console; still captured in file
    logging.getLogger("nmf_localizer").setLevel(logging.WARNING)
    logging.getLogger("nmf_localizer.core.localizer").setLevel(logging.WARNING)
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
        logger.info("Angles shape=%s values=%s", angles_t.shape, angles_t.tolist())
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
        # Build tokenizer with requested sequence controls
        if args.feature == 'nmf':
            fea = NMFTokenizer(W=W_np, n_iter=args.nmf_iter, mode=args.s_mode, l1=args.nmf_l1, topk=args.nmf_topk)
        else:
            fea = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
        token_lists = []; precomputed = []
        # Parse extra alphas for more direction tokens
        extra_alphas = []
        if args.dir_alphas.strip():
            try:
                extra_alphas = [float(x) for x in args.dir_alphas.split(',') if x.strip()]
            except Exception:
                extra_alphas = []
        recon_samples: List[Dict[str, Any]] = []
        recon_metrics: List[Dict[str, float]] = []
        nmf_aux = None
        if args.feature == 'patch':
            nmf_aux = NMFTokenizer(W=W_np, n_iter=args.nmf_iter, mode=args.s_mode,
                                   l1=args.nmf_l1, topk=args.nmf_topk)

        for idx, item in enumerate(dl):
            Y_t = item['Y'].squeeze(0)
            Y = Y_t.numpy()
            logger.info("train_single: sample %d Y shape=%s", idx, Y.shape)

            if args.feature == 'nmf':
                toks, s_hat_np = fea(Y, H=H_np)
            else:
                toks = fea(Y)
                # Auxiliary NMF for reconstruction/advantage
                assert nmf_aux is not None
                _, s_hat_np = nmf_aux(Y, H=H_np)

            logger.info("train_single: sample %d content tokens=%d", idx, len(toks))
            if args.add_dir_tokens:
                base = direction_projection_tokens(Y.mean(axis=1), H_np, topM=args.dir_topm)
                toks = toks + base
                logger.info("train_single: sample %d dir tokens alpha=1.0 count=%d", idx, len(base))
                # Add more physics tokens with different alpha to lengthen the sequence
                for a in extra_alphas:
                    extra_tokens = direction_projection_tokens(Y.mean(axis=1), H_np, alpha=a, topM=args.dir_topm)
                    toks = toks + extra_tokens
                    logger.info("train_single: sample %d dir tokens alpha=%.2f count=%d", idx, a, len(extra_tokens))
            token_lists.append(toks)
            ai = item['angle_index']
            try:
                gt = int(ai)
            except Exception:
                gt = int(ai[0])
            logger.info("train_single: sample %d ground-truth index=%d angle=%s", idx, gt, item['angle_deg'])

            s_hat_t = torch.from_numpy(s_hat_np).float()
            rec = {"tokens": toks,
                   "Y": Y_t,
                   "path": item['path'][0],
                   "gt": gt,
                   "s_hat": s_hat_t}
            precomputed.append(rec)

            if args.feature == 'patch':
                recon = _compute_patch_reconstruction(Y, s_hat_np, H_np, gt)
                metrics = recon["metrics"]
                recon_metrics.append(metrics)
                logger.info(
                    "Patch reconstruction sample %d: mse=%.3e mae=%.3e IS=%.3e ratio[min=%.3g max=%.3g]",
                    idx, metrics["mse"], metrics["mae"], metrics["is_div"],
                    metrics["ratio_min"], metrics["ratio_max"])
                if len(recon_samples) < max(0, args.viz_samples):
                    recon_samples.append({
                        "index": idx,
                        "Y": Y,
                        "Y_mix": recon["Y_mix"],
                        "path": item['path'][0],
                        "metrics": metrics,
                    })
        if args.feature == 'patch' and recon_metrics:
            mse_vals = np.array([m["mse"] for m in recon_metrics], dtype=np.float32)
            mae_vals = np.array([m["mae"] for m in recon_metrics], dtype=np.float32)
            is_vals = np.array([m["is_div"] for m in recon_metrics], dtype=np.float32)
            logger.info(
                "Patch reconstruction summary: MSE mean=%.3e std=%.3e | MAE mean=%.3e | IS mean=%.3e",
                float(mse_vals.mean()), float(mse_vals.std()),
                float(mae_vals.mean()), float(is_vals.mean()))
        if args.feature == 'patch' and recon_samples:
            viz_dir = Path("outputs") / "patch_recon"
            _visualize_reconstructions(recon_samples, viz_dir, logger)

        vocab = Vocab(); vocab.build(token_lists)
        logger.info("train_single: vocab size=%d pad_id=%d cls_id=%d", len(vocab.itos), vocab.pad_id, vocab.cls_id)
        policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H_t.shape[1])
        from doa_rl.training.ppo_trainer import PPOTrainer
        from doa_rl.training.buffer import OnPolicyBuffer
        trainer = PPOTrainer(policy, lr=1e-3, clip_eps=0.2, target_kl=0.02, entropy_coef=0.01, device=str(device))
        logs = []
        adv_evolution = {"adv_sampled_mean": [], "adv_sampled_median": [], "adv_max_mean": [], "adv_mean_mean": []}
        acc_history = []
        for ep in range(args.epochs):
            buf = OnPolicyBuffer(); buf.clear()
            adv_samples = []
            adv_maxes = []
            adv_means = []
            # Mini-batch policy forward for better device utilization
            B = max(1, int(args.policy_batch))
            for i0 in range(0, len(precomputed), B):
                chunk = precomputed[i0:i0+B]
                id_seqs = [vocab.encode(rec["tokens"], add_cls=True) for rec in chunk]
                input_ids, attn = pad_sequences(id_seqs, vocab.pad_id)
                logger.info("train_single: batch %d input_ids shape=%s attention_mask shape=%s",
                            i0 // B, input_ids.shape, attn.shape)
                with torch.no_grad():
                    logits = policy(input_ids.to(device), attn.to(device))  # (b, D)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()  # (b,)
                    logps = dist.log_prob(actions)
                    pi = torch.softmax(logits, dim=-1).cpu()
                logger.info("train_single: logits shape=%s actions shape=%s logps shape=%s",
                            logits.shape, actions.shape, logps.shape)
                # Per-sample bookkeeping + buffer append
                for j, rec in enumerate(chunk):
                    outA = adv(rec['Y'], pi=pi[j], s_hat=rec.get('s_hat'))
                    A = outA['A']
                    logger.info("train_single: rec advantage shape=%s", A.shape)
                    adv_val = A[actions[j].item()].view(1)
                    buf.add(logits=logits[j].cpu(), action=actions[j].cpu(), logp=logps[j].cpu(),
                            advantage=adv_val.squeeze(0), input_ids=input_ids[j].cpu(), attention_mask=attn[j].cpu())
                    adv_samples.append(float(adv_val.item()))
                    adv_maxes.append(float(torch.max(A).item()))
                    adv_means.append(float(torch.mean(A).item()))
            info = trainer.update(buf.to_tensors(), epochs=1)
            import statistics as _st
            a_mean = float(_st.mean(adv_samples)) if adv_samples else 0.0
            a_med = float(_st.median(adv_samples)) if adv_samples else 0.0
            amax_mean = float(_st.mean(adv_maxes)) if adv_maxes else 0.0
            amean_mean = float(_st.mean(adv_means)) if adv_means else 0.0
            adv_evolution["adv_sampled_mean"].append(a_mean)
            adv_evolution["adv_sampled_median"].append(a_med)
            adv_evolution["adv_max_mean"].append(amax_mean)
            adv_evolution["adv_mean_mean"].append(amean_mean)
            # Accuracy evaluation on CPU to avoid MPS nested-tensor issues
            try:
                from copy import deepcopy as _dc
                policy_eval = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H_t.shape[1]).to("cpu")
                policy_eval.load_state_dict(_dc(policy.state_dict()))
                policy_eval.eval()
                correct = 0; n = 0
                with torch.no_grad():
                    for rec in precomputed:
                        ids = vocab.encode(rec["tokens"], add_cls=True)
                        input_ids, attn = pad_sequences([ids], vocab.pad_id)
                        logits = policy_eval(input_ids, attn)
                        pred = int(torch.argmax(logits, dim=-1).item())
                        if "gt" in rec and rec["gt"] >= 0:
                            correct += int(pred == int(rec["gt"]))
                            n += 1
                acc_top1 = (100.0 * correct / max(n, 1)) if n else 0.0
            except Exception as _e:
                acc_top1 = 0.0
                logger.warning("Eval failed at epoch %d: %s", ep, str(_e))
            acc_history.append(acc_top1)
            row = {"mode": "real", "epoch": ep, **info, "acc_top1": acc_top1,
                   "adv_sampled_mean": a_mean, "adv_sampled_median": a_med,
                   "adv_max_mean": amax_mean, "adv_mean_mean": amean_mean}
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
            # Advantage convergence plot
            fig2, ax = plt.subplots(figsize=(6,4))
            ax.plot(xs, adv_evolution["adv_sampled_mean"], '-o', label='adv_sampled_mean')
            ax.plot(xs, adv_evolution["adv_sampled_median"], '-s', label='adv_sampled_median')
            ax.plot(xs, adv_evolution["adv_max_mean"], '-x', label='adv_max_mean')
            ax.set_xlabel('epoch'); ax.set_ylabel('advantage')
            ax.legend(loc='best')
            fig2.tight_layout(); fig2.savefig(out_dir / 'adv_convergence.png', dpi=150)
            plt.close(fig2)
            # Accuracy convergence plot
            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.plot(xs, acc_history, '-o', label='acc_top1 (%)')
            ax3.set_xlabel('epoch'); ax3.set_ylabel('top-1 accuracy (%)')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='best')
            fig3.tight_layout(); fig3.savefig(out_dir / 'accuracy_convergence.png', dpi=150)
            plt.close(fig3)
            # Save accuracy series
            with open(out_dir / 'accuracy_per_epoch.json', 'w') as f_acc:
                json.dump({"epochs": xs, "acc_top1": acc_history}, f_acc)
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
