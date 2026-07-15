"""Diagnostic v2: Force-load trained FinePointMatcher, compare PCA vs learned equivariant vectors.
Oracle pose table, channel collapse stats for learned vectors only."""
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fangzhouc/alignmodel702/.worktrees/exclusive-region-labels")
from cryo_registration.config import ModelConfig, RegistrationConfig
from cryo_registration.model import (
    ProteinRegistrationModel, apply_transform,
    _equivariant_svd_rotation, weighted_procrustes,
)
from cryo_registration.train import _seed_everything, _move_to_device
from cryo_registration.train_fine_mpn import build_hierarchical_refiner
from cryo_registration.training import random_rigid_transform, inverse_transform
from cryo_registration.fine_point_matching import FinePointMatcher

torch.set_grad_enabled(False)

def compute_rre(R_pred, R_gt):
    rel = R_pred @ R_gt.T
    c = ((torch.trace(rel) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).cpu())

def channel_collapse_stats(vectors, label=""):
    if vectors is None or vectors.numel() == 0: return None
    v = F.normalize(vectors.float(), dim=-1)
    C = v.shape[1]
    mask = torch.triu(torch.ones(C, C, device=v.device), diagonal=1).bool()
    cos_matrix = torch.einsum("nic,njc->nij", v, v).abs()
    abs_cos = cos_matrix[:, mask].mean(dim=1)
    gram = torch.einsum("nic,njc->nij", v, v)
    eigenvalues = torch.linalg.eigvalsh(gram.float())
    eff_rank = eigenvalues.sum(dim=1) / eigenvalues.max(dim=1).values.clamp_min(1e-8)
    sv_ratios = []
    for i in range(len(vectors)):
        _, S, _ = torch.linalg.svd(vectors[i].float(), full_matrices=False)
        sv_ratios.append((
            float(S[1]/S[0].clamp_min(1e-8)) if len(S)>=2 else 0.0,
            float(S[2]/S[0].clamp_min(1e-8)) if len(S)>=3 else 0.0))
    sv2 = np.array([r[0] for r in sv_ratios])
    sv3 = np.array([r[1] for r in sv_ratios])
    return {
        "label": label,
        "num_points": len(vectors),
        "mean_abs_cosine": float(abs_cos.mean()),
        "median_abs_cosine": float(abs_cos.median()),
        "frac_cosine_gt_08": float((abs_cos > 0.8).float().mean()),
        "mean_effective_rank": float(eff_rank.mean()),
        "median_effective_rank": float(eff_rank.median()),
        "frac_rank_lt_2": float((eff_rank < 2.0).float().mean()),
        "mean_sigma2_sigma1": float(np.mean(sv2)),
        "median_sigma2_sigma1": float(np.median(sv2)),
        "mean_sigma3_sigma1": float(np.mean(sv3)),
        "median_sigma3_sigma1": float(np.median(sv3)),
        "frac_sigma2_lt_01": float(np.mean(sv2 < 0.1)),
        "frac_sigma3_lt_005": float(np.mean(sv3 < 0.05)),
    }

def run_single_hypothesis(src_vecs, tgt_vecs, R_gt):
    """SVD rotation from a single pair of equivariant vector sets."""
    try:
        R_pred, sv = _equivariant_svd_rotation(src_vecs, tgt_vecs)
        if R_pred is not None and torch.isfinite(sv).all() and sv[0] > 1e-8:
            return compute_rre(R_pred, R_gt)
    except RuntimeError:
        pass
    return None

def run_multi_hypothesis(src_vecs_list, tgt_vecs_list, R_gt):
    """SVD rotation from accumulated covariance over multiple correspondences."""
    if len(src_vecs_list) == 0: return None
    H = torch.zeros(3, 3, dtype=torch.float32, device=src_vecs_list[0].device)
    for sv, tv in zip(src_vecs_list, tgt_vecs_list):
        H += sv.T @ tv
    try:
        U, S, Vh = torch.linalg.svd(H.float())
        corr = torch.eye(3, dtype=torch.float32, device=H.device)
        corr[-1, -1] = torch.sign(torch.det(Vh.T @ U.T))
        R_multi = Vh.T @ corr @ U.T
        return compute_rre(R_multi.to(src_vecs_list[0].dtype), R_gt)
    except RuntimeError:
        return None

def oracle_pose_table(src_pts, tgt_pts, src_vecs_pca, tgt_vecs_pca,
                      src_vecs_learned, tgt_vecs_learned,
                      src_desc, tgt_desc, gt, scale):
    """Compare PCA vs learned vs oracle vectors on same data."""
    results = {}
    R_gt = gt[:3, :3]; t_gt = gt[:3, 3]
    # GT correspondences (3A threshold in Angstrom)
    aligned = apply_transform(src_pts, gt)
    dists = torch.cdist(aligned, tgt_pts) * scale
    gt_mask = dists <= 3.0
    gt_src, gt_tgt = torch.nonzero(gt_mask, as_tuple=True)
    # Predicted correspondences (descriptor nearest neighbor)
    sim = src_desc @ tgt_desc.T
    _, pred_tgt = sim.topk(1, dim=1)
    pred_src = torch.arange(len(src_desc), device=src_desc.device)
    pred_tgt = pred_tgt.squeeze(-1)

    def eval_vectors(vec_src, vec_tgt, label, corr_src, corr_tgt):
        if vec_src is None or vec_tgt is None: return
        n = min(len(corr_src), 32)
        if n < 3: return
        # Single-point
        single_rres = []
        for i in range(n):
            rre = run_single_hypothesis(vec_src[corr_src[i]], vec_tgt[corr_tgt[i]], R_gt)
            if rre is not None: single_rres.append(rre)
        if single_rres:
            results[f"{label}_single_median_rre"] = float(np.median(single_rres))
            results[f"{label}_single_mean_rre"] = float(np.mean(single_rres))
        # Multi-point
        src_list = [vec_src[corr_src[i]] for i in range(n)]
        tgt_list = [vec_tgt[corr_tgt[i]] for i in range(n)]
        rre_m = run_multi_hypothesis(src_list, tgt_list, R_gt)
        if rre_m is not None: results[f"{label}_multi_rre"] = rre_m

    # PCA vectors with GT corr
    eval_vectors(src_vecs_pca, tgt_vecs_pca, "pca_gt", gt_src, gt_tgt)
    # PCA vectors with predicted corr
    eval_vectors(src_vecs_pca, tgt_vecs_pca, "pca_pred", pred_src, pred_tgt)
    # Learned vectors with GT corr
    eval_vectors(src_vecs_learned, tgt_vecs_learned, "learned_gt", gt_src, gt_tgt)
    # Learned vectors with predicted corr
    eval_vectors(src_vecs_learned, tgt_vecs_learned, "learned_pred", pred_src, pred_tgt)

    # Oracle: GT corr + Procrustes on coordinates
    if len(gt_src) >= 3:
        wp = weighted_procrustes(
            src_pts[gt_src], tgt_pts[gt_tgt],
            torch.ones(len(gt_src), device=src_pts.device))
        results["oracle_procrustes_rre"] = compute_rre(wp[:3,:3], R_gt)
        results["oracle_procrustes_rte"] = float(
            (torch.linalg.norm(wp[:3,3]-t_gt)*scale).cpu())
    results["num_gt_corr"] = len(gt_src)
    results["num_pred_corr"] = len(pred_src)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp-checkpoint", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--num-structures", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", default="/tmp/diag_v2_result.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)
    processed_root = Path(args.processed_root)

    # Load coarse checkpoint
    print("Loading coarse checkpoint...", file=sys.stderr)
    coarse_ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = coarse_ck.get("config", {}).get("model", {})
    valid_keys = set(ModelConfig.__dataclass_fields__.keys())
    cfg_dict["use_fine_point_matcher"] = True  # FORCE
    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    print(f"use_equivariant_pose: {model_cfg.use_equivariant_hypothesis_pose}", file=sys.stderr)
    print(f"use_fine_point_matcher: {model_cfg.use_fine_point_matcher} (FORCED)", file=sys.stderr)
    print(f"equivariant_feature_dim: {model_cfg.equivariant_feature_dim}", file=sys.stderr)

    coarse = ProteinRegistrationModel(
        shot_dim=model_cfg.shot_dim, feature_dim=model_cfg.feature_dim,
        num_heads=model_cfg.num_heads, kernel_points=model_cfg.kernel_points,
        ops_topk=model_cfg.ops_topk, mutual_topk=model_cfg.mutual_topk,
        max_points_per_patch=model_cfg.max_points_per_patch,
        use_equivariant_pose=model_cfg.use_equivariant_hypothesis_pose,
        equivariant_feature_dim=model_cfg.equivariant_feature_dim,
        equivariant_max_hypotheses=model_cfg.equivariant_max_hypotheses,
        equivariant_acceptance_radius_angstrom=model_cfg.equivariant_acceptance_radius_angstrom,
    )
    coarse.load_state_dict(coarse_ck["model"], strict=False)
    coarse.to(device); coarse.eval()

    config = RegistrationConfig(model=model_cfg)
    refiner = build_hierarchical_refiner(config, coarse)
    refiner.to(device); refiner.eval()
    assert refiner.fine_point_matcher is not None, "FinePointMatcher was not created!"
    print(f"FinePointMatcher created: {type(refiner.fine_point_matcher).__name__}", file=sys.stderr)

    # Load FinePointMatcher weights
    print(f"Loading FinePointMatcher from {args.fp_checkpoint}...", file=sys.stderr)
    fp_ck = torch.load(args.fp_checkpoint, map_location="cpu", weights_only=False)
    if "fine_point_matcher" in fp_ck:
        state_dict = fp_ck["fine_point_matcher"]
    elif "model_state_dict" in fp_ck:
        state_dict = fp_ck["model_state_dict"]
    else:
        state_dict = fp_ck
    print(f"  processed_structures in checkpoint: {fp_ck.get('processed_structures', 'N/A')}", file=sys.stderr)

    # Load with key reporting
    model_state = refiner.fine_point_matcher.state_dict()
    missing = set(model_state.keys()) - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - set(model_state.keys())
    if missing: print(f"  MISSING keys ({len(missing)}): {sorted(missing)[:10]}...", file=sys.stderr)
    if unexpected: print(f"  UNEXPECTED keys ({len(unexpected)}): {sorted(unexpected)[:10]}...", file=sys.stderr)
    if not missing and not unexpected: print("  All keys matched (no missing, no unexpected)", file=sys.stderr)

    refiner.fine_point_matcher.load_state_dict(state_dict, strict=False)
    refiner.fine_point_matcher.eval()
    for p in refiner.fine_point_matcher.parameters():
        p.requires_grad_(False)

    # Also create a random-init FinePointMatcher for comparison
    random_fp = FinePointMatcher(
        shot_dim=model_cfg.shot_dim, encoder_dim=64,
        feature_dim=model_cfg.fine_point_feature_dim,
        num_heads=model_cfg.fine_point_attention_heads,
        equivariant_channels=model_cfg.equivariant_feature_dim,
        query_chunk_size=model_cfg.fine_point_attention_query_chunk,
        max_pair_elements=model_cfg.max_fine_point_pair_elements,
    ).to(device).eval()
    for p in random_fp.parameters(): p.requires_grad_(False)

    # Run on validation structures
    split = json.loads((processed_root / "split.json").read_text())
    structure_ids = list(split["val"])[:args.num_structures]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    all_channel_learned = []
    all_channel_random = []
    all_oracle = []
    all_status = []

    print(f"Running on {len(structure_ids)} structures...", file=sys.stderr)
    for sidx, structure_id in enumerate(structure_ids):
        print(f"  [{sidx+1}/{len(structure_ids)}] {structure_id}", file=sys.stderr)
        structure = _move_to_device(
            torch.load(processed_root / structure_id / "structure.pt",
                       map_location="cpu", weights_only=False), device)
        scale = float(structure["normalization"]["scale"])
        for chain_id in sorted(structure["chains"]):
            chain = structure["chains"][chain_id]
            augmentation = random_rigid_transform(
                chain["2.00"].device, chain["2.00"].dtype, generator)
            gt = inverse_transform(augmentation)

            try:
                output = refiner(structure, chain_id, source_transform=augmentation)
            except Exception as e:
                continue

            for diag in output.get("fine_diagnostics", []):
                for half_data in diag.get("halves", []):
                    pm_outputs = half_data.get("point_matcher_outputs", [])
                    pm_inputs = half_data.get("point_matcher_inputs", [])
                    pm_statuses = half_data.get("point_matcher_statuses", [])

                    for pmi, pm_out in enumerate(pm_outputs):
                        status = {
                            "structure": structure_id,
                            "chain": chain_id,
                            "candidate": pmi,
                            "matcher_status": pm_statuses[pmi] if pmi < len(pm_statuses) else "unknown",
                            "pm_out_is_none": pm_out is None,
                        }
                        if pm_out is None:
                            all_status.append(status)
                            continue

                        # Learned vectors from FinePointMatcher
                        lsrc = pm_out.source_equivariant
                        ltgt = pm_out.target_equivariant
                        ch_l_src = channel_collapse_stats(lsrc, "learned_source")
                        ch_l_tgt = channel_collapse_stats(ltgt, "learned_target")
                        if ch_l_src: all_channel_learned.append(ch_l_src)
                        if ch_l_tgt: all_channel_learned.append(ch_l_tgt)

                        # Random-init vectors
                        if pmi < len(pm_inputs):
                            mi = pm_inputs[pmi]
                            try:
                                rand_out = random_fp(
                                    mi["source_points"], mi["target_points"],
                                    mi["source_shot"], mi["target_shot"],
                                    mi["source_encoded"], mi["target_encoded"])
                                ch_r_src = channel_collapse_stats(rand_out.source_equivariant, "random_source")
                                ch_r_tgt = channel_collapse_stats(rand_out.target_equivariant, "random_target")
                                if ch_r_src: all_channel_random.append(ch_r_src)
                                if ch_r_tgt: all_channel_random.append(ch_r_tgt)
                            except Exception:
                                pass

                        # PCA vectors (from coarse model)
                        pca_src = None; pca_tgt = None
                        if pmi < len(pm_inputs):
                            mi = pm_inputs[pmi]
                            coarse_src_vecs = coarse._equivariant_vectors(
                                mi["source_points"], mi["source_encoded"])
                            coarse_tgt_vecs = coarse._equivariant_vectors(
                                mi["target_points"], mi["target_encoded"])
                            if coarse_src_vecs is not None and coarse_tgt_vecs is not None:
                                pca_src = coarse_src_vecs
                                pca_tgt = coarse_tgt_vecs

                        # Oracle pose table
                        if pmi < len(pm_inputs):
                            mi = pm_inputs[pmi]
                            oracle = oracle_pose_table(
                                mi["source_points"], mi["target_points"],
                                pca_src, pca_tgt,
                                lsrc, ltgt,
                                pm_out.source_descriptors, pm_out.target_descriptors,
                                gt, scale)
                            oracle["structure"] = structure_id
                            oracle["chain"] = chain_id
                            oracle["candidate"] = pmi
                            all_oracle.append(oracle)

                        status["equivariant_source"] = "learned_fine_point_matcher"
                        status["descriptor_shape"] = list(pm_out.source_descriptors.shape)
                        status["equivariant_shape"] = list(lsrc.shape)
                        all_status.append(status)

    # ── Aggregate ──
    result = {
        "status_summary": {},
        "channel_collapse_learned": {},
        "channel_collapse_random": {},
        "oracle_pose_table": {},
    }

    # Status summary
    n_total = len(all_status)
    n_none = sum(1 for s in all_status if s["pm_out_is_none"])
    n_ok = n_total - n_none
    result["status_summary"] = {
        "total_candidates": n_total,
        "pm_out_is_none": n_none,
        "pm_out_valid": n_ok,
        "sample_statuses": all_status[:5],
    }

    def agg_channel(stats_list, label):
        if not stats_list: return {"label": label, "count": 0}
        keys = ["mean_abs_cosine","median_abs_cosine","frac_cosine_gt_08",
                "mean_effective_rank","median_effective_rank","frac_rank_lt_2",
                "mean_sigma2_sigma1","median_sigma2_sigma1",
                "mean_sigma3_sigma1","median_sigma3_sigma1",
                "frac_sigma2_lt_01","frac_sigma3_lt_005"]
        agg = {k: float(np.mean([s[k] for s in stats_list if k in s])) for k in keys}
        agg["label"] = label
        agg["total_vector_sets"] = len(stats_list)
        agg["total_points"] = sum(s["num_points"] for s in stats_list)
        return agg

    result["channel_collapse_learned"] = agg_channel(all_channel_learned, "learned")
    result["channel_collapse_random"] = agg_channel(all_channel_random, "random_init")

    # Oracle pose aggregation
    if all_oracle:
        by_key = defaultdict(list)
        for o in all_oracle:
            for k, v in o.items():
                if isinstance(v, (int, float)) and k not in ("num_gt_corr", "num_pred_corr"):
                    by_key[k].append(v)
                elif k in ("num_gt_corr", "num_pred_corr"):
                    by_key[k].append(v)
        agg_o = {}
        for k, vals in by_key.items():
            if vals:
                agg_o[k] = {"count": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals))}
        result["oracle_pose_table"] = agg_o

    # Print
    for section, title in [
        ("status_summary", "STATUS SUMMARY"),
        ("channel_collapse_learned", "CHANNEL COLLAPSE - LEARNED"),
        ("channel_collapse_random", "CHANNEL COLLAPSE - RANDOM INIT"),
        ("oracle_pose_table", "ORACLE POSE TABLE"),
    ]:
        print(f"\n{'='*60}\n{title}\n{'='*60}")
        print(json.dumps(result[section], indent=2))

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
