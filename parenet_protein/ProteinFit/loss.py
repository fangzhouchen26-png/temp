"""PARE-Net losses with empty-correspondence safeguards for protein pairs."""

import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss
from pareconv.modules.ops.pairwise_distance import pairwise_distance
from pareconv.modules.ops.transformation import apply_transform
from pareconv.modules.registration.metrics import isotropic_transform_error


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    return values.mean()


def _safe_log_mean(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0
    return probabilities.clamp(min=1e-8, max=1.0 - 1e-8).log().mean()


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict["ref_feats_c"]
        src_feats = output_dict["src_feats_c"]
        indices = output_dict["gt_node_corr_indices"]
        corr_overlaps = output_dict["gt_node_corr_overlaps"]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))
        overlaps = torch.zeros_like(feat_dists)
        if indices.numel() > 0:
            overlaps[indices[:, 0], indices[:, 1]] = corr_overlaps
        pos_masks = overlaps > self.positive_overlap
        neg_masks = overlaps == 0
        if not torch.any(pos_masks) or not torch.any(neg_masks):
            return feat_dists.sum() * 0.0
        pos_scales = torch.sqrt(overlaps * pos_masks.float())
        return self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positive_radius = cfg.fine_loss.positive_radius
        self.negative_radius = cfg.fine_loss.negative_radius
        self.positive_margin = cfg.fine_loss.positive_margin
        self.negative_margin = cfg.fine_loss.negative_margin

    def forward(self, output_dict, data_dict):
        ref_points = output_dict["ref_node_corr_knn_points"]
        src_points = output_dict["src_node_corr_knn_points"]
        ref_masks = output_dict["ref_node_corr_knn_masks"]
        src_masks = output_dict["src_node_corr_knn_masks"]
        ref_scores = output_dict["ref_node_corr_knn_scores"]
        src_scores = output_dict["src_node_corr_knn_scores"]
        matching_scores = output_dict["matching_scores"]
        transform = data_dict["transform"]

        src_points = apply_transform(src_points, transform)
        dists = pairwise_distance(ref_points, src_points)
        valid_map = torch.logical_and(ref_masks.unsqueeze(2), src_masks.unsqueeze(1))
        positive_map = torch.logical_and(dists < self.positive_radius**2, valid_map)
        negative_map = torch.logical_and(dists > self.negative_radius**2, valid_map)

        slack_rows = torch.logical_and(positive_map.sum(2) == 0, ref_masks)
        slack_cols = torch.logical_and(positive_map.sum(1) == 0, src_masks)
        fine_ri_loss = -(
            _safe_log_mean(matching_scores[positive_map])
            + 0.5 * _safe_log_mean(1.0 - ref_scores[slack_rows])
            + 0.5 * _safe_log_mean(1.0 - src_scores[slack_cols])
        )
        fine_re_loss = self.fine_re_loss(
            output_dict, positive_map, negative_map, transform
        )
        return fine_ri_loss, fine_re_loss

    def fine_re_loss(self, output_dict, positive_map, negative_map, transform):
        ref_feats = output_dict["re_ref_node_corr_knn_feats"]
        src_feats = output_dict["re_src_node_corr_knn_feats"]

        batch, ref_idx, src_idx = torch.nonzero(positive_map, as_tuple=True)
        if batch.numel() > 0:
            ref_positive = ref_feats[batch, ref_idx]
            src_positive = src_feats[batch, src_idx]
            src_positive = torch.einsum(
                "bck,lk->bcl", src_positive, transform[:3, :3]
            )
            positive_loss = _safe_mean(
                torch.relu(
                    torch.linalg.norm(src_positive - ref_positive, dim=-1)
                    - self.positive_margin
                )
            )
        else:
            positive_loss = ref_feats.sum() * 0.0

        batch, ref_idx, src_idx = torch.nonzero(negative_map, as_tuple=True)
        if batch.numel() > 0:
            ref_negative = ref_feats[batch, ref_idx]
            src_negative = src_feats[batch, src_idx]
            src_negative = torch.einsum(
                "bck,lk->bcl", src_negative, transform[:3, :3]
            )
            negative_loss = _safe_mean(
                torch.relu(
                    self.negative_margin
                    - torch.linalg.norm(src_negative - ref_negative, dim=-1)
                )
            )
        else:
            negative_loss = ref_feats.sum() * 0.0
        return positive_loss + negative_loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_ri_loss = cfg.loss.weight_fine_ri_loss
        self.weight_fine_re_loss = cfg.loss.weight_fine_re_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_ri_loss, fine_re_loss = self.fine_loss(output_dict, data_dict)
        loss = (
            self.weight_coarse_loss * coarse_loss
            + self.weight_fine_ri_loss * fine_ri_loss
            + self.weight_fine_re_loss * fine_re_loss
        )
        return {
            "loss": loss,
            "c_loss": coarse_loss,
            "f_ri_loss": fine_ri_loss,
            "f_re_loss": fine_re_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_count = output_dict["ref_points_c"].shape[0]
        src_count = output_dict["src_points_c"].shape[0]
        overlaps = output_dict["gt_node_corr_overlaps"]
        indices = output_dict["gt_node_corr_indices"]
        indices = indices[overlaps > self.acceptance_overlap]
        corr_map = torch.zeros(
            ref_count,
            src_count,
            device=output_dict["ref_points_c"].device,
        )
        if indices.numel() > 0:
            corr_map[indices[:, 0], indices[:, 1]] = 1.0
        ref_pred = output_dict["ref_node_corr_indices"]
        src_pred = output_dict["src_node_corr_indices"]
        if ref_pred.numel() == 0:
            return corr_map.sum() * 0.0
        return corr_map[ref_pred, src_pred].mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        ref_points = output_dict["ref_corr_points"]
        src_points = output_dict["src_corr_points"]
        if src_points.shape[0] == 0:
            return data_dict["transform"].sum() * 0.0
        src_points = apply_transform(src_points, data_dict["transform"])
        distances = torch.linalg.norm(ref_points - src_points, dim=1)
        return (distances < self.acceptance_radius).float().mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict["transform"]
        estimated = output_dict["estimated_transform"]
        src_points = output_dict["src_points"]
        rre, rte = isotropic_transform_error(transform, estimated)
        residual = torch.matmul(torch.inverse(transform), estimated)
        realigned = apply_transform(src_points, residual)
        rmse = torch.linalg.norm(realigned - src_points, dim=1).mean()
        recall = (rmse < self.acceptance_rmse).float()
        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        coarse_precision = self.evaluate_coarse(output_dict)
        fine_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)
        return {
            "PIR": coarse_precision,
            "IR": fine_precision,
            "RRE": rre,
            "RTE": rte,
            "RMSE": rmse,
            "RR": recall,
        }
