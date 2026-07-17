import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from pareconv.utils.common import ensure_dir


_C = edict()
_C.seed = 7351

_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "wandb_events")
_C.feature_dir = osp.join(_C.output_dir, "features")
_C.registration_dir = osp.join(_C.output_dir, "registration")
for path in (
    _C.output_dir,
    _C.snapshot_dir,
    _C.log_dir,
    _C.event_dir,
    _C.feature_dir,
    _C.registration_dir,
):
    ensure_dir(path)

# The installer places PARENet at <temp>/external/PARENet, so the repository
# dataset defaults to <temp>/dataset. Override with PROTEINFIT_DATASET_ROOT.
_C.data = edict()
_C.data.dataset_root = os.environ.get(
    "PROTEINFIT_DATASET_ROOT",
    osp.abspath(osp.join(_C.root_dir, "..", "..", "dataset")),
)
_C.data.split_file = os.environ.get("PROTEINFIT_SPLIT_FILE") or None
_C.data.point_suffix = os.environ.get("PROTEINFIT_POINT_SUFFIX", "_2.00.txt")
_C.data.split_seed = 7351
_C.data.train_ratio = 0.70
_C.data.val_ratio = 0.15
_C.data.min_source_points = 128
_C.data.min_target_points = 256

# Input-level spherical target cropping.  The crop diameter is exactly
# diameter_scale times the robust source-chain diameter.  With the default
# 1.25 setting: D_crop = 1.25 * D_chain and r_crop = 1.25 * r_chain.
_C.crop = edict()
_C.crop.enabled = True
_C.crop.diameter_scale = 1.25
_C.crop.radius_quantile = 0.99
# Sliding-center spacing as a fraction of the source-chain diameter.
_C.crop.stride_ratio = 0.25
_C.crop.min_stride = 2.0  # Angstrom
_C.crop.min_points = 128
_C.crop.max_candidates = None  # compare every valid sliding window
# Training/validation use the known chain location to provide positive crops.
# Testing searches the complete target map with target-supported sliding centers.
_C.crop.train_mode = "oracle"
_C.crop.val_mode = "oracle"
_C.crop.test_mode = "sliding"
_C.crop.train_center_jitter_ratio = 0.10
_C.crop.min_chain_coverage = 0.85
# Reject oracle training/validation pairs whose source-to-crop point overlap is too low.
_C.crop.min_oracle_overlap = 0.50

_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 4
_C.train.point_limit = 30000
_C.train.augmentation_noise = 0.25  # Angstrom
_C.train.augmentation_translation = 30.0  # Angstrom per axis
_C.train.source_keep_ratio = 0.90
_C.train.matching_radius = 4.0  # Angstrom

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 2
_C.test.point_limit = 30000
_C.test.augmentation_noise = 0.0
_C.test.augmentation_translation = 30.0

_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 4.0
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 5.0
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 5.0
_C.eval.feat_rre_threshold = 20.0

_C.ransac = edict()
_C.ransac.distance_threshold = 4.0
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 80
_C.optim.grad_acc_steps = 1

_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.num_neighbors = [35] * _C.backbone.num_stages
# precompute_subsample multiplies before generating stage 1; 1.0 therefore
# produces approximately 2, 4 and 8 Angstrom voxel grids at stages 1..3.
_C.backbone.init_voxel_size = 1.0
_C.backbone.subsample_ratio = 2
_C.backbone.kernel_size = 4
_C.backbone.share_nonlinearity = False
_C.backbone.conv_way = "edge_conv"
_C.backbone.use_xyz = True
_C.backbone.init_dim = 96
_C.backbone.output_dim = 256

_C.model = edict()
_C.model.ground_truth_matching_radius = 4.0
_C.model.num_points_in_patch = 32

_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 64
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 128
_C.coarse_matching.dual_normalization = True

_C.geotransformer = edict()
_C.geotransformer.input_dim = 768
_C.geotransformer.hidden_dim = 192
_C.geotransformer.output_dim = 192
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 8.0
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 4.0
_C.fine_matching.confidence_threshold = 0.005
_C.fine_matching.num_hypotheses = 512
_C.fine_matching.num_refinement_steps = 5
_C.fine_matching.use_encoder_re_feats = True

_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

_C.fine_loss = edict()
_C.fine_loss.positive_radius = 3.0
_C.fine_loss.negative_radius = 8.0
_C.fine_loss.positive_margin = 0.1
_C.fine_loss.negative_margin = 1.4

_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_ri_loss = 1.0
_C.loss.weight_fine_re_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.link_output and not osp.exists("output"):
        os.symlink(_C.output_dir, "output")


if __name__ == "__main__":
    main()
