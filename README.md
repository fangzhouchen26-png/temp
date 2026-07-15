# FinePointMatcher: equivariant channel collapse fix

## Key files
- model.py: EquivariantVectorHead (L124-196), equivariant_pose_hypotheses, estimate_rigid_transform
- fine_point_matching.py: FinePointMatcher, fine_equivariant_alignment_loss
- hierarchical.py: inference pipeline, _run_fine_stage with equivariant wiring
- train_fine_point_matching.py: training loop, fine_point_training_loss, checkpoint save
- training.py: random_rigid_transform, inverse_transform, compute_chain_rmse_angstrom
- config.py: ModelConfig with equivariant settings
- diag_v2.py: diagnostic script for channel collapse + oracle poses

## Root cause
EquivariantVectorHead shares same k=16 neighbors across all 8 channels.
Softmax per channel + same neighbor pool = all channels collapse to same direction.
Random init mean_abs_cosine=0.970; trained mean_abs_cosine=0.993, effective_rank=1.006.

## Fix targets
1. EquivariantVectorHead: force channel diversity (different neighbor subsets per channel)
2. fine_equivariant_alignment_loss: add rank/anti-collapse regularization
3. equivariant_pose_hypotheses: single-point SVD → multi-point joint hypothesis
