# Key files for equivariant channel collapse diagnosis

## Files
- model.py: EquivariantVectorHead (line 124-196), equivariant_pose_hypotheses, estimate_rigid_transform
- fine_point_matching.py: FinePointMatcher, fine_equivariant_alignment_loss (line 171-175)
- diag_v2_result.json: diagnostic output showing channel collapse (mean_abs_cosine=0.993, effective_rank=1.006)

## Root cause
EquivariantVectorHead shares the same k=16 neighbors across all 8 channels.
Softmax per channel + same neighbor pool = all channels collapse to same direction.
Random init already has mean_abs_cosine=0.970; training worsens it to 0.993.
