"""Run a small real GPU forward/loss/backward/optimizer smoke test."""

import argparse
import math

import torch

from config import make_cfg
from dataset import train_valid_data_loader
from loss import OverallLoss
from model import create_model
from pareconv.utils.torch import to_cuda


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    return parser


def main():
    args = make_parser().parse_args()
    cfg = make_cfg()
    train_loader, _, _ = train_valid_data_loader(cfg, distributed=False)
    model = create_model(cfg).cuda().train()
    loss_fn = OverallLoss(cfg).cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            data_dict = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data_dict = next(iterator)
        data_dict = to_cuda(data_dict)
        optimizer.zero_grad(set_to_none=True)
        output_dict = model(data_dict)

        valid_gt = int(
            (
                output_dict["gt_node_corr_overlaps"]
                > cfg.coarse_matching.overlap_threshold
            ).sum().item()
        )
        if valid_gt == 0:
            raise RuntimeError("No valid GT coarse patch in smoke-test batch")

        losses = loss_fn(output_dict, data_dict)
        loss = losses["loss"]
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"Non-finite loss at step {step}: {losses}")
        loss.backward()

        finite_gradients = True
        gradient_parameters = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient_parameters += 1
            if not torch.isfinite(parameter.grad).all():
                finite_gradients = False
                break
        if gradient_parameters == 0:
            raise RuntimeError("No trainable parameter received a gradient")
        if not finite_gradients:
            raise RuntimeError(f"Non-finite gradients at step {step}")

        optimizer.step()
        print(
            f"step={step + 1} loss={float(loss.detach().cpu()):.6f} "
            f"valid_gt_patches={valid_gt} grad_params={gradient_parameters}"
        )

    print("PASS: forward, loss, backward and optimizer step completed.")


if __name__ == "__main__":
    main()
