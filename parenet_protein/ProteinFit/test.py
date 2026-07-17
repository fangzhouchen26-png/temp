import argparse
import os.path as osp
import time

import numpy as np

from pareconv.engine import SingleTester
from pareconv.utils.common import ensure_dir, get_log_string
from pareconv.utils.torch import release_cuda

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="test", choices=["train", "val", "test"])
    return parser


def _scalar(value):
    array = np.asarray(release_cuda(value))
    return array.item()


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())
        start = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        self.logger.info(f"Data loader created in {time.time() - start:.3f}s")
        self.logger.info(f"Neighbor limits: {neighbor_limits}")
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        return self.model(data_dict)

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        candidate_id = int(_scalar(data_dict["candidate_id"]))
        message = (
            f"{data_dict['case_id']}/chain{data_dict['chain_id']}"
            f"/candidate{candidate_id:04d}"
        )
        return message + ", " + get_log_string(result_dict=result_dict)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        case_id = data_dict["case_id"]
        chain_id = data_dict["chain_id"]
        candidate_id = int(_scalar(data_dict["candidate_id"]))
        ensure_dir(osp.join(self.output_dir, case_id))
        path = osp.join(
            self.output_dir,
            case_id,
            f"chain{chain_id}_candidate{candidate_id:04d}.npz",
        )
        np.savez_compressed(
            path,
            ref_points=release_cuda(output_dict["ref_points"]),
            src_points=release_cuda(output_dict["src_points"]),
            ref_corr_points=release_cuda(output_dict["ref_corr_points"]),
            src_corr_points=release_cuda(output_dict["src_corr_points"]),
            corr_scores=release_cuda(output_dict["corr_scores"]),
            estimated_transform=release_cuda(output_dict["estimated_transform"]),
            transform=release_cuda(data_dict["transform"]),
            overlap=release_cuda(data_dict["overlap"]),
            candidate_id=candidate_id,
            window_center=release_cuda(data_dict["window_center"]),
            crop_radius=release_cuda(data_dict["crop_radius"]),
            crop_diameter=release_cuda(data_dict["crop_diameter"]),
            chain_radius=release_cuda(data_dict["chain_radius"]),
            chain_diameter=release_cuda(data_dict["chain_diameter"]),
        )


def main():
    Tester(make_cfg()).run()


if __name__ == "__main__":
    main()
