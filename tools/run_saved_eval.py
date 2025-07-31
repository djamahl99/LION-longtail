import datetime
import pickle
from pathlib import Path
import json

from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


from pcdet.datasets.argo2.argo2_dataset import Argo2Dataset

def main():
    result_dir = Path("/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/")
    cfg_file = Path("cfgs/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2.yaml")

    assert result_dir.exists(), f"{result_dir=} does not exist!"
    assert cfg_file.exists(), f"{cfg_file=} does not exist!"

    cfg_from_yaml_file(cfg_file, cfg)

    final_output_dir = result_dir / 'final_result' / 'data' # so many subdirs aha
    final_output_dir.mkdir(parents=True, exist_ok=True)


    pklfile_prefix = result_dir / "processed_results.feather"


    log_file = result_dir / ('log_run_saved_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    root_path = Path("/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/data/argo2")

    dataset = Argo2Dataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
        root_path=root_path
    )

    with open(result_dir / 'result.pkl', 'rb') as f:
        det_annos = pickle.load(f)

    class_names = dataset.class_names


    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
        pklfile_prefix=str(pklfile_prefix)
    )

    print(result_str)

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(result_dict, f)

    logger.info(result_str)
    logger.info('****************Evaluation done.*****************')


if __name__ == "__main__":
    main()