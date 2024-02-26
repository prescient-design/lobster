import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from lobster.affinity_experiments.exp import ExpLLM
from lobster.affinity_experiments.utils.tools import flatten_config


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep="/")

    logger.debug(str(wandb.config))
    wandb.config["cwd"] = os.getcwd()
    logger.info("Working directory : {}".format(os.getcwd()))
    wandb.login(host="https://genentech.wandb.io/")
    wandb.init(**cfg.wandb, config=wandb.config, settings=wandb.Settings(start_method="thread"))

    cfg.use_gpu = True if torch.cuda.is_available() and cfg.use_gpu else False
    if cfg.use_gpu and cfg.use_multi_gpu:
        cfg.devices = cfg.devices.replace(" ", "")
        device_ids = cfg.devices.split(",")
        device_ids = [int(id_) for id_ in device_ids]
        cfg.gpu = device_ids

    logger.info(f"model: {cfg.model}")

    # Exp= ExpPLM

    print("cfg.itr", cfg.itr)
    # Loop for running the experiments
    for ii in range(cfg.itr):
        setting_part1 = f"{cfg.oracle}_{cfg.model}"
        if cfg.use_moe:
            setting_part2 = "_MoE"
        else:
            setting_part2 = ""

        if cfg.use_mlp:
            setting_part3 = "_MLP"
        else:
            setting_part3 = ""

        if cfg.init_step:
            setting_part4 = "_LP"
        elif cfg.path_for_init != "":
            setting_part4 = "_LPFT"
        else:
            setting_part4 = "_FT"

        setting_part5 = f"_bs{cfg.batch_size}" f"_{cfg.preprocess}" f"_lr{cfg.lr}" f"_{cfg.task}"

        if cfg.append != "":
            setting_part6 = "_" + cfg.append + f"_{ii}"
        else:
            setting_part6 = f"_{ii}"

        if cfg.training_type != "regular":
            setting_part7 = "_" + cfg.training_type
        else:
            setting_part7 = ""

        setting = (
            setting_part1
            + setting_part2
            + setting_part3
            + setting_part4
            + setting_part5
            + setting_part6
            + setting_part7
        )

        wandb.config["target_value"] = cfg.target_value
        exp = ExpLLM(wandb.config)  # set experiments
        OmegaConf.save(config=cfg, f=setting + ".yaml")

        if cfg.train:
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
        print(">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<".format(setting))

        options = ["_last", "_best_valid"]

        # for option in options:
        option = "_best_valid"
        pred_df = exp.predict(setting, load=True, option=option)

        return pred_df


if __name__ == "__main__":
    import sys

    sys.argv.append("hydra.job.chdir=True")
    main()
