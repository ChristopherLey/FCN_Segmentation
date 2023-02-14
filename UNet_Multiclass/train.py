import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import torch
import yaml
from dataset_cityscapes import CityScapes
from UNet_Multiclass.model import UNetMulticlassExperiment
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from utilities import convert_dict_2_html


def main():
    parser = argparse.ArgumentParser(description="Generic runner for UNet-Multiclass")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="tcn_skab_lossy_rdo_config.yaml",
    )
    parser.add_argument("--latent_dim", "-d", default=None)
    args = parser.parse_args()
    with open(args.filename, "r") as file:
        try:
            configuration = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        print("Debugging Mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        configuration["data_params"]["num_workers"] = 0

    version_path = (
        f"{configuration['model_params']['model']}"
        f"-{configuration['data_params']['data_type']}-{datetime.now().strftime('%d-%m_%H:%M:%S')}"
    )
    configuration["logging_params"]["version_path"] = version_path

    rng_state = torch.get_rng_state()
    train_dataset = CityScapes(
        root=Path(configuration["data_params"]["data_path"]), type="train"
    )
    configuration["model_params"]["num_classes"] = train_dataset.num_classes
    val_dataset = CityScapes(
        root=Path(configuration["data_params"]["data_path"]), type="val"
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=configuration["data_params"]["num_workers"],
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=configuration["data_params"]["num_workers"],
        drop_last=False,
    )

    model = UNetMulticlassExperiment(
        in_channels=configuration["model_params"]["in_channels"],
        out_channels=configuration["model_params"]["num_classes"],
        feature_sizes=configuration["model_params"]["feature_sizes"],
    )

    loss_callback = ModelCheckpoint(
        monitor="mIoU",
        save_top_k=4,
        mode="max",
        filename="loss-{epoch:02d}-{f1_score:.6f}",
    )

    callbacks = [
        loss_callback,
    ]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=configuration["logging_params"]["save_dir"],
        version=version_path,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=configuration["trainer_params"]["gpus"],
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=configuration["trainer_params"]["max_epochs"],
        default_root_dir=configuration["logging_params"]["save_dir"],
        log_every_n_steps=1,
    )
    pprint(configuration)
    trainer.logger.experiment.add_text(
        "configuration", convert_dict_2_html(configuration), global_step=0
    )
    log_path = Path(tb_logger.log_dir)
    torch.save(rng_state, log_path / "rng_state.pt")
    with open(log_path / "config.yaml", "w") as yml_file:
        yaml.dump(configuration, yml_file, default_flow_style=False)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
