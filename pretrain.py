from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from model import PretrainAudioMAEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="./conf", config_name="audio_mae_pretrain_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    model = PretrainAudioMAEModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
