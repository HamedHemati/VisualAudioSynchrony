import sys
from vasync.utils.common import load_config
from vasync.models.pretext_model import ClassifierPretextModel
from vasync.trainers import ClassficationTrainer


def main(config_path):
    config = load_config(config_path)
    if config["pretext_task"] == "img_cls":
        pretext_base_model = config["pretext_base_model"]
        num_cls = config["num_cls"]
        model = ClassifierPretextModel(pretext_base_model=pretext_base_model, 
                                       num_cls=num_cls)
        trainer = ClassficationTrainer(config, model)
        trainer.train()


if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)