from transformers import AutoModelForImageClassification
from omegaconf import DictConfig


def build_model(cfg: DictConfig):
    """Load a pretrained ViT and adapt its head for num_labels classes."""
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,      # fixed: under the model group
        ignore_mismatched_sizes=True,
    )
# Freezing (train only the head) is faster, needs less data -- in case requires_grad = False;
# otherwise all the weights will be updated
    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model
