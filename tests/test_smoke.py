"""Config composition smoke tests."""
from hydra import compose, initialize


def test_config_composes():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
    assert cfg.model.num_labels == 2
    assert cfg.data.batch_size > 0
    assert cfg.train.max_train_steps > 0 
    assert cfg.seed == 42


def test_config_override():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["train.lr=1e-4", "train.max_train_steps=100"])
    assert cfg.train.lr == 1e-4
    assert cfg.train.max_train_steps == 100