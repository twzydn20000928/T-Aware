"""One model definition, retaining historical state_dict parameter names."""
import inspect

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class TemperatureAwareDetector(nn.Module):
    def __init__(self, config, lambda_reg=1.0, backbone=None):
        super().__init__()
        self.config = config
        self.roberta = backbone if backbone is not None else AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.lambda_reg = lambda_reg

    @classmethod
    def pretrained(cls, base_model, lambda_reg=1.0):
        backbone = AutoModel.from_pretrained(base_model)
        return cls(backbone.config, lambda_reg, backbone)

    def forward(self, input_ids, attention_mask=None, labels=None, t_labels=None):
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.classifier(hidden)
        predicted_temperature = self.regressor(hidden).squeeze(-1)
        loss_cls = logits.sum() * 0.0
        loss_reg = predicted_temperature.sum() * 0.0
        if labels is not None:
            loss_cls = nn.functional.cross_entropy(logits, labels)
            if t_labels is not None and self.lambda_reg > 0:
                mask = (labels == 1) & (t_labels >= 0)
                if mask.any():
                    loss_reg = nn.functional.mse_loss(predicted_temperature[mask], t_labels[mask])
        return {"loss": loss_cls + self.lambda_reg * loss_reg, "logits": logits,
                "temperature": predicted_temperature, "classification_loss": loss_cls,
                "regression_loss": loss_reg}


def load_checkpoint(path, config_path, lambda_reg=1.0):
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True) if not str(config_path) == "roberta-base" else AutoConfig.from_pretrained("roberta-base")
    model = TemperatureAwareDetector(config, lambda_reg)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    # On old PyTorch, only load trusted, locally owned historical checkpoints.
    state = torch.load(path, **kwargs)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model
