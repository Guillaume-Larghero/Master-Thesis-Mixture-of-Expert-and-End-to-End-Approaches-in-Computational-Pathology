import os
import tempfile
from typing import Optional

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, login
from timm.models.vision_transformer import VisionTransformer
from torch.hub import load_state_dict_from_url
from transformers import SwinModel, ViTModel


def get_resnet50():
    model = timm.create_model("resnet50", pretrained=True)
    model.fc = nn.Identity()
    return model


def get_phikon():
    # =============== Owkin Phikon ===============
    # https://github.com/owkin/HistoSSLscaling
    # =============== Owkin Phikon ===============
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = ViTModel.from_pretrained(
                "owkin/phikon", add_pooling_layer=True
            )

        def forward(self, x):
            x = {"pixel_values": x}
            features = self.model(**x).last_hidden_state[:, 0, :]
            return features

    return Model()


def get_lunit():
    # =============== Lunit Dino ===============
    # https://github.com/lunit-io/benchmark-ssl-pathology
    # =============== Lunit Dino ===============
    url_prefix = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    key = "DINO_p16"
    pretrained_url = f"{url_prefix}/{model_zoo_registry.get(key)}"
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0
    )
    state_dict = load_state_dict_from_url(pretrained_url, progress=False)
    model.load_state_dict(state_dict)
    return model


def get_uni(hf_token: Optional[str]):
    # =============== UNI ===============
    # https://huggingface.co/MahmoodLab/UNI
    # Warning: this model requires an access request to the model owner.
    # =============== UNI ===============
    if hf_token:
        login(token=hf_token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_hub_download(
            "MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=tmp_dir,
            force_download=True,
        )
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(os.path.join(tmp_dir, "pytorch_model.bin"), map_location="cpu"),
            strict=True,
        )
        return model


def get_swin_224():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = SwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224"
            )

        def forward(self, x):
            return self.model(x).pooler_output

    model = Model()
    return model


# def get_ctrans():
#     TODO: fix ctranspath
#     current_dir = os.path.dirname(__file__)
#     ctranspath_pth = os.path.join(current_dir, "..", "ctranspath.pth")
#     model = ctranspath()
#     model.head = nn.Identity()
#     td = torch.load(ctranspath_pth)
#     model.load_state_dict(td["model"], strict=True)
#     model = model.eval()
#     return model
