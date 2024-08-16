import argparse
import enum
import os
from typing import Optional


# !! torch has to be imported before onnxruntime !!
import torch
import torch.nn as nn
import onnxruntime
import requests
import timm
import torchvision
from huggingface_hub import hf_hub_download, login
from timm.models.vision_transformer import VisionTransformer
from torch.hub import load_state_dict_from_url
from transformers import CLIPModel, SwinModel, ViTModel


class ModelType(enum.Enum):
    CTRANS = 1
    LUNIT = 2
    RESNET50 = 3
    UNI = 4
    SWIN224 = 5
    PHIKON = 6
    CHIEF = 7
    PLIP = 8
    GIGAPATH = 9
    CIGAR = 10
    NONE = None

    def __str__(self):
        return self.name


def parse_model_type(models_str):
    models = models_str.upper().split(",")
    try:
        return [ModelType[model] for model in models]
    except KeyError as e:
        raise argparse.ArgumentTypeError(f"Invalid model name: {e}")


def get_model(args, model: str) -> nn.Module:
    m_type = ModelType[model.upper()]
    if m_type == ModelType.CTRANS:
        model = get_ctrans()
    elif m_type == ModelType.LUNIT:
        model = get_lunit()
    elif m_type == ModelType.RESNET50:
        model = get_resnet50()
    elif m_type == ModelType.UNI:
        model = get_uni(args.hf_token)
    elif m_type == ModelType.SWIN224:
        model = get_swin_224()
    elif m_type == ModelType.PHIKON:
        model = get_phikon()
    elif m_type == ModelType.CHIEF:
        model = get_chief()
    elif m_type == ModelType.PLIP:
        model = get_plip()
    elif m_type == ModelType.GIGAPATH:
        model = get_gigapath(args.hf_token)
    elif m_type == ModelType.CIGAR:
        model = get_cigar()
    else:
        raise Exception("Invalid model type")
    return model


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


def get_uni(hf_token: Optional[str] = None):
    """
    =============== UNI ===============
    https://huggingface.co/MahmoodLab/UNI
    Warning: this model requires an access request to the model owner.
    =============== UNI ===============
    """
    if hf_token:
        login(token=hf_token)

    model_dir = os.path.expanduser("~/.models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = hf_hub_download(
        "MahmoodLab/UNI",
        filename="pytorch_model.bin",
        cache_dir=model_dir,
        force_download=False,
    )
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
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


def get_ctrans():
    # =============== CTransPath ===============
    # requires the ctranspath.onnx file to be located
    # at CTRANS_PTH environment variable
    # =============== CTransPath ===============
    model_pth = os.environ.get("CTRANS_PTH", "ctranspath.onnx")

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            session_options = onnxruntime.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            # session_options.log_severity_level = 0   # 0 for verbose / fix slurm crap
            session_options.enable_profiling = False

            self.ort_session = onnxruntime.InferenceSession(
                model_pth, session_options, providers=["CUDAExecutionProvider"]
            )

        def forward(self, x):
            """
            CTransPath was exported to onnx based on the old timm library.
            That library has hard-coded shapes within its library.
            Thus it was not possible to export the model with dynamic shapes.
            Therfore, the model expects a batch size of 1.
            """

            dev = x.device
            input_name = self.ort_session.get_inputs()[0].name

            all_outputs = []
            for item in x:
                ort_inputs = {input_name: item.unsqueeze(0).cpu().numpy()}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output = torch.tensor(ort_outs[0]).to(dev)
                all_outputs.append(output)
            return torch.stack(all_outputs).squeeze().to(dev)

    model = Model()
    model = model.eval()
    return model


def get_chief():
    # =============== Chief ===============
    # requires the chief.onnx file to be located
    # at CHIEF_PTH environment variable
    # =============== Chief ===============
    model_pth = os.environ.get("CHIEF_PTH", "chief.onnx")

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            session_options = onnxruntime.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            # session_options.log_severity_level = 0   # 0 for verbose / fix slurm crap
            session_options.enable_profiling = False

            self.ort_session = onnxruntime.InferenceSession(
                model_pth, session_options, providers=["CUDAExecutionProvider"]
            )

        def forward(self, x):
            """
            Chief is based on CTransPath, and was exported to onnx based on the old timm library.
            That library has hard-coded shapes within its library.
            Thus it was not possible to export the model with dynamic shapes.
            Therfore, the model expects a batch size of 1.
            """
            dev = x.device
            input_name = self.ort_session.get_inputs()[0].name

            all_outputs = []
            for item in x:
                ort_inputs = {input_name: item.unsqueeze(0).cpu().numpy()}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output = torch.tensor(ort_outs[0]).to(dev)
                all_outputs.append(output)
            return torch.stack(all_outputs).squeeze().to(dev)

    model = Model()
    model = model.eval()
    return model


def get_gigapath(hf_token: Optional[str] = None):
    """
    =============== Gigapath ===============
    Requires a huggingface token to download the model

    See here:
        https://github.com/prov-gigapath/prov-gigapath
    =============== Gigapath ===============
    """
    if hf_token:
        login(token=hf_token)
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return model


def get_plip():
    """
    =============== PLIP ===============
    Pathology Language and Image Pre-Training (PLIP)
    We only utilize the vision backbone here.

    See here:
        https://github.com/PathologyFoundation/plip
    =============== PLIP ===============
    """

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = CLIPModel.from_pretrained("vinid/plip").vision_model

        def forward(self, x):
            return self.model(x).pooler_output

    return Model()


def get_cigar(ckpt_path: str = "~/.models"):
    """
    =============== CIGAR ===============
    See here:
        https://github.com/ozanciga/self-supervised-histopathology
    =============== CIGAR ===============
    """
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_url = "https://github.com/ozanciga/self-supervised-histopathology/releases/download/nativetenpercent/pytorchnative_tenpercent_resnet18.ckpt"
    ckpt_path = os.path.join(ckpt_path, "pytorchnative_tenpercent_resnet18.ckpt")
    download_checkpoint(ckpt_url, ckpt_path)
    model = torchvision.models.__dict__["resnet18"](weights=None)

    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print("No weight could be loaded..")
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(
            key
        )
    model = load_model_weights(model, state_dict)
    model.fc = nn.Identity()
    return model


def download_checkpoint(ckpt_url, ckpt_path):
    if not os.path.exists(ckpt_path):
        print("Downloading checkpoint...")
        response = requests.get(ckpt_url, allow_redirects=True)
        with open(ckpt_path, "wb") as f:
            f.write(response.content)
    else:
        print("Checkpoint exists.")
