from hydra.utils import instantiate
from omegaconf import DictConfig

from models.expert import MILExpert
from models.moe import MILMoE, MoE
from models.router import MILRouter


def build_simple_mil_moe(cfg: DictConfig) -> MILMoE:
    experts = []
    device = cfg.train.device
    for expert_cfg in cfg.moe.expert_heads:
        expert_model = instantiate(expert_cfg)
        expert = MILExpert(
            model=expert_model, model_dim=expert_cfg.in_features, drop_p=0.1
        ).to(device)
        experts.append(expert)
    router_config = cfg.moe.router
    router_model = instantiate(router_config)
    router = MILRouter(
        model=router_model,
        drop_p=cfg.moe.drop_p,
        temperature=cfg.moe.router_temperature,
    ).to(device)
    return MILMoE(
        experts=experts,
        router=router,
        strategy=cfg.moe.strategy,
        temperature=cfg.moe.moe_temperature,
    ).to(device)


def build_simple_moe(cfg: DictConfig) -> MoE:
    experts = []
    device = cfg.train.device
    for expert_cfg in cfg.moe.expert_heads:
        expert = instantiate(expert_cfg).to(device)
        experts.append(expert)
    router = instantiate(cfg.moe.router).to(device)
    return MoE(
        experts=experts,
        router=router,
        strategy=cfg.moe.strategy,
        temperature=cfg.moe.moe_temperature,
    ).to(device)
