import argparse
import os
import sys

import torch
import yaml
from mlagents.plugins.trainer_type import register_trainer_plugins
from mlagents.trainers.settings import RunOptions
from mlagents_envs.base_env import (
    ActionSpec,
    BehaviorSpec,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
)
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.model_serialization import (
    ModelSerializer,
    exporting_to_onnx,
)
from mlagents.trainers.settings import SerializationSettings
from mlagents.trainers.torch_entities.networks import SharedActorCritic, SimpleActor


def infer_obs_size(policy_state, checkpoint_state):
    for name, tensor in policy_state.items():
        if name.endswith("seq_layers.0.weight") and hasattr(tensor, "shape"):
            if len(tensor.shape) == 2:
                return int(tensor.shape[1])
    for key, tensor in checkpoint_state.get("Optimizer:critic", {}).items():
        if "running_mean" in key and hasattr(tensor, "shape"):
            return int(tensor.shape[0])
    raise RuntimeError("Failed to infer observation size from checkpoint.")


def infer_continuous_actions(policy_state):
    if "continuous_act_size_vector" in policy_state:
        return int(float(policy_state["continuous_act_size_vector"][0]))
    raise RuntimeError("Failed to infer continuous action size from checkpoint.")


def build_policy(config_path, behavior_name, obs_size, action_size):
    register_trainer_plugins()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    options = RunOptions.from_dict(config)
    trainer_settings = options.behaviors[behavior_name]

    obs_spec = ObservationSpec(
        shape=(obs_size,),
        dimension_property=(DimensionProperty.NONE,),
        observation_type=ObservationType.DEFAULT,
        name="vector_observation",
    )
    action_spec = ActionSpec.create_continuous(action_size)
    behavior_spec = BehaviorSpec([obs_spec], action_spec)

    shared_critic = getattr(trainer_settings.hyperparameters, "shared_critic", False)
    actor_cls = SharedActorCritic if shared_critic else SimpleActor
    actor_kwargs = {
        "conditional_sigma": False,
        "tanh_squash": False,
    }

    policy = TorchPolicy(
        seed=0,
        behavior_spec=behavior_spec,
        network_settings=trainer_settings.network_settings,
        actor_cls=actor_cls,
        actor_kwargs=actor_kwargs,
    )
    return policy


def main():
    parser = argparse.ArgumentParser(description="Export ML-Agents checkpoint to ONNX.")
    parser.add_argument("--run-id", default="nav-ppo-02")
    parser.add_argument("--behavior", default="NavAgent")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to checkpoint.pt (default: results/<run-id>/<behavior>/checkpoint.pt)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to configuration.yaml (default: results/<run-id>/configuration.yaml)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path without extension (default: results/<run-id>/<behavior>)",
    )
    args = parser.parse_args()

    run_id = args.run_id
    behavior = args.behavior
    checkpoint_path = args.checkpoint or os.path.join(
        "results", run_id, behavior, "checkpoint.pt"
    )
    config_path = args.config or os.path.join("results", run_id, "configuration.yaml")
    output_path = args.output or os.path.join("results", run_id, behavior)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    policy_state = checkpoint_state["Policy"]

    obs_size = infer_obs_size(policy_state, checkpoint_state)
    action_size = infer_continuous_actions(policy_state)

    policy = build_policy(config_path, behavior, obs_size, action_size)
    missing, unexpected = policy.actor.load_state_dict(policy_state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing={missing}, unexpected={unexpected}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializer = ModelSerializer(policy)
    onnx_path = f"{output_path}.onnx"
    with exporting_to_onnx():
        torch.onnx.export(
            policy.actor,
            serializer.dummy_input,
            onnx_path,
            opset_version=SerializationSettings.onnx_opset,
            input_names=serializer.input_names,
            output_names=serializer.output_names,
            dynamic_axes=serializer.dynamic_axes,
            dynamo=False,
        )
    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}")
        sys.exit(1)
