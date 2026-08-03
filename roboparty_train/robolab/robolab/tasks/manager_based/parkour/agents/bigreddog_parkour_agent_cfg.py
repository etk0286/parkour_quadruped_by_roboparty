"""RSL-RL configuration for the BigReddog parkour task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BigReddogEncoderMoEActorCriticCfg:
    """Actor-critic configuration with a depth-image encoder.

    Observation and action dimensions are inferred by RSL-RL from the
    environment, so this policy does not encode the humanoid's 23-DoF action
    dimension and can be used with BigReddog's 12 actuated joints.
    """

    # AMPRunner currently resolves this field with eval(), while OnPolicyRunner
    # also supports the simple rsl_rl class name through resolve_callable().
    class_name: str = "EncoderMoEActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    num_moe_experts: int = 5
    moe_gate_hidden_dims: list[int] = []
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    # Keep normalization disabled because the depth encoder is exported
    # separately for deployment.
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = ["depth_image"]
    encoder_cfg: dict = {
        "channels": [4],
        "kernel_sizes": [3],
        "strides": [1],
        "hidden_sizes": [256, 256],
        "output_size": 128,
        "paddings": [1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
    }
    encoder_onnx_stems: dict[str, str] = {"depth_image": "depth_encoder"}
    encoder_onnx_sequential_idx: int = 0


@configclass
class BigReddogParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard PPO runner for BigReddog parkour without AMP style rewards."""

    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "bigreddog_parkour"
    wandb_project = "bigreddog_parkour"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = BigReddogEncoderMoEActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms:PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
    )
