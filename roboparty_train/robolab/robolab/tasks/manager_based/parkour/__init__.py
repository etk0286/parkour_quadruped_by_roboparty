import gymnasium as gym

from . import agents

gym.register(
    id="RPO-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="RPO-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="BigReddog-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_env:BigReddogParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_env_cfg:BigReddogParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bigreddog_parkour_agent_cfg:BigReddogParkourPPORunnerCfg",
    },
)

gym.register(
    id="BigReddog-Parkour0",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_env:BigReddogParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_env_cfg0:BigReddogParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bigreddog_parkour_agent_cfg:BigReddogParkourPPORunnerCfg",
    },
)

gym.register(
    id="BigReddog-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_env:BigReddogParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_env_cfg:BigReddogParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bigreddog_parkour_agent_cfg:BigReddogParkourPPORunnerCfg",
    },
)

gym.register(
    id="BigReddog-Parkour-Play0",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_env:BigReddogParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_env_cfg0:BigReddogParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bigreddog_parkour_agent_cfg:BigReddogParkourPPORunnerCfg",
    },
)

gym.register(
    id="BigReddog-Parkour-AMP",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_amp_env:BigReddogParkourAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_amp_env_cfg:BigReddogParkourAmpEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.bigreddog_parkour_amp_agent_cfg:BigReddogParkourAmpRunnerCfg"
        ),
    },
)

gym.register(
    id="BigReddog-Parkour-AMP0",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_amp_env:BigReddogParkourAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigreddog_parkour_amp_env_cfg0:BigReddogParkourAmpEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.bigreddog_parkour_amp_agent_cfg:BigReddogParkourAmpRunnerCfg"
        ),
    },
)

gym.register(
    id="BigReddog-Parkour-AMP-Play",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_amp_env:BigReddogParkourAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.bigreddog_parkour_amp_env_cfg:BigReddogParkourAmpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.bigreddog_parkour_amp_agent_cfg:BigReddogParkourAmpRunnerCfg"
        ),
    },
)

gym.register(
    id="BigReddog-Parkour-AMP-Play0",
    entry_point="robolab.tasks.manager_based.parkour.bigreddog_parkour_amp_env:BigReddogParkourAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.bigreddog_parkour_amp_env_cfg0:BigReddogParkourAmpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.bigreddog_parkour_amp_agent_cfg:BigReddogParkourAmpRunnerCfg"
        ),
    },
)