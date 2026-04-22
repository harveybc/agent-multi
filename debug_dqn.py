import sys, json, os
from app.plugin_loader import load_plugin

try:
    AgentCls, _ = load_plugin("agent.plugins", "dqn_agent")
    inst = AgentCls(config={})
    print("has wrap_env:", hasattr(inst, "wrap_env"), callable(getattr(inst,"wrap_env",None)))

    EnvCls, _ = load_plugin("env.plugins", "gym_fx_env")
    cfg_path = "examples/config/dqn_btc_1h_twelve_atr.json"
    cfg = json.load(open(cfg_path))
    cfg["total_timesteps"]=200; cfg["learning_starts"]=50
    env_plugin = EnvCls(config=cfg)
    base_env = env_plugin.make_env(cfg)
    print("base obs space type:", type(base_env.observation_space).__name__)
    print("base obs space:", base_env.observation_space)

    w = inst.wrap_env(base_env, cfg)
    print("wrapped obs space type:", type(w.observation_space).__name__)
    print("wrapped obs space:", w.observation_space)

    # now invoke build
    print("---building DQN---")
    try:
        model = inst.build(w, cfg)
        print("build OK, policy:", type(model.policy).__name__)
    except Exception as e:
        import traceback
        print("BUILD FAIL:", type(e).__name__, e)
        traceback.print_exc()
except Exception as e:
    import traceback
    print("SCRIPT FAIL:", type(e).__name__, e)
    traceback.print_exc()
