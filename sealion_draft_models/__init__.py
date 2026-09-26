def register():
    from vllm import ModelRegistry

    from .qwen3_dflash2_adaflash import DFlash2Qwen3AdaFlashForCausalLM

    for name, cls in [
        ("DFlash2DraftModelAdaFlash", DFlash2Qwen3AdaFlashForCausalLM),
    ]:
        if name not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(name, cls)
