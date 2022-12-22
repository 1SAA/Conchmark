from transformers import AutoConfig, OPTForCausalLM


def _create_opt(config, checkpoint=True):
    model = OPTForCausalLM(config)
    if checkpoint:
        model.gradient_checkpointing_enable()
    return model


def opt_6b(checkpoint=True):
    config = AutoConfig.from_pretrained("facebook/opt-6.7b")
    return _create_opt(config, checkpoint=checkpoint)


def opt_2b(checkpoint=True):
    config = AutoConfig.from_pretrained("facebook/opt-2.7b")
    return _create_opt(config, checkpoint=checkpoint)


def opt_1b(checkpoint=True):
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
    return _create_opt(config, checkpoint=checkpoint)
