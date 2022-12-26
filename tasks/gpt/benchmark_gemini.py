from functools import partial
from time import time

import colossalai
from colossalai.gemini import GeminiManager
from colossalai.gemini.chunk import init_chunk_manager
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import ZeroDDP
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.zero import ZeroOptimizer
from omegaconf import OmegaConf

from conchmark.models import GPTLMLoss, GPTLMModel
from conchmark.utils import (
    get_model_size,
    get_random_data_with_mask,
    get_tflops,
    memory_info,
)


def main():
    conf = OmegaConf.load("gemini_config.yaml")
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    # colo_set_process_memory_fraction(0.5)
    logger = get_dist_logger()
    logger.info(memory_info(), ranks=[0])

    # build GPT model
    with ColoInitContext(device=get_current_device()):
        model = GPTLMModel(
            hidden_size=conf.gpt_config.hidden_size,
            num_layers=conf.gpt_config.num_layers,
            num_attention_heads=conf.gpt_config.num_attention_heads,
            checkpoint=conf.gpt_config.checkpoint,
        )
    model_numel = get_model_size(model)
    logger.info(f"Model numel: {model_numel}", ranks=[0])
    logger.info(memory_info(), ranks=[0])
    get_tflops_func = partial(get_tflops, model_numel, conf.batch_size, conf.seq_len)

    chunk_manager = init_chunk_manager(
        model=model,
        init_device=get_current_device(),
        hidden_dim=conf.gpt_config.hidden_size,
        search_range_mb=64,
    )

    gemini_manager = GeminiManager(conf.placement_policy, chunk_manager)
    if conf.placement_policy == "const":
        gemini_manager._placement_policy.set_const_memory_boundary(10 * 1024)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)
    logger.info(memory_info(prefix="After init model, "), ranks=[0])
    # logger.info(chunk_manager, ranks=[0])
    logger.info(memory_info(), ranks=[0])

    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    optimizer = ZeroOptimizer(
        optimizer, model, initial_scale=2**5, gpu_margin_mem_ratio=0.0
    )

    # build criterion
    criterion = GPTLMLoss()

    model.train()
    num_steps = conf.number_steps

    def one_turn():
        # we just use randomly generated data here
        input_ids, attn_mask = get_random_data_with_mask(
            conf.batch_size, conf.seq_len, conf.vocab_size
        )

        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(memory_info(prefix=f"[{n + 1}/{num_steps}] Forward "), ranks=[0])
        fwd_end = time()
        fwd_time = fwd_end - start

        optimizer.backward(loss)
        logger.info(memory_info(prefix=f"[{n + 1}/{num_steps}] Backward "), ranks=[0])
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        optimizer.step()
        logger.info(
            memory_info(prefix=f"[{n+1}/{num_steps}] Optimizer step "), ranks=[0]
        )
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(
            f"[{n + 1}/{num_steps}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        tflops_list.append(get_tflops_func(step_time))

    tflops_list = []
    for n in range(num_steps):
        one_turn()

    tflops_list.sort()
    middle = num_steps >> 1
    logger.info(f"Median TFLOPS is {tflops_list[middle]:.3f}")

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              schedule=schedule(wait=1, warmup=2, active=2),
    #              on_trace_ready=tensorboard_trace_handler(
    #                  f'opt-6.7b/v3-full-{PLACEMENT_POLICY}-{dist.get_world_size()}gpu'),
    #              record_shapes=True,
    #              profile_memory=True) as prof:
    #     for n in range(NUM_STEPS):
    #         one_turn()
    #         prof.step()
    # dist.barrier()


if __name__ == "__main__":
    main()
