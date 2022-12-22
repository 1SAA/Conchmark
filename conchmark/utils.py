import psutil
import torch
import torch.nn as nn


def get_random_data_with_mask(batch_size, seq_len, vocab_size):
    my_dev = torch.device(torch.cuda.current_device())
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=my_dev)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def cur_cpu_mem_mb():
    return psutil.Process().memory_info().rss / 1024**2


def max_gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def cur_gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2


def memory_info(prefix=""):
    return "{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB".format(
        prefix, cur_gpu_mem_mb(), max_gpu_mem_mb(), cur_cpu_mem_mb()
    )


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel
