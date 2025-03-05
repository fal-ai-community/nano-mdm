# train_MDM.py

import os
import sys
import uuid
import glob
import time
import click
import wandb
import logging
from dataclasses import dataclass
import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors import safe_open
import json
from datetime import datetime
from tqdm import tqdm
import lovely_tensors

lovely_tensors.monkey_patch()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


MASK_TOKEN = 50304 + 1


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    # def next_batch(self):
    #     B = self.B
    #     T = self.T
    #     buf = self.tokens[self.current_position : self.current_position + B * T + 1]
    #     buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
    #     x = (buf[:-1]).view(B, T)  # inputs
    #     y = (buf[1:]).view(B, T)  # targets
    #     # advance current position and load next shard if necessary
    #     self.current_position += B * T * self.num_processes
    #     if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
    #         self.advance()
    #     return x.cuda(), y.cuda()

    def next_batch(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.process_rank + self.num_processes * self.current_position
        )

        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        x_noisy = x.clone()

        rand_per_index = torch.rand((B, T), generator=generator)
        timestep_per_batch = torch.rand((B, 1), generator=generator)
        for b in range(B):  # on batch axis
            x_noisy[b, rand_per_index[b] < timestep_per_batch[b]] = MASK_TOKEN

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x_noisy.cuda(), x.cuda()


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
)


def log_gpu_memory():
    """Log GPU memory usage for all available GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
            logging.info(
                f"GPU {i} Memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached"
            )


@dataclass
class MDMConfig:
    vocab_size: int = 50304 + 1  # NanoMDM's vocab size + 1 for mask token
    block_size: int = 1024  # Maximum sequence length
    n_layer: int = 12
    n_head: int = 6
    n_embed: int = 768
    wte_init_std: float = 0.02
    v_residual: bool = False
    ff_expand: float = 1.0


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return (
            self.cos_cached[None, :, None, :],
            self.sin_cached[None, :, None, :],
        )  # [1, seq_len, 1, dim]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = self.n_embed // self.n_head
        assert self.n_embed % self.n_head == 0
        self.c_q = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.c_k = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.c_v = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.c_proj.weight.data.zero_()

        if config.v_residual:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
            self.lamb2 = nn.Parameter(torch.tensor(0.5))
        else:
            self.lamb1 = 1.0
            self.lamb2 = 0.0

    @torch.compile()
    def forward(self, x, kv_cache=None, freq=None, v1=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)  # B, T, n_head, D
        cos, sin = freq

        if v1 is None:
            v1 = v

        v = self.lamb1 * v + self.lamb2 * v1.view_as(v)

        if kv_cache is not None:
            raise NotImplementedError("kv_cache is not implemented")
            k_cache, v_cache = kv_cache

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))

            if k_cache is not None:
                if isinstance(k_cache, int):
                    k_cache = k
                    v_cache = v
                else:
                    k = torch.cat([k_cache, k], dim=1)
                    v = torch.cat([v_cache, v], dim=1)  # it cats in T dim.

                new_kv_cache = (k, v)

            # do classic attention.
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )

        else:
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            new_kv_cache = None
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return (y, v1), new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embed, int(config.ff_expand * config.n_embed), bias=False
        )
        self.c_proj = nn.Linear(
            int(config.ff_expand * config.n_embed), config.n_embed, bias=False
        )
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, freq=None, v1=None):
        (attn_out, v1), new_kv_cache = self.attn(
            F.rms_norm(x, (x.size(-1),)), kv_cache, freq, v1=v1
        )
        x = x + attn_out
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return (x, v1), new_kv_cache


def nearest_mult_of(x, mult):
    return int(math.ceil(x / mult) * mult)


class MDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    nearest_mult_of(config.vocab_size, 256), config.n_embed
                ),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        self.rotary = Rotary(config.n_embed // (config.n_head))
        self.transformer.wte.weight.data.normal_(mean=0.0, std=config.wte_init_std)

    def forward(self, idx, targets=None):
        # forward the MDM model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embed)
        x = F.rms_norm(x, (x.size(-1),))
        freq = self.rotary(x)
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, freq=freq, v1=v1)[0]

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = logits.float()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:

            # where mask is
            B, T = idx.shape
            mask = idx == MASK_TOKEN

            # number of tokens actually masked
            ratio = mask.sum(dim=1) / idx.shape[1]  # B

            # Batch size
            ratio = ratio.view(B, 1).repeat(1, T) + 1e-4

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="none",
            )  # B * T, V -> B * T

            loss = (loss * mask.view(-1) / ratio.view(-1)).mean()

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


@click.command()
@click.option("--run_name", default="test", help="Name of the run")
@click.option("--project_name", default="nanoMDM", help="Name of the project")
@click.option(
    "--train_data",
    default="data/fineweb10B/fineweb_train_*.bin",
    help="Path to training data",
)
@click.option(
    "--val_data",
    default="data/fineweb10B/fineweb_val_*.bin",
    help="Path to validation data",
)
@click.option(
    "--global_batch_size", default=8 * 64, help="Global batch size across all GPUs"
)
@click.option("--per_gpu_batch_size", default=64, help="Per GPU batch size")
@click.option("--num_iterations", default=5100, help="Number of training iterations")
@click.option("--learning_rate", default=3.6e-3, help="Learning rate")
@click.option("--weight_decay", default=0.1, help="Weight decay")
@click.option("--warmup_iters", default=0, help="Warmup iterations")
@click.option("--warmdown_iters", default=1450, help="Warmdown iterations")
@click.option("--val_every", default=125, help="Validation frequency")
@click.option("--save_every", default=1000, help="Checkpoint save frequency")
@click.option("--n_embed", default=768, help="Embedding dimension")
@click.option("--init_ckpt", default=None, help="Path to initial checkpoint")
@click.option("--vres", default=False, help="Use vres")
@click.option("--n_layer", default=12, help="Number of layers")
@click.option("--n_head", default=6, help="Number of heads")
@click.option("--ff_expand", default=2, help="FF expand")
@click.option("--sequence_length", default=1024, help="Sequence length")
@click.option("--tags", default="", help="Tags for the run")
def train(
    run_name,
    project_name,
    train_data,
    val_data,
    global_batch_size,
    per_gpu_batch_size,
    num_iterations,
    learning_rate,
    weight_decay,
    warmup_iters,
    warmdown_iters,
    val_every,
    save_every,
    n_embed,
    init_ckpt,
    vres,
    n_layer,
    n_head,
    ff_expand,
    sequence_length,
    tags,
):
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # test NCCL
    print(f"Rank {ddp_rank}, World size: {ddp_world_size}")

    tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {ddp_rank}, tensor: {tensor}")

    if master_process:
        logging.info(f"Starting training with {ddp_world_size} GPUs")
        logging.info(f"Model config: layers={n_layer}, embed_dim={n_embed}")
        log_gpu_memory()

    # fix all the seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    grad_accum_steps = int(global_batch_size // (per_gpu_batch_size * ddp_world_size))

    if master_process:
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
        logging.info(
            f"Effective batch size: {per_gpu_batch_size * ddp_world_size * grad_accum_steps}"
        )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{date_time}_{run_name}"

    config = MDMConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embed=n_embed,
        v_residual=vres,
        ff_expand=ff_expand,
    )

    model = MDM(config)
    model = model.to(device)

    if init_ckpt is not None:
        print(f"Loading checkpoint from {init_ckpt}")
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # count total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    # count total activated parameters.
    total_activated_params = 0
    for name, param in model.named_parameters():
        if any(name.startswith(x) for x in ["lm_head", "rotary", "wte"]):
            continue
        total_activated_params += param.numel()

    print(f"Total number of activated parameters: {total_activated_params}")

    # broadcast all parameters to ensure they start in sync across GPUs
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module

    # Create datasets
    if master_process:
        logging.info("Loading datasets...")
        t0 = time.time()

    train_loader = DistributedDataLoader(
        train_data, per_gpu_batch_size, sequence_length, ddp_rank, ddp_world_size
    )
    val_loader = DistributedDataLoader(
        val_data, per_gpu_batch_size, sequence_length, ddp_rank, ddp_world_size
    )

    def get_lr(it):
        if it < warmup_iters:
            return it / warmup_iters
        if it > num_iterations - warmdown_iters:
            return (num_iterations - it) / warmdown_iters
        return 1.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate * 256 / n_embed,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    if master_process:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "train_data": train_data,
                "val_data": val_data,
                "global_batch_size": global_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_iters": warmup_iters,
                "warmdown_iters": warmdown_iters,
                "n_embed": n_embed,
                "n_head": n_head,
                "n_layer": n_layer,
                "vres": vres,
                "total_params": total_params,
                "total_activated_params": total_activated_params,
                "optimizer_type": "AdamW",
            },
            tags=tags.split(",") if tags else [],
        )
        wandb.run.log_code(".")

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()

    x, y = train_loader.next_batch()
    if master_process:
        logging.info("Starting training loop...")
        t_start = time.time()
        tokens_processed = 0

    for step in range(num_iterations):
        for micro_step in range(grad_accum_steps):
            with ctx:
                # print(f"Rank {ddp_rank}, batch: {x}")
                logits, loss = model(x.to(device), y.to(device))
                loss = loss / grad_accum_steps
                # print(f"Rank {ddp_rank}, loss: {loss.item()}")
                loss.backward()

            if master_process:
                tokens_processed += x.numel() * ddp_world_size

            # Get next batch for the next iteration
            x, y = train_loader.next_batch()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if master_process:
            lr = scheduler.get_last_lr()[0]
            tokens_per_sec = tokens_processed / (time.time() - t_start)
            logging.info(
                f"step: {step}, loss: {loss.item():.4f}, lr: {lr:.2e}, tokens/sec: {tokens_per_sec:.1f}"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/step": step,
                    "train/tokens_per_sec": tokens_per_sec,
                }
            )

        if step % val_every == 0:
            if master_process:
                logging.info("Starting validation...")
                t_val_start = time.time()

            model.eval()
            val_losses = []
            val_loader.reset()
            pbar = tqdm(val_loader)
            counter = 0
            for _ in range(20):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    _, val_loss = model(x_val.to(device), y_val.to(device))
                    val_losses.append(val_loss.item())
                    pbar.set_description(f"val_loss: {val_loss.item():.4f}")

            val_loss = torch.tensor(np.mean(val_losses)).to(device)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if master_process:
                val_time = time.time() - t_val_start
                logging.info(f"Validation completed in {val_time:.2f}s")
                logging.info(f"step: {step}, val_loss: {val_loss.item():.4f}")
                wandb.log(
                    {
                        "val/loss": val_loss.item(),
                        "val/time": val_time,
                        f"val_perstep/loss_{step}": val_loss.item(),
                    }
                )

            model.train()

        if master_process and step % save_every == 0 and step > 0:
            logging.info(f"Saving checkpoint at step {step}...")
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": raw_model.config,
            }
            os.makedirs(f"logs/ckpts_{run_id}", exist_ok=True)
            ckpt_path = f"logs/ckpts_{run_id}/step_{step}.pt"
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")

    if master_process:
        total_time = time.time() - t_start
        total_tokens = tokens_processed
        logging.info(f"Training completed in {total_time:.2f}s")
        logging.info(f"Average tokens/sec: {total_tokens/total_time:.1f}")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
