from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from onnxsim import simplify
from onnx_tf.backend import prepare as tf_prepare
from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_tf_saved_model
from time import monotonic
from torch import Tensor
from torch.nn import (Dropout, Linear, Module, ReLU, Sequential) 
from torch.optim import (AdamW, Optimizer)
from torch.optim.lr_scheduler import (OneCycleLR, _LRScheduler)
from torch.utils.data import (DataLoader, random_split)
from torch.cuda.amp import (autocast, GradScaler)
from torchsummary import summary
from torchvision.datasets import MNIST
from torchvision.transforms import (CenterCrop, Compose, GaussianBlur, Lambda, RandomApply, RandomAffine, ToTensor)
from tqdm import tqdm

import onnx
import torch
import torch.quantization as quant
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.onnx as tonnx
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)


class Split(Enum):
    TRAIN: str = "Train"
    VALID: str = "Valid"
    TEST : str = "Test"


def fit_step(model: Module, loader: DataLoader, optim: Optimizer, scheduler: _LRScheduler, scaler: GradScaler, split: Split, mixed: bool = True) -> tuple[float, float]:
    try: device = next(model.parameters()).device
    except: device = "cpu"

    train = split == Split.TRAIN
    total_loss, total_acc = 0.0, 0.0
    
    model.train(mode=train)
    with torch.inference_mode(mode=not train), tqdm(loader, desc=split.value) as pbar:
        for x, l in pbar:
            x, l = x.to(device=device), l.to(device=device)

            with autocast(enabled=mixed):
                logits = model(x)
                loss = F.cross_entropy(logits, l)
                acc = (logits.argmax(dim=1) == l).sum() / x.size(0)

            if split == Split.TRAIN:
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scheduler.step()
                scaler.update()

            total_loss += loss.item() / len(loader)
            total_acc += acc.item() / len(loader)
            pbar.set_postfix(loss=f"{total_loss:.2e}", acc=f"{total_acc * 100:.2f}%", lr="None" if scheduler is None else f"{scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss, total_acc


def fit(model: Module, epochs: int, loaders: dict[Split, DataLoader]) -> None:
    summary(model, (28 * 28, ))
    
    optim = AdamW(model.parameters())
    scheduler = OneCycleLR(optim, max_lr=1e-3, total_steps=len(loaders[Split.TRAIN]) * epochs)
    scaler = GradScaler()

    for _ in tqdm(range(epochs), desc="Epoch"):
        train_loss, train_acc = fit_step(model, loaders[Split.TRAIN], optim, scheduler, scaler, Split.TRAIN)
        valid_loss, valid_acc = fit_step(model, loaders[Split.VALID], None, None, None, Split.VALID)
    test_loss, test_acc = fit_step(model, loaders[Split.TEST], None, None, None, Split.TEST)
    print("----------------------------------")
    print(f"[TRAIN] loss: {train_loss:.2e} acc: {train_acc*100:.2f}%")
    print(f"[VALID] loss: {valid_loss:.2e} acc: {valid_acc*100:.2f}%")
    print(f"[TEST]  loss: {test_loss :.2e} acc: {test_acc *100:.2f}%")
    print("----------------------------------")


@torch.inference_mode()
def benchmark_step(model: Module, input_size: tuple, n: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[float, float, float]:
    model = model.to(device=device, dtype=dtype)
    x = torch.rand(batch_size, *input_size).to(device=device, dtype=dtype)
    dts = []
    for _ in range(n):
        t_start = monotonic()
        model(x)
        dts.append(monotonic() - t_start)
    return min(dts), sum(dts) / len(dts), max(dts)
    
    

@torch.inference_mode()
def benchmark(model: Module, input_size: tuple, n: int, devices: list[str], dtypes: list[str]) -> None:
    model = model.eval()
    print("----------------------------------")
    for device in devices:
        for dtype in dtypes:
            try:
                for batch_size in [2 ** i for i in range(12)]:
                    min_dt, avg_dt, max_dt = benchmark_step(model, input_size, n, batch_size, torch.device(device), dtype)
                    print(f"batch_size: {str(batch_size):>4s} device: {device:<6s} dtype: {dtype} | min: {min_dt * 1_000:.2f} ms  avg: {avg_dt * 1_000:.2f} ms  max: {max_dt * 1_000:.2f} ms")
                print("----------------------------------")
            except: ...

@torch.inference_mode()
def sparsity(model: Module, min_value: float) -> float:
    parameters = list(model.parameters())
    s = sum((param.data.abs() <= min_value).sum().item() for param in parameters)
    n = sum(param.data.flatten().size(0) for param in parameters)
    print("----------------------------------")
    print(f"Sparsity: {s / n * 100:.2f}%  {s}/{n}")
    print("----------------------------------")
    return s / n


def prune_desc(model: Module, desc: list[tuple[Module, str]], amount: float) -> None:
    prune.global_unstructured(desc, pruning_method=prune.L1Unstructured, amount=amount)
    for m, n in desc: prune.remove(m, n)


def quantize(model: Module, delete: list[Module], fuse: list[tuple[str, str]], loader: DataLoader) -> Module:
    for m in delete: del m
    q_model = Sequential(OrderedDict({"q": quant.QuantStub(), "m": model.float().eval().cpu(), "d": quant.DeQuantStub()}))
    q_model.qconfig = quant.get_default_qconfig('fbgemm')
    q_model = quant.prepare(quant.fuse_modules(q_model, fuse))
    q_model.fit_step(loader, None, None, None, Split.TEST, mixed=False)
    return quant.convert(q_model)


def jit(model: Module, example: Tensor, device: torch.device, dtype: torch.dtype) -> torch.jit.ScriptModule:
    return torch.jit.optimize_for_inference(torch.jit.trace(model.eval().to(device=device), example.to(device=device, dtype=dtype)))


Module.fit_step = fit_step
Module.fit = fit
Module.benchmark = benchmark
Module.sparsity = sparsity
Module.prune = prune_desc
Module.save = lambda m, path: torch.save(m.state_dict(), path)
Module.quantize = quantize
Module.jit = jit


if __name__ == "__main__":
    aug = Compose([RandomAffine(25, (0.2, 0.2), (0.9, 1.1), 4), CenterCrop(28), RandomApply([GaussianBlur(3)], 0.2)])
    flat = Lambda(lambda x: x.reshape(-1))

    sets = {
        Split.TRAIN: MNIST("/tmp/mnist", download=True, train=True, transform=Compose([ToTensor(), aug, flat])),
        **{s: d for s, d in zip([Split.VALID, Split.TEST], random_split(MNIST("/tmp/mnist", download=True, train=False, transform=Compose([ToTensor(), flat])), [0.5, 0.5]))},
    }

    loaders = {
        Split.TRAIN: DataLoader(sets[Split.TRAIN], batch_size=256, shuffle=True,  pin_memory=True, drop_last=True,  num_workers=22),
        Split.VALID: DataLoader(sets[Split.VALID], batch_size=256, shuffle=False, pin_memory=True, drop_last=False, num_workers=22),
        Split.TEST : DataLoader(sets[Split.TEST],  batch_size=256, shuffle=False, pin_memory=True, drop_last=False, num_workers=22),
    }

    h_dim = 512
    model = Sequential(OrderedDict({
        "i": Sequential(OrderedDict({"l": Linear(28 * 28, h_dim), "d": Dropout(0.2), "a": ReLU()})),
        "h": Sequential(OrderedDict({"l": Linear(  h_dim, h_dim), "d": Dropout(0.2), "a": ReLU()})),
        "d": Dropout(0.2),
        "o": Linear(h_dim, 10),
    })).cuda()
    model.o.weight.data.normal_(0, 0.02)
    model.o.bias.data.zero_()

    # ==== TRAIN
    model.fit(10, loaders)
    model.benchmark((28 * 28, ), 100, ["cpu", "cuda:0"], [torch.float32, torch.half])
    
    # ==== SPARSITY EXPLORATION
    # from copy import deepcopy
    # import matplotlib.pyplot as plt
    # amounts, accuracies = [0.2, 0.5, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9], []
    # for amount in [0.2, 0.5, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]:
    #     prunned = deepcopy(model)
    #     prunned.prune(((model.i.l, "weight"), (model.h.l, "weight"), (model.o, "weight")), amount)
    #     accuracies.append(prunned.fit_step(loaders[Split.TEST], None, None, None, Split.TEST)[1])
    #     prunned.sparsity(0.0)
    # plt.figure()
    # plt.title("Accuracy vs Amounts")
    # plt.plot([amount * 100 for amount in amounts], [acc * 100 for acc in accuracies])
    # plt.xlabel("Amount (%)")
    # plt.ylabel("Accuracy (%)")
    # plt.show()

    # ==== SPARSITY PRUNNING
    # model.sparsity(0.0)
    # model.prune(((model.i.l, "weight"), (model.h.l, "weight"), (model.o, "weight")), 0.7)
    # model.fit_step(loaders[Split.TEST], None, None, None, Split.TEST)
    # model.sparsity(0.0)

    # ==== SAVE MODEL
    model.save("mnist.pt")

    # ==== UINT8 QUANTIZATION
    # q_model = model.quantize(delete=[model.i.d, model.h.d, model.d], fuse=[["m.i.l", "m.i.a"], ["m.h.l", "m.h.a"]], loader=loaders[Split.TEST])
    # q_model.benchmark((28 * 28, ), 100, ["cpu"], [None])
    # q_model.fit_step(loaders[Split.TEST], None, None, None, Split.TEST, mixed=False)

    # ==== JIT + UINT8 QUANTIZATION
    # traced_q_model = q_model.jit(next(iter(loaders[Split.TEST]))[0], device="cpu", dtype=torch.float32)
    # traced_q_model.benchmark((28 * 28, ), 100, ["cpu"], [None])

    # ==== ONNX EXPORT
    tonnx.export(
        model.eval().float().cpu(),
        next(iter(loaders[Split.TEST]))[0],
        "mnist.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # ==== ONNX SANITY CHECK
    o_model = onnx.load("mnist.onnx")
    onnx.checker.check_model(o_model)

    # ==== MODEL ONNX GRAPH SIMPLIFICATION
    o_model, check = simplify(o_model)

    # ==== TENSORFLOW JS EXPORT
    tf_prepare(o_model).export_graph("mnist.pb")
    convert_tf_saved_model("mnist.pb", "mnist.js")