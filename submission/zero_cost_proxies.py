"""Deterministic zero-cost proxies used by the compute-aware NAS search."""

import math

import numpy as np
import torch
import torch.nn as nn


def _get_input_batch(data_source, num_samples, device):
    """Return a fixed input tensor from either a tensor or a DataLoader."""
    if torch.is_tensor(data_source):
        return data_source[:num_samples].detach().to(device)

    collected = []
    count = 0
    for batch in data_source:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        collected.append(data)
        count += data.shape[0]
        if count >= num_samples:
            break
    if not collected:
        raise ValueError("Cannot compute a proxy from an empty data source")
    return torch.cat(collected, dim=0)[:num_samples].detach().to(device)


def compute_synflow(model, input_shape, device):
    """SynFlow saliency, returned on a log scale."""
    original_state = None
    try:
        original_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.abs_()

        inputs = torch.ones(1, *input_shape, device=device)
        model(inputs).sum().backward()

        score = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                score += float((parameter.grad * parameter).abs().sum().item())

        return float(math.log(score + 1e-12))
    except Exception:
        return float("nan")
    finally:
        if original_state is not None:
            try:
                model.load_state_dict(original_state)
            except Exception:
                pass
        model.zero_grad(set_to_none=True)


def compute_jacob_cov(model, data_source, num_classes, device, num_samples=24):
    """
    Jacobian correlation score.

    Rows correspond to samples (rather than output classes), matching the
    quantity the proxy is intended to measure.  A single backward pass keeps
    this affordable and deterministic.
    """
    del num_classes  # Kept in the public signature for evaluator compatibility.
    try:
        model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        inputs = _get_input_batch(data_source, num_samples, device)
        inputs = inputs.clone().requires_grad_(True)
        logits = model(inputs)
        jacobian = torch.autograd.grad(
            logits.sum(), inputs, retain_graph=False, create_graph=False
        )[0]
        jacobian = jacobian.flatten(1)
        jacobian = jacobian / jacobian.norm(dim=1, keepdim=True).clamp_min(1e-8)

        correlation = jacobian @ jacobian.t()
        eps = 1e-5
        eigenvalues = torch.linalg.eigvalsh(
            correlation + eps * torch.eye(correlation.size(0), device=device)
        ).clamp_min(eps)
        score = -torch.sum(torch.log(eigenvalues) + 1.0 / eigenvalues)
        value = float(score.item())
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")
    finally:
        model.zero_grad(set_to_none=True)


def compute_naswot(model, data_source, device, num_samples=24):
    """
    NASWOT log-determinant using agreements of active and inactive neurons.

    The kernel is accumulated inside hooks, so repeated uses of the same ReLU
    module and all residual-block activations contribute without concatenating
    enormous activation tensors.
    """
    handles = []
    try:
        model.to(device)
        model.eval()
        inputs = _get_input_batch(data_source, num_samples, device)
        batch_size = inputs.shape[0]
        kernel = torch.zeros(batch_size, batch_size, device=device)

        def activation_hook(module, hook_inputs, output):
            del module, hook_inputs
            if not torch.is_tensor(output) or output.shape[0] != batch_size:
                return
            binary = (output.detach() > 0).flatten(1).float()
            kernel.add_(binary @ binary.t())
            inverse = 1.0 - binary
            kernel.add_(inverse @ inverse.t())

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                handles.append(module.register_forward_hook(activation_hook))

        if not handles:
            return float("nan")

        with torch.no_grad():
            model(inputs)

        eps = 1e-6
        sign, logdet = torch.linalg.slogdet(
            kernel + eps * torch.eye(batch_size, device=device)
        )
        value = float(logdet.item())
        if sign.item() <= 0 or not math.isfinite(value):
            return float("nan")
        return value
    except Exception:
        return float("nan")
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


def compute_all_proxies(model, data_source, input_shape, num_classes, device):
    """Compute all proxies on the same cached calibration inputs."""
    return {
        "synflow": compute_synflow(model, input_shape, device),
        "jacob_cov": compute_jacob_cov(model, data_source, num_classes, device),
        "naswot": compute_naswot(model, data_source, device),
    }


def _average_tie_ranks(values, higher_is_better=True):
    """Return zero-based average ranks while treating equal/invalid values fairly."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), (len(values) - 1) / 2.0)

    worst = np.nanmin(values[finite]) - max(1.0, np.nanstd(values[finite]))
    clean = np.where(finite, values, worst)
    order = np.argsort(-clean if higher_is_better else clean, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and clean[order[end]] == clean[order[position]]:
            end += 1
        average_rank = 0.5 * (position + end - 1)
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def rank_aggregate(proxy_scores_list, weights=None):
    """
    Weighted Borda aggregation with tie handling.

    Additional higher-is-better fields such as ``efficiency`` can be supplied
    by the NAS controller.  Failed proxies receive the worst rank instead of an
    arbitrary order.
    """
    if not proxy_scores_list:
        return []
    if len(proxy_scores_list) == 1:
        return [0]

    weights = weights or {
        "synflow": 0.25,
        "jacob_cov": 0.75,
        "naswot": 1.0,
        "efficiency": 0.50,
    }
    rank_sum = np.zeros(len(proxy_scores_list), dtype=np.float64)
    total_weight = 0.0

    for name, weight in weights.items():
        if weight <= 0 or not any(name in entry for entry in proxy_scores_list):
            continue
        values = [entry.get(name, float("nan")) for entry in proxy_scores_list]
        rank_sum += float(weight) * _average_tie_ranks(values, higher_is_better=True)
        total_weight += float(weight)

    if total_weight == 0:
        return list(range(len(proxy_scores_list)))
    return list(np.argsort(rank_sum / total_weight, kind="mergesort"))
