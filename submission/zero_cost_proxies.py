"""
zero_cost_proxies.py — Self-contained zero-cost proxy implementations.

Three proxies for rapidly ranking candidate architectures without training:
  - synflow:   Data-agnostic parameter saliency via gradient flow
  - jacob_cov: Data-dependent Jacobian covariance (gradient diversity)
  - naswot:    Data-dependent activation pattern diversity

Plus rank aggregation via Borda count for combining multiple proxy scores.
"""

import time
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Synflow
# =============================================================================

def compute_synflow(model, input_shape, device):
    """
    Compute the Synflow (Synaptic Flow) score for a model.

    Data-agnostic proxy that measures parameter saliency by computing
    the sum of (gradient * parameter) products through the network.

    Args:
        model: nn.Module to evaluate
        input_shape: tuple (C, H, W) of the input tensor
        device: torch device

    Returns:
        float: log(synflow_score) — higher is better
    """
    try:
        # Save original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.to(device)
        model.eval()

        # Set all parameters to their absolute values
        for p in model.parameters():
            p.data = p.data.abs()
            p.requires_grad_(True)

        # Forward pass with all-ones input
        x = torch.ones(1, *input_shape, device=device)
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Score = sum of (grad * param) across all parameters
        score = 0.0
        for p in model.parameters():
            if p.grad is not None:
                score += (p.grad * p.data).sum().item()

        # Restore original state
        model.load_state_dict(original_state)
        model.zero_grad()

        return float(np.log(abs(score) + 1e-8))

    except Exception:
        try:
            model.load_state_dict(original_state)
        except Exception:
            pass
        return 0.0


# =============================================================================
# Jacob_cov
# =============================================================================

def compute_jacob_cov(model, dataloader, num_classes, device, num_samples=32):
    """
    Compute the Jacobian covariance score for a model.

    Data-dependent proxy that measures the diversity of gradients
    of the output w.r.t. the input across different classes.

    Args:
        model: nn.Module to evaluate
        dataloader: DataLoader providing input batches
        num_classes: number of output classes
        device: torch device
        num_samples: number of input samples to use

    Returns:
        float: log|det(J @ J^T)| — higher means more diverse gradients
    """
    try:
        model.to(device)
        model.eval()

        # Get a batch of inputs
        inputs = _get_input_batch(dataloader, num_samples, device)
        inputs.requires_grad_(True)

        # Forward pass
        logits = model(inputs)

        # Compute Jacobian rows for a subset of classes
        num_classes_to_use = min(num_classes, 10)
        jacobian_rows = []

        for j in range(num_classes_to_use):
            model.zero_grad()
            if inputs.grad is not None:
                inputs.grad = None

            logits[:, j].sum().backward(retain_graph=True)

            if inputs.grad is not None:
                grad_j = inputs.grad.detach().clone().flatten()
                jacobian_rows.append(grad_j)

        if len(jacobian_rows) < 2:
            return 0.0

        # Stack into matrix J [num_classes_used, flattened_input_dim]
        J = torch.stack(jacobian_rows)

        # Correlation matrix K = J @ J^T
        K = J @ J.T

        # Score = log|det(K)|
        sign, logdet = torch.slogdet(K)
        score = logdet.item()

        if np.isnan(score) or np.isinf(score):
            return 0.0

        model.zero_grad()
        return score

    except Exception:
        return 0.0


# =============================================================================
# NASWOT
# =============================================================================

class _ActivationHook:
    """Forward hook that captures binary activation patterns after ReLU."""

    def __init__(self):
        self.activations = None

    def __call__(self, module, input, output):
        # Binary activation: which neurons fired (> 0)
        self.activations = (output > 0).float().detach()


def compute_naswot(model, dataloader, device, num_samples=32):
    """
    Compute the NASWOT (NAS Without Training) score for a model.

    Data-dependent proxy that measures the diversity of activation patterns
    across different inputs. More diverse patterns = better architecture.

    Args:
        model: nn.Module to evaluate
        dataloader: DataLoader providing input batches
        device: torch device
        num_samples: number of input samples to use

    Returns:
        float: log|det(K)| where K is the activation kernel matrix
    """
    try:
        model.to(device)
        model.eval()

        # Register hooks on all ReLU modules
        hooks = []
        hook_objs = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                h = _ActivationHook()
                hook_objs.append(h)
                hooks.append(module.register_forward_hook(h))

        if not hooks:
            # No ReLU layers found — try looking for functional relus via any module
            for h in hooks:
                h.remove()
            return 0.0

        # Get a batch and do a forward pass
        inputs = _get_input_batch(dataloader, num_samples, device)

        with torch.no_grad():
            model(inputs)

        # Collect binary activation codes
        # Each hook captured [B, C, H, W] of binary activations
        binary_codes = []
        for h in hook_objs:
            if h.activations is not None:
                # Flatten spatial dims: [B, C*H*W]
                flat = h.activations.view(h.activations.shape[0], -1)
                binary_codes.append(flat)

        # Remove hooks
        for h in hooks:
            h.remove()

        if not binary_codes:
            return 0.0

        # Concatenate across all layers: [B, total_neurons]
        K_binary = torch.cat(binary_codes, dim=1).to(device)

        # Limit feature dim to avoid memory issues
        if K_binary.shape[1] > 50000:
            indices = torch.randperm(K_binary.shape[1])[:50000]
            K_binary = K_binary[:, indices]

        # Kernel matrix: K = binary_codes @ binary_codes^T
        K = K_binary @ K_binary.T

        # Score = log|det(K)|
        sign, logdet = torch.slogdet(K)
        score = logdet.item()

        if np.isnan(score) or np.isinf(score):
            return 0.0

        return score

    except Exception:
        # Clean up hooks on failure
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        return 0.0


# =============================================================================
# Utilities
# =============================================================================

def _get_input_batch(dataloader, num_samples, device):
    """Extract a batch of input tensors from a dataloader."""
    collected = []
    count = 0
    for batch in dataloader:
        # Handle both (data, labels) and (data,) formats
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        collected.append(data)
        count += data.shape[0]
        if count >= num_samples:
            break

    inputs = torch.cat(collected, dim=0)[:num_samples]
    return inputs.to(device)


def compute_all_proxies(model, dataloader, input_shape, num_classes, device):
    """
    Compute all three zero-cost proxies for a model.

    Args:
        model: nn.Module to evaluate
        dataloader: training DataLoader
        input_shape: tuple (C, H, W)
        num_classes: number of output classes
        device: torch device

    Returns:
        dict: {'synflow': float, 'jacob_cov': float, 'naswot': float}
    """
    scores = {}

    # Synflow (data-agnostic)
    t0 = time.time()
    try:
        scores['synflow'] = compute_synflow(model, input_shape, device)
    except Exception:
        scores['synflow'] = 0.0
    t_sf = time.time() - t0

    # Jacob_cov (data-dependent)
    t0 = time.time()
    try:
        scores['jacob_cov'] = compute_jacob_cov(model, dataloader, num_classes, device)
    except Exception:
        scores['jacob_cov'] = 0.0
    t_jc = time.time() - t0

    # NASWOT (data-dependent)
    t0 = time.time()
    try:
        scores['naswot'] = compute_naswot(model, dataloader, device)
    except Exception:
        scores['naswot'] = 0.0
    t_nw = time.time() - t0

    return scores


# =============================================================================
# Rank Aggregation
# =============================================================================

def rank_aggregate(proxy_scores_list):
    """
    Aggregate multiple proxy scores via Borda count.

    Args:
        proxy_scores_list: list of dicts, each mapping proxy_name -> score.
                           One dict per candidate architecture.

    Returns:
        list[int]: indices of architectures sorted by average rank (best first)
    """
    n = len(proxy_scores_list)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Get all proxy names from the first entry
    proxy_names = list(proxy_scores_list[0].keys())

    # Compute per-proxy rankings (higher score = better = lower rank number)
    rank_sums = np.zeros(n)

    for proxy in proxy_names:
        scores = [entry.get(proxy, 0.0) for entry in proxy_scores_list]
        # argsort ascending, then invert to get ranks (0 = best)
        order = np.argsort(scores)[::-1]  # indices sorted by score descending
        ranks = np.zeros(n)
        for rank, idx in enumerate(order):
            ranks[idx] = rank
        rank_sums += ranks

    # Average rank across proxies
    avg_ranks = rank_sums / len(proxy_names)

    # Return indices sorted by average rank (lowest = best)
    return list(np.argsort(avg_ranks))
