"""Compact probe-guided collusion defense for federated learning.

Supports CIFAR-10, FEMNIST, Shakespeare, and N-BaIoT.

Per round:
1. sample client groups;
2. compare each group's clean and triggered probe behavior with the full aggregate;
3. flag unusually trigger-sensitive groups;
4. convert repeated suspicious co-membership into client/pair risk;
5. decay risk across rounds;
6. downweight or optionally reject high-risk clients before aggregation.

Attacker IDs, when supplied, are used only for evaluation telemetry.
"""

import copy
import csv
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


_CLIENT_MEMORY = {}
_PAIR_MEMORY = {}
_ROUND_CACHE = {}


def normalize_weights(weights):
    values = [float(x) for x in weights]
    total = sum(values)
    return [x / total for x in values] if total > 0 else [1.0 / len(values)] * len(values)


def aggregate_updates(w_updates, indices, global_state, weights):
    out = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for idx, weight in zip(indices, weights):
        for key, value in w_updates[int(idx)].items():
            if key in out and torch.is_floating_point(out[key]):
                out[key] += value.to(out[key].device, out[key].dtype) * float(weight)
    return out


def add_update(global_state, update):
    return {
        key: value + update[key].to(value.device, value.dtype)
        if key in update and torch.is_floating_point(value)
        else value.clone()
        for key, value in global_state.items()
    }


def update_vector(update):
    parts = []
    for key, value in update.items():
        if not torch.is_floating_point(value):
            continue
        if key.split(".")[-1] in ("num_batches_tracked", "running_mean", "running_var"):
            continue
        parts.append(value.detach().reshape(-1).float().cpu())
    return torch.cat(parts) if parts else torch.zeros(1)


def median_clip_updates(w_updates):
    if not w_updates:
        return []
    norms = [float(torch.norm(update_vector(u), p=2)) for u in w_updates]
    clip_norm = float(np.median(norms)) + 1e-12
    out = []
    for update, norm in zip(w_updates, norms):
        scale = min(1.0, clip_norm / (norm + 1e-12))
        out.append({
            k: v.clone() * scale if torch.is_floating_point(v) else v.clone()
            for k, v in update.items()
        })
    return out


def get_probe_batches(dataset_test, args, device):
    loader = DataLoader(
        dataset_test,
        batch_size=max(1, int(getattr(args, "probe_batch_size", 64))),
        shuffle=False,
    )
    limit = max(1, int(getattr(args, "probe_num_batches", 2)))
    batches = []

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, y = batch[:2]
        elif isinstance(batch, dict):
            x = next(batch[k] for k in ("x", "features", "inputs", "tokens") if k in batch)
            y = next(batch[k] for k in ("y", "labels", "label", "targets") if k in batch)
        else:
            raise TypeError("Expected dataset batches containing inputs and labels.")

        x = x if torch.is_tensor(x) else torch.as_tensor(x)
        y = y if torch.is_tensor(y) else torch.as_tensor(y)
        batches.append((x.to(device), y.to(device)))
        if len(batches) >= limit:
            break

    if not batches:
        raise ValueError("dataset_test produced no probe batches")
    return batches


def apply_probe_trigger(x, args):
    x = x.clone()
    dataset = str(getattr(args, "dataset", "")).lower()

    if x.dim() == 4:
        _, channels, height, width = x.shape
        patch = max(1, min(int(getattr(args, "probe_patch_size", 6)), height, width))
        col = width - patch

        if dataset in ("cifar", "cifar10", "cifar-10") and channels == 3:
            mean = torch.tensor(
                [0.4914, 0.4822, 0.4465],
                device=x.device,
                dtype=x.dtype,
            ).view(1, 3, 1, 1)
        
            std = torch.tensor(
                [0.2470, 0.2435, 0.2616],
                device=x.device,
                dtype=x.dtype,
            ).view(1, 3, 1, 1)
        
            x[:, :, :patch, col:] = (
                torch.ones_like(mean) - mean
            ) / std
        
        elif dataset == "femnist" and channels == 1:
            # FEMNIST normalization:
            # (1.0 - 0.5) / 0.5 = 1.0
            x[:, :, :patch, col:] = 1.0
        
        else:
            value = getattr(args, "probe_image_value", None)
        
            value = (
                max(1.0, float(x.detach().max()))
                if value is None
                else float(value)
            )
        
            x[:, :, :patch, col:] = value

    if dataset in ("shakespeare", "shake") and x.dim() >= 2:
        repeat = min(max(1, int(getattr(args, "probe_trigger_repeat", 3))), x.shape[-1])
        x[..., :repeat] = int(getattr(args, "probe_trigger_token", 1))
        return x

    if dataset in ("nbaiot", "n-baiot", "n_baiot") and x.dim() >= 2:
        indices = getattr(args, "probe_feature_indices", (0, 1, 2))
        if isinstance(indices, str):
            indices = [int(v.strip()) for v in indices.split(",") if v.strip()]
        indices = [int(i) for i in indices if 0 <= int(i) < x.shape[-1]]
        if indices:
            x[..., indices] = float(getattr(args, "probe_tabular_value", 999.0))
    return x


def output_probabilities(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.dim() == 3:
        output = output[:, -1, :]
    if output.dim() == 1:
        p1 = torch.sigmoid(output).view(-1, 1)
        return torch.cat([1.0 - p1, p1], dim=1)
    if output.dim() == 2 and output.size(1) == 1:
        p1 = torch.sigmoid(output)
        return torch.cat([1.0 - p1, p1], dim=1)
    if output.dim() != 2:
        raise ValueError("Unsupported probe output shape: {}".format(tuple(output.shape)))
    return F.softmax(output, dim=1)


def model_probe_stats(model, clean_batches, trigger_batches, target_class):
    clean_vals, trigger_vals = [], []
    with torch.no_grad():
        for (clean_x, _), trigger_x in zip(clean_batches, trigger_batches):
            clean = output_probabilities(model(clean_x))
            trigger = output_probabilities(model(trigger_x))
            target = min(max(int(target_class), 0), clean.shape[1] - 1)
            clean_vals.append(float(clean[:, target].mean().cpu()))
            trigger_vals.append(float(trigger[:, target].mean().cpu()))
    return float(np.mean(clean_vals)), float(np.mean(trigger_vals))


def deterministic_seed(args, users_idx, n_clients):
    base = int(getattr(args, "seed", 1))
    round_idx = int(getattr(args, "current_round", 0))
    user_term = n_clients * 9973 if users_idx is None else sum(
        (i + 1) * int(uid) * 131 for i, uid in enumerate(users_idx)
    )
    return base + round_idx * 104729 + user_term


def sample_groups(n_clients, group_size, num_groups, seed):
    if n_clients == 1:
        return [(0,)]

    group_size = max(2, min(int(group_size), n_clients))
    maximum = math.comb(n_clients, group_size)
    target = min(max(1, int(num_groups)), maximum)

    if maximum <= target and maximum <= 5000:
        return list(itertools.combinations(range(n_clients), group_size))

    rng, groups = random.Random(seed), set()
    for _ in range(max(100, target * 25)):
        groups.add(tuple(sorted(rng.sample(range(n_clients), group_size))))
        if len(groups) >= target:
            break

    if len(groups) < target:
        for group in itertools.combinations(range(n_clients), group_size):
            groups.add(group)
            if len(groups) >= target:
                break
    return sorted(groups)


def select_suspicious(records, args):
    if not records:
        return [], 0.0, 0.0

    scores = np.asarray([r["score"] for r in records], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)) + 1e-12)
    threshold = max(
        float(getattr(args, "probe_excess_floor", 0.005)),
        median + float(getattr(args, "probe_threshold_z", 2.0)) * mad,
    )

    candidates = [r for r in records if r["score"] >= threshold and r["score"] > 0]
    candidates.sort(key=lambda r: r["score"], reverse=True)

    fraction = max(0.01, min(1.0, float(getattr(args, "probe_suspicious_fraction", 0.20))))
    limit = max(1, int(math.ceil(len(records) * fraction)))
    return candidates[:limit], threshold, float(scores.max())


def client_key(i, users_idx):
    return str(int(users_idx[i])) if users_idx is not None else "local_{}".format(i)


def pair_key(i, j, users_idx):
    a, b = (int(users_idx[i]), int(users_idx[j])) if users_idx is not None else (int(i), int(j))
    return "{}:{}".format(min(a, b), max(a, b))


def current_risk(groups, suspicious, n_clients, users_idx, args):
    appearances = np.zeros(n_clients)
    pair_appearances = np.zeros((n_clients, n_clients))
    flags = np.zeros(n_clients)
    pair_flags = np.zeros((n_clients, n_clients))

    for group in groups:
        for i in group:
            appearances[i] += 1
        for i, j in itertools.combinations(group, 2):
            pair_appearances[i, j] += 1
            pair_appearances[j, i] += 1

    if suspicious:
        scale = max(float(np.median([r["score"] for r in suspicious])), 1e-12)
        for record in suspicious:
            strength = min(3.0, max(0.25, float(record["score"]) / scale))
            for i in record["group"]:
                flags[i] += strength
            for i, j in itertools.combinations(record["group"], 2):
                pair_flags[i, j] += strength
                pair_flags[j, i] += strength

    individual = flags / np.maximum(appearances, 1.0)
    matrix = pair_flags / np.maximum(pair_appearances, 1.0)
    top_k = max(1, min(int(getattr(args, "probe_pair_topk", 3)), max(1, n_clients - 1)))
    pair_pressure = np.array([
        float(np.mean(np.sort(matrix[i])[::-1][:top_k]))
        for i in range(n_clients)
    ])

    pair_updates = {}
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            if matrix[i, j] > 0:
                pair_updates[pair_key(i, j, users_idx)] = float(matrix[i, j])
    return individual, pair_pressure, pair_updates


def persistent_risk(individual, pair_updates, users_idx, n_clients, args, per_run, first_call, update_memory):
    key = (str(getattr(args, "dataset", "")), str(getattr(args, "iid", "")), int(per_run))

    if first_call:
        _CLIENT_MEMORY[key], _PAIR_MEMORY[key] = {}, {}

    clients = _CLIENT_MEMORY.setdefault(key, {})
    pairs = _PAIR_MEMORY.setdefault(key, {})
    client_decay = float(getattr(args, "probe_risk_decay", 0.85))
    pair_decay = float(getattr(args, "probe_pair_decay", 0.90))
    cap = float(getattr(args, "probe_max_risk", 3.0))

    for table, decay in ((clients, client_decay), (pairs, pair_decay)):
        for k in list(table):
            table[k] *= decay
            if table[k] < 1e-8:
                del table[k]

    if update_memory:
        for i in range(n_clients):
            k = client_key(i, users_idx)
            clients[k] = min(cap, float(clients.get(k, 0.0)) + float(individual[i]))
        for k, value in pair_updates.items():
            pairs[k] = min(cap, float(pairs.get(k, 0.0)) + float(value))

    individual_mem = np.array([
        float(clients.get(client_key(i, users_idx), 0.0))
        for i in range(n_clients)
    ])

    matrix = np.zeros((n_clients, n_clients))
    real_to_local = (
        {int(real): i for i, real in enumerate(users_idx)}
        if users_idx is not None else None
    )

    for k, value in pairs.items():
        try:
            a, b = [int(x) for x in k.split(":", 1)]
        except Exception:
            continue

        if real_to_local is not None:
            if a not in real_to_local or b not in real_to_local:
                continue
            i, j = real_to_local[a], real_to_local[b]
        else:
            i, j = a, b
            if not (0 <= i < n_clients and 0 <= j < n_clients):
                continue

        matrix[i, j] = matrix[j, i] = max(matrix[i, j], float(value))

    top_k = max(1, min(int(getattr(args, "probe_pair_topk", 3)), max(1, n_clients - 1)))
    pair_mem = np.array([
        float(np.mean(np.sort(matrix[i])[::-1][:top_k]))
        for i in range(n_clients)
    ])

    mix = max(0.0, min(1.0, float(getattr(args, "probe_persistent_pair_mix", 0.70))))
    combined = (1.0 - mix) * individual_mem + mix * pair_mem
    return np.maximum(combined, pair_mem)


def ProbeGroupDefense(
    w_list,
    w_updates,
    global_model,
    dataset_test,
    args,
    per_run,
    first_call,
    w_length,
    users_idx=None,
    idx_attacker=None,
    debug=False,
):
    del w_list
    n_clients = len(w_updates)
    if n_clients == 0:
        return global_model.state_dict()

    device = getattr(args, "device", next(global_model.parameters()).device)
    global_state = global_model.state_dict()
    base_weights = normalize_weights(w_length)

    full_update = aggregate_updates(w_updates, range(n_clients), global_state, base_weights)
    full_state = add_update(global_state, full_update)

    full_model = copy.deepcopy(global_model).to(device)
    full_model.load_state_dict(full_state)
    full_model.eval()

    group_model = copy.deepcopy(global_model).to(device)
    group_model.eval()

    clean_batches = get_probe_batches(dataset_test, args, device)
    trigger_batches = [apply_probe_trigger(x, args) for x, _ in clean_batches]
    target = int(getattr(args, "probe_target", 1))
    full_clean, full_trigger = model_probe_stats(
        full_model, clean_batches, trigger_batches, target
    )

    default_group_size = 5 if n_clients >= 12 else max(2, min(4, n_clients // 2))
    group_size = int(getattr(args, "probe_group_size", 0) or default_group_size)
    num_groups = int(
        getattr(args, "probe_num_groups", 0)
        or min(180, max(32, n_clients * 5))
    )

    groups = sample_groups(
        n_clients,
        group_size,
        num_groups,
        deterministic_seed(args, users_idx, n_clients),
    )

    records = []
    for group in groups:
        weights = normalize_weights([w_length[i] for i in group])
        update = aggregate_updates(w_updates, group, global_state, weights)
        group_model.load_state_dict(add_update(global_state, update))
        group_model.eval()

        clean_prob, trigger_prob = model_probe_stats(
            group_model, clean_batches, trigger_batches, target
        )
        clean_gap = abs(clean_prob - full_clean)
        trigger_gap = abs(trigger_prob - full_trigger)

        records.append({
            "group": tuple(group),
            "score": float(max(0.0, trigger_gap - clean_gap)),
        })

    suspicious, threshold, max_score = select_suspicious(records, args)
    evidence = (
        len(suspicious) >= int(getattr(args, "probe_attack_gate_min_groups", 2))
        and max_score >= threshold
    )

    individual, pair_pressure, pair_updates = current_risk(
        groups,
        suspicious if evidence else [],
        n_clients,
        users_idx,
        args,
    )

    risk = persistent_risk(
        individual,
        pair_updates,
        users_idx,
        n_clients,
        args,
        per_run,
        first_call,
        evidence,
    )

    tau = float(getattr(args, "probe_tau", 3.0))
    grace = float(getattr(args, "probe_risk_grace", 0.02))
    weights = np.asarray(base_weights) * np.exp(-tau * np.maximum(0.0, risk - grace))

    dropped = []
    if bool(getattr(args, "probe_hard_drop", False)):
        threshold_drop = float(getattr(args, "probe_hard_drop_threshold", 0.75))
        candidates = [
            int(i) for i in np.argsort(risk)[::-1]
            if risk[int(i)] >= threshold_drop
        ][:max(0, n_clients - 1)]

        for i in candidates:
            weights[i] = 0.0
            dropped.append(i)

    if weights.sum() <= 0:
        weights = np.asarray(base_weights)

    weights = (weights / weights.sum()).tolist()
    aggregation_updates = (
        median_clip_updates(w_updates)
        if bool(getattr(args, "probe_use_median_clip", True))
        else w_updates
    )
    defended_update = aggregate_updates(
        aggregation_updates,
        range(n_clients),
        global_state,
        weights,
    )

    cache_round_metrics(
        args,
        per_run,
        users_idx,
        idx_attacker,
        records,
        suspicious,
        evidence,
        threshold,
        max_score,
        risk,
        individual,
        pair_pressure,
        dropped,
    )

    if debug and bool(getattr(args, "probe_debug", False)):
        print("ProbeGroup:", {
            "groups": len(records),
            "suspicious": len(suspicious),
            "evidence": evidence,
            "threshold": round(threshold, 6),
            "max_score": round(max_score, 6),
            "dropped": dropped,
        })

    return add_update(global_state, defended_update)


FIELDS = [
    "round", "asr", "clean_acc", "train_acc", "test_loss", "train_loss",
    "avg_train_loss", "groups", "suspicious_groups", "attack_evidence",
    "score_threshold", "max_group_score", "mean_individual_risk",
    "mean_pair_pressure", "max_persistent_risk", "top_client_real",
    "top_client_is_attacker", "hard_drop_real", "hard_drop_count",
    "hard_drop_precision", "hard_drop_recall",
]


def real_id(i, users_idx):
    return int(users_idx[i]) if users_idx is not None else int(i)


def cache_round_metrics(
    args,
    per_run,
    users_idx,
    idx_attacker,
    records,
    suspicious,
    evidence,
    threshold,
    max_score,
    risk,
    individual,
    pair_pressure,
    dropped,
):
    round_idx = int(getattr(args, "current_round", 0))
    attackers = set(int(x) for x in (idx_attacker or []))
    top_local = int(np.argmax(risk))
    top_real = real_id(top_local, users_idx)
    dropped_real = [real_id(i, users_idx) for i in dropped]
    true_positive = sum(rid in attackers for rid in dropped_real)

    _ROUND_CACHE[(int(per_run), round_idx)] = {
        "round": round_idx,
        "groups": len(records),
        "suspicious_groups": len(suspicious),
        "attack_evidence": int(bool(evidence)),
        "score_threshold": threshold,
        "max_group_score": max_score,
        "mean_individual_risk": float(np.mean(individual)),
        "mean_pair_pressure": float(np.mean(pair_pressure)),
        "max_persistent_risk": float(np.max(risk)),
        "top_client_real": top_real,
        "top_client_is_attacker": int(top_real in attackers) if attackers else "",
        "hard_drop_real": str(dropped_real),
        "hard_drop_count": len(dropped_real),
        "hard_drop_precision": true_positive / len(dropped_real) if dropped_real else "",
        "hard_drop_recall": true_positive / len(attackers) if attackers else "",
    }


def summary_path(args, per_run):
    out_dir = os.path.join(
        ".",
        str(getattr(args, "save", "save")),
        str(getattr(args, "dataset", "dataset")),
        str(getattr(args, "iid", "partition")),
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "ProbeGroup_round_metrics_run_{}.csv".format(per_run))


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def update_probe_round_summary_metrics(
    args,
    per_run,
    round_idx,
    asr=None,
    clean_acc=None,
    train_acc=None,
    test_loss=None,
    train_loss=None,
    avg_train_loss=None,
):
    row = _ROUND_CACHE.pop(
        (int(per_run), int(round_idx)),
        {"round": int(round_idx)},
    )
    row.update({
        "asr": scalar(asr),
        "clean_acc": scalar(clean_acc),
        "train_acc": scalar(train_acc),
        "test_loss": scalar(test_loss),
        "train_loss": scalar(train_loss),
        "avg_train_loss": scalar(avg_train_loss),
    })

    path = summary_path(args, per_run)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: scalar(row.get(field, "")) for field in FIELDS})

    return path


__all__ = [
    "ProbeGroupDefense",
    "apply_probe_trigger",
    "median_clip_updates",
    "update_probe_round_summary_metrics",
]
