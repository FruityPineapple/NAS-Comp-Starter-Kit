"""Regression tests for diverse quotas and decoupled recipe selection."""

import os
import random
import sys
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from nas import NAS
from search_space import ArchSpec


RECIPES = [
    {"name": "stable"},
    {"name": "regularized"},
    {"name": "fast_fit"},
]


class _ConstantClock:
    def __init__(self, remaining):
        self.remaining = float(remaining)

    def check(self):
        return self.remaining


class _TinyRecipeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, data):
        return torch.zeros((data.size(0), 2), device=data.device)


class _TinyRecipeSpace:
    def __init__(self):
        self.models = []

    def build_model(self, spec):
        model = _TinyRecipeModel()
        self.models.append(model)
        return model


class _TinyArchitectureModel(_TinyRecipeModel):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec


class _TinyArchitectureSpace:
    max_safe_stages = 4

    def __init__(self, current_seed, initializations):
        self.current_seed = current_seed
        self.initializations = initializations

    def build_model(self, spec):
        self.initializations[spec] = self.current_seed()
        return _TinyArchitectureModel(spec)


def _run_recipe_race(
    schedule,
    incumbent_accuracy=0.84,
    incumbent_loss=0.70,
    samples=1000,
    remaining=1000.0,
    seconds_per_step=0.001,
    loader_steps=20,
):
    controller = object.__new__(NAS)
    controller.clock = _ConstantClock(remaining)
    controller.seed = 17
    controller.device = torch.device("cpu")
    controller.recipe_min_accuracy_gain = 0.001
    recipes = [dict(recipe) for recipe in RECIPES]
    controller._training_recipes = MethodType(
        lambda self: recipes, controller
    )
    controller._default_recipe = MethodType(
        lambda self: recipes[0], controller
    )

    stage_counts = {recipe["name"]: 0 for recipe in recipes}
    train_calls = []
    seed_calls = []
    evaluated_sources = []
    marker_metrics = {}
    for stages in schedule.values():
        for marker, accuracy, loss in stages:
            marker_metrics[float(marker)] = (float(accuracy), float(loss))

    def fake_train(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
        start_step=0,
    ):
        name = recipe["name"]
        stage = stage_counts[name]
        stage_counts[name] += 1
        marker, _, _ = schedule[name][stage]
        train_calls.append(
            {
                "name": name,
                "stage": stage + 1,
                "start_marker": float(model.marker.item()),
                "max_steps": int(max_steps),
                "start_step": int(start_step),
                "optimizer_marker": (
                    None
                    if optimizer_state is None
                    else optimizer_state["param_groups"][0]["marker"]
                ),
            }
        )
        with torch.no_grad():
            model.marker.fill_(float(marker))
        state = {
            "state": {},
            "param_groups": [{"marker": float(marker)}],
        }
        accuracy = marker_metrics[float(marker)][0]
        stats = {
            "examples_seen": int(max_steps),
            "train_accuracy": min(1.0, float(accuracy) + 0.05),
            "train_loss": 1.0 - float(accuracy),
            "final_lr": 1e-3,
            "peak_memory_mb": 0.0,
        }
        return int(max_steps), 0.1, False, state, stats

    def fake_evaluate(self, model, source=None):
        evaluated_sources.append(source)
        marker = float(model.marker.item())
        if abs(marker - 1.0) < 1e-6:
            accuracy, loss = incumbent_accuracy, incumbent_loss
        else:
            accuracy, loss = marker_metrics[marker]
        return {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(accuracy),
            "loss": float(loss),
            "samples": int(samples),
        }

    controller._train_low_fidelity = MethodType(fake_train, controller)
    controller._evaluate = MethodType(fake_evaluate, controller)
    controller._seed_everything = seed_calls.append

    architecture_model = _TinyRecipeModel()
    with torch.no_grad():
        architecture_model.marker.fill_(1.0)
    spec = ArchSpec(2, 16, 1, "basic", 3, False, 3, "spatial")
    winner = {
        "model": architecture_model,
        "spec": spec,
        "params": 1,
        "val_acc": incumbent_accuracy,
        "seconds_per_step": seconds_per_step,
        "trained_steps": 10,
        "train_seconds": 1.0,
        "optimizer_state": {
            "state": {},
            "param_groups": [{"marker": 1.0}],
        },
    }
    space = _TinyRecipeSpace()
    validation_source = [None]
    result = controller._recipe_tournament(
        space,
        winner,
        [None] * loader_steps,
        validation_source,
        0.0,
    )
    return result, train_calls, space, {
        "seed_calls": seed_calls,
        "evaluated_sources": evaluated_sources,
        "validation_source": validation_source,
    }


def _ranked_entries(families, per_family=6):
    entries = []
    for family in reversed(families):
        for index in range(per_family):
            spec = ArchSpec(
                2,
                16,
                1 + index % 3,
                "basic",
                3 if index < 3 else 5,
                bool(index % 2),
                3,
                family,
            )
            entries.append(
                {
                    "spec": spec,
                    "recipe": RECIPES[index % len(RECIPES)],
                    "params": 20_000 * (index + 1),
                    "proxy_prior": 1.0 - len(entries) / 100.0,
                }
            )
    return entries


def test_label_aware_slots_are_family_balanced():
    controller = object.__new__(NAS)
    natural_families = ["spatial", "spatial_pyramid", "factorized"]
    selected = controller._select_label_aware_entries(
        _ranked_entries(natural_families), 12
    )
    counts = {
        family: sum(
            entry["spec"].model_family == family for entry in selected
        )
        for family in natural_families
    }
    assert set(counts.values()) == {4}
    for family in natural_families:
        family_params = sorted(
            entry["params"]
            for entry in selected
            if entry["spec"].model_family == family
        )
        assert family_params[0] < family_params[-1]

    structured_families = natural_families + [
        "axis_width",
        "axis_height",
        "dual_axis",
    ]
    selected = controller._select_label_aware_entries(
        _ranked_entries(structured_families), 12
    )
    counts = {
        family: sum(
            entry["spec"].model_family == family for entry in selected
        )
        for family in structured_families
    }
    assert set(counts.values()) == {2}


def test_label_aware_slots_reserve_feasible_anchor_endpoints():
    controller = object.__new__(NAS)
    controller.input_h = 27
    controller.input_w = 18
    space = SimpleNamespace(max_safe_stages=4)
    compact, central, capacity = controller._anchor_specs("spatial", space)
    smaller_decoy = ArchSpec(
        2, 16, 1, "basic", 3, False, 3, "spatial"
    )
    larger_decoy = ArchSpec(
        4, 64, 3, "basic", 5, True, 7, "spatial"
    )
    entries = [
        {"spec": smaller_decoy, "params": 1_000},
        {"spec": larger_decoy, "params": 2_000_000},
        {"spec": central, "params": 90_000},
        {"spec": compact, "params": 20_000},
        {"spec": capacity, "params": 500_000},
    ]
    selected = controller._select_label_aware_entries(
        entries,
        2,
        anchor_specs_by_family={
            "spatial": [compact, central, capacity]
        },
    )
    assert {entry["spec"] for entry in selected} == {compact, capacity}


def _run_ordered_macro_seed_trial(specs):
    controller = object.__new__(NAS)
    controller.seed = 17
    controller.input_h = 16
    controller.input_w = 16
    controller.num_classes = 2
    controller.data_props = {}
    controller.metadata = {}
    controller.device = torch.device("cpu")
    controller.clock = _ConstantClock(100.0)
    controller.train_loader = [None] * 4
    current_seed = {"value": None}
    initializations = {}
    training_seeds = {}

    def record_seed(seed):
        current_seed["value"] = int(seed)

    def fake_train(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
        start_step=0,
    ):
        training_seeds[model.spec] = current_seed["value"]
        accuracy = 0.60 if model.spec.model_family == "spatial" else 0.50
        return (
            int(max_steps),
            0.1,
            False,
            {"state": {}, "param_groups": [{"lr": 1e-3}]},
            {
                "examples_seen": int(max_steps),
                "train_accuracy": accuracy,
                "final_lr": 1e-3,
                "peak_memory_mb": 0.0,
            },
        )

    def fake_evaluate(self, model, source=None):
        accuracy = 0.60 if model.spec.model_family == "spatial" else 0.50
        return {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "loss": 0.70,
            "margin": 0.20,
            "samples": 100,
        }

    controller._seed_everything = record_seed
    controller._train_low_fidelity = MethodType(fake_train, controller)
    controller._evaluate = MethodType(fake_evaluate, controller)
    controller._make_refinement_loader = MethodType(
        lambda self: None, controller
    )
    controller._recipe_tournament = MethodType(
        lambda self, space, winner, *args, **kwargs: winner,
        controller,
    )
    space = _TinyArchitectureSpace(
        lambda: current_seed["value"], initializations
    )
    ranked = [
        {"spec": spec, "params": 10_000, "proxy_evaluated": True}
        for spec in specs
    ]
    controller._successive_halving(
        space,
        ranked,
        n_top=2,
        cached_batches=[None] * 4,
        validation_batches=[None],
        round_plan=((1.0, 1),),
        deadline_remaining=0.0,
    )
    return initializations, training_seeds


def test_macro_seed_schedule_is_candidate_order_independent():
    spatial = ArchSpec(3, 32, 2, "basic", 3, True, 3, "spatial")
    factorized = ArchSpec(
        3, 32, 2, "basic", 3, True, 3, "factorized"
    )
    forward = _run_ordered_macro_seed_trial([spatial, factorized])
    reverse = _run_ordered_macro_seed_trial([factorized, spatial])
    assert forward == reverse
    assert set(forward[0].values()) == {517}
    assert set(forward[1].values()) == {1017}


def test_capacity_anchor_gets_bounded_late_repechage():
    controller = object.__new__(NAS)
    normal_best = {
        "utility": 0.42,
        "val_acc": 0.42,
        "val_samples": 1000,
    }
    normal_cutoff = {
        "utility": 0.40,
        "val_acc": 0.40,
        "val_samples": 1000,
    }
    capacity = {
        "utility": 0.38,
        "val_acc": 0.38,
        "val_samples": 1000,
        "val_history": [0.30, 0.38],
        "anchor_kind": "capacity",
    }
    results = [normal_best, normal_cutoff, capacity]
    early, insured = controller._select_halving_survivors(results, 2)
    assert insured is capacity
    assert len(early) == 2
    assert capacity in early
    late, insured = controller._select_halving_survivors(
        results, 2, final_round=True
    )
    assert insured is capacity
    assert late == [normal_best, normal_cutoff, capacity]

    far_behind = dict(capacity)
    far_behind.update(
        {
            "utility": 0.20,
            "val_acc": 0.20,
            "val_history": [0.19, 0.20],
        }
    )
    bounded, insured = controller._select_halving_survivors(
        [normal_best, normal_cutoff, far_behind], 2, final_round=True
    )
    assert insured is None
    assert bounded == [normal_best, normal_cutoff]


def test_insured_anchor_is_parked_and_handed_off_as_one_challenger():
    controller = object.__new__(NAS)
    controller.seed = 17
    controller.input_h = 27
    controller.input_w = 18
    controller.num_classes = 6
    controller.data_props = {}
    controller.metadata = {}
    controller.device = torch.device("cpu")
    controller.clock = _ConstantClock(100.0)
    controller.train_loader = [None] * 4
    space_stub = SimpleNamespace(max_safe_stages=4)
    compact, central, capacity = controller._anchor_specs(
        "spatial", space_stub
    )
    calls = {compact: 0, central: 0, capacity: 0}

    class _AnchorSpace(_TinyArchitectureSpace):
        pass

    current_seed = {"value": None}
    initializations = {}
    space = _AnchorSpace(lambda: current_seed["value"], initializations)

    def record_seed(seed):
        current_seed["value"] = int(seed)

    def fake_train(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
        start_step=0,
    ):
        calls[model.spec] += 1
        train_accuracy = 1.0 if model.spec == central else 0.50
        return (
            int(max_steps),
            0.1,
            False,
            {"state": {}, "param_groups": [{"lr": 1e-3}]},
            {
                "examples_seen": int(max_steps),
                "train_accuracy": train_accuracy,
                "final_lr": 1e-3,
                "peak_memory_mb": 0.0,
            },
        )

    def fake_evaluate(self, model, source=None):
        if model.spec == central:
            accuracy, loss = 0.42, 2.20
        elif model.spec == compact:
            accuracy, loss = 0.40, 1.70
        else:
            accuracy = 0.38 if calls[capacity] <= 1 else 0.39
            loss = 1.55
        return {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "loss": loss,
            "margin": 0.20,
            "samples": 1000,
        }

    controller._seed_everything = record_seed
    controller._train_low_fidelity = MethodType(fake_train, controller)
    controller._evaluate = MethodType(fake_evaluate, controller)
    controller._make_refinement_loader = MethodType(
        lambda self: [None] * 4, controller
    )
    controller._recipe_tournament = MethodType(
        lambda self, space, winner, *args, **kwargs: winner,
        controller,
    )
    ranked = [
        {"spec": compact, "params": 7_000_000},
        {"spec": central, "params": 8_000_000},
        {"spec": capacity, "params": 1_000_000},
    ]
    winner = controller._successive_halving(
        space,
        ranked,
        n_top=3,
        cached_batches=[None] * 4,
        validation_batches=["selection"],
        confirmation_batches=["confirmation"],
        round_plan=((1.0, 2),),
        deadline_remaining=0.0,
    )
    bundle = winner["model"].architecture_challenger_bundle
    assert winner["spec"] == central
    assert calls[capacity] == 3
    assert controller.metadata["nas_anchor_insurance_parked"] == 1
    assert bundle["spec"] == capacity
    assert bundle["reason"] == "efficient_overfit_insurance"
    assert not any(
        module is bundle["model"] for module in winner["model"].modules()
    )


def test_architectures_are_not_confounded_with_recipes():
    controller = object.__new__(NAS)
    controller.input_h = 28
    controller.input_w = 28
    controller.rng = random.Random(7)
    families = ["spatial", "spatial_pyramid", "factorized"]
    space = SimpleNamespace(
        model_families=families,
        max_safe_stages=4,
    )

    pool = []
    for family in families:
        for index in range(12):
            spec = ArchSpec(
                3,
                32,
                1 + index % 3,
                "basic" if index < 6 else "bottleneck",
                3 if index % 2 == 0 else 5,
                bool(index % 3 == 0),
                3 if index < 9 else 5,
                family,
            )
            if spec not in {existing for existing, _ in pool}:
                pool.append((spec, 50_000 + index * 1_000))

    entries = controller._portfolio_candidates(
        space, pool, 27, RECIPES
    )
    for family in families:
        names = [
            entry["recipe"]["name"]
            for entry in entries
            if entry["spec"].model_family == family
        ]
        assert names
        assert set(names) == {"architecture_probe"}


def test_promoted_pool_restores_high_capacity_size_strata():
    controller = object.__new__(NAS)
    controller.metadata = {}
    controller.input_h = 27
    controller.input_w = 18
    small = ArchSpec(2, 16, 1, "basic", 3, False, 3, "spatial")
    medium = ArchSpec(2, 32, 2, "basic", 3, False, 3, "spatial")
    wide = ArchSpec(2, 64, 2, "basic", 3, False, 3, "spatial")
    largest = ArchSpec(3, 64, 3, "basic", 3, False, 3, "spatial")
    ranked = [
        {
            "spec": small,
            "params": 20_000,
            "proxy_prior": 1.0,
        },
        {
            "spec": medium,
            "params": 80_000,
            "proxy_prior": 0.8,
        },
    ]
    pool = [
        (small, 20_000),
        (medium, 80_000),
        (wide, 320_000),
        (largest, 1_200_000),
    ]
    expanded = controller._expand_promoted_macro_pool(
        ranked, pool, {"spatial"}
    )
    selected = controller._select_label_aware_entries(expanded, 3)
    assert len(expanded) == 4
    assert any(entry["spec"].init_channels == 64 for entry in selected)
    assert max(entry["params"] for entry in selected) == 1_200_000
    anchors = controller._anchor_specs(
        "spatial", SimpleNamespace(max_safe_stages=4)
    )
    assert any(spec.init_channels == 64 for spec in anchors)


def test_final_score_does_not_let_rank_override_final_accuracy():
    controller = object.__new__(NAS)
    stable = {
        "full_val_acc": 0.82,
        "val_history": [0.60, 0.70, 0.80],
    }
    late_flip = {
        "full_val_acc": 0.825,
        "val_history": [0.40, 0.55, 0.79],
    }
    entries = controller._final_selection_scores([stable, late_flip])
    assert entries[1]["selection_score"] > entries[0]["selection_score"]
    assert (
        entries[0]["history_rank_score"]
        > entries[1]["history_rank_score"]
    )


def test_clear_confirmation_winner_cannot_be_overturned_by_history():
    controller = object.__new__(NAS)
    historical_leader = {
        "selection_val_acc": 0.7214,
        "confirmation_val_acc": 0.7126,
        "confirmation_val_loss": 0.90,
        "confirmation_samples": 1023,
        "val_history": [0.50, 0.65, 0.755, 0.7214],
    }
    holdout_winner = {
        "selection_val_acc": 0.7494,
        "confirmation_val_acc": 0.7468,
        "confirmation_val_loss": 0.80,
        "confirmation_samples": 1023,
        "val_history": [0.40, 0.55, 0.7185, 0.7494],
    }
    ordered, tie_threshold = controller._order_final_results(
        [historical_leader, holdout_winner]
    )
    assert ordered[0] is holdout_winner
    assert tie_threshold is None


def test_rank_history_is_only_a_confirmation_tie_break():
    controller = object.__new__(NAS)
    stable = {
        "selection_val_acc": 0.80,
        "confirmation_val_acc": 0.75,
        "confirmation_val_loss": 0.70,
        "confirmation_samples": 1000,
        "val_history": [0.65, 0.72, 0.80],
    }
    late = {
        "selection_val_acc": 0.80,
        "confirmation_val_acc": 0.75,
        "confirmation_val_loss": 0.70,
        "confirmation_samples": 1000,
        "val_history": [0.45, 0.60, 0.80],
    }
    ordered, tie_threshold = controller._order_final_results(
        [late, stable]
    )
    assert tie_threshold is not None
    assert ordered[0] is stable


def test_late_overfit_risk_breaks_only_a_confirmation_tie():
    controller = object.__new__(NAS)
    controller.num_classes = 6
    overfit = {
        "selection_val_acc": 0.43,
        "selection_val_loss": 2.20,
        "confirmation_val_acc": 0.42,
        "confirmation_val_loss": 2.20,
        "confirmation_samples": 1000,
        "train_stats": {"train_accuracy": 1.0},
        "refinement_history": [0.41, 0.42],
        "val_history": [0.38, 0.41, 0.43],
    }
    stable = {
        "selection_val_acc": 0.42,
        "selection_val_loss": 1.55,
        "confirmation_val_acc": 0.42,
        "confirmation_val_loss": 1.55,
        "confirmation_samples": 1000,
        "train_stats": {"train_accuracy": 0.48},
        "refinement_history": [0.41, 0.42],
        "val_history": [0.35, 0.40, 0.42],
    }
    ordered, tie_threshold = controller._order_final_results(
        [overfit, stable]
    )
    assert tie_threshold is not None
    assert ordered[0] is stable


def test_risky_winner_retains_materially_smaller_challenger():
    controller = object.__new__(NAS)
    controller.num_classes = 6
    winner = {
        "params": 8_000_000,
        "confirmation_val_acc": 0.42,
        "confirmation_val_loss": 2.20,
        "train_stats": {"train_accuracy": 1.0},
        "refinement_history": [0.40, 0.42],
    }
    efficient = {
        "params": 1_000_000,
        "confirmation_val_acc": 0.38,
        "confirmation_val_loss": 1.55,
        "train_stats": {"train_accuracy": 0.50},
        "refinement_history": [0.37, 0.38],
    }
    too_large = {
        "params": 7_000_000,
        "confirmation_val_acc": 0.41,
        "confirmation_val_loss": 1.60,
        "train_stats": {"train_accuracy": 0.50},
        "refinement_history": [0.40, 0.41],
    }
    challenger, reason = controller._select_retained_architecture_challenger(
        winner, [winner, too_large, efficient]
    )
    assert challenger is efficient
    assert reason == "efficient_overfit_insurance"
    not_ready = dict(winner)
    not_ready.pop("refinement_history")
    challenger, reason = controller._select_retained_architecture_challenger(
        not_ready, [not_ready, efficient]
    )
    assert challenger is None
    assert reason is None


def test_refinement_checkpoint_restores_weights_and_optimizer():
    controller = object.__new__(NAS)
    model = _TinyRecipeModel()
    with torch.no_grad():
        model.marker.fill_(3.0)
    candidate = {
        "model": model,
        "optimizer_state": {
            "state": {},
            "param_groups": [{"marker": 3.0}],
        },
        "val_acc": 0.80,
        "balanced_acc": 0.79,
        "val_loss": 0.60,
        "val_margin": 0.30,
        "val_samples": 100,
        "trained_steps": 30,
        "data_cursor": 30,
        "train_stats": {"train_accuracy": 0.90},
    }
    checkpoint = controller._snapshot_refinement_checkpoint(candidate)
    with torch.no_grad():
        model.marker.fill_(7.0)
    candidate.update(
        {
            "optimizer_state": {
                "state": {},
                "param_groups": [{"marker": 7.0}],
            },
            "val_acc": 0.70,
            "val_loss": 0.90,
            "trained_steps": 70,
            "data_cursor": 70,
        }
    )
    assert not controller._is_better_refinement_checkpoint(
        candidate, checkpoint
    )
    controller._restore_refinement_checkpoint(candidate, checkpoint)
    assert abs(model.marker.item() - 3.0) < 1e-12
    assert candidate["val_acc"] == 0.80
    assert candidate["trained_steps"] == 30
    assert (
        candidate["optimizer_state"]["param_groups"][0]["marker"]
        == 3.0
    )


def test_search_plateau_keeps_state_and_reduces_lr():
    controller = object.__new__(NAS)
    entry = {
        "val_history": [0.70, 0.701, 0.7005],
        "optimizer_state": {
            "state": {0: {"step": 10}},
            "param_groups": [{"lr": 1e-3}],
        },
    }
    original_state = entry["optimizer_state"]["state"]
    controller._adapt_search_lr(entry)
    assert entry["optimizer_state"]["state"] is original_state
    assert abs(
        entry["optimizer_state"]["param_groups"][0]["lr"] - 3.5e-4
    ) < 1e-12


def test_recipe_race_preserves_a_stronger_incumbent():
    result, calls, _, observations = _run_recipe_race(
        {
            "stable": [(2, 0.80, 0.75), (5, 0.805, 0.72)],
            "regularized": [(3, 0.79, 0.78), (6, 0.795, 0.74)],
            "fast_fit": [(4, 0.78, 0.80)],
        }
    )
    assert abs(result["val_acc"] - 0.84) < 1e-12
    assert abs(result["model"].marker.item() - 1.0) < 1e-12
    assert (
        result["model"].search_optimizer_state["param_groups"][0]["marker"]
        == 1.0
    )
    stage_one = [call for call in calls if call["stage"] == 1]
    stage_two = [call for call in calls if call["stage"] == 2]
    assert [call["start_marker"] for call in stage_one] == [1.0, 1.0, 1.0]
    assert len(stage_two) == 2
    assert len({call["max_steps"] for call in stage_one}) == 1
    assert len({call["max_steps"] for call in stage_two}) == 1
    assert observations["seed_calls"] == [
        12017,
        12017,
        12017,
        13017,
        13017,
    ]
    assert all(
        source is observations["validation_source"]
        for source in observations["evaluated_sources"]
    )


def test_recipe_race_restores_best_stage_and_matching_optimizer():
    result, calls, space, _ = _run_recipe_race(
        {
            "stable": [(2, 0.860, 0.60), (5, 0.850, 0.55)],
            "regularized": [(3, 0.858, 0.52), (6, 0.870, 0.45)],
            "fast_fit": [(4, 0.800, 0.68)],
        }
    )
    assert result["recipe"]["name"] == "regularized"
    assert abs(result["val_acc"] - 0.870) < 1e-12
    assert abs(result["model"].marker.item() - 6.0) < 1e-12
    assert (
        result["model"].search_optimizer_state["param_groups"][0]["marker"]
        == 6.0
    )
    assert abs(
        result["model"].independent_retry_state["marker"].item() - 1.0
    ) < 1e-12

    stable_model = next(
        model
        for model in space.models
        if model.training_recipe["name"] == "stable"
    )
    assert abs(stable_model.marker.item() - 2.0) < 1e-12
    stage_two = [call for call in calls if call["stage"] == 2]
    assert {call["start_marker"] for call in stage_two} == {2.0, 3.0}
    assert {call["optimizer_marker"] for call in stage_two} == {2.0, 3.0}


def test_recipe_race_rejects_a_gain_inside_sampling_uncertainty():
    result, _, _, _ = _run_recipe_race(
        {
            "stable": [(2, 0.842, 0.68), (5, 0.842, 0.67)],
            "regularized": [(3, 0.841, 0.69), (6, 0.841, 0.68)],
            "fast_fit": [(4, 0.800, 0.75)],
        },
        incumbent_accuracy=0.840,
        samples=1_000,
    )
    assert result["recipe"]["name"] == "stable"
    assert abs(result["model"].marker.item() - 1.0) < 1e-12
    assert abs(result["val_acc"] - 0.840) < 1e-12


def test_recipe_trial_score_uses_validation_loss_slope():
    flat = {
        "best_val_acc": 0.800,
        "metrics_history": [
            {"loss": 1.00},
            {"loss": 0.99},
        ],
    }
    improving = {
        "best_val_acc": 0.799,
        "metrics_history": [
            {"loss": 1.00},
            {"loss": 0.70},
        ],
    }
    NAS._recipe_trial_score(flat)
    NAS._recipe_trial_score(improving)
    assert improving["selection_score"] > flat["selection_score"]


def test_controller_evaluation_reports_cross_entropy_loss():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    metrics = controller._evaluate(torch.nn.Identity(), [(logits, targets)])
    expected = torch.nn.functional.cross_entropy(logits, targets).item()
    assert abs(metrics["loss"] - expected) < 1e-12
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 2


def test_controller_evaluation_places_fresh_model_on_controller_device():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    model = torch.nn.Linear(3, 2)
    observed_devices = []
    placement_calls = []

    class _DeviceRecorder(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def to(self, *args, **kwargs):
            placement_calls.append(str(args[0]))
            return super().to(*args, **kwargs)

        def forward(self, data):
            observed_devices.append(
                (
                    data.device.type,
                    next(self.parameters()).device.type,
                )
            )
            return self.wrapped(data)

    recorded_model = _DeviceRecorder(model)
    controller._evaluate(
        recorded_model,
        [(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))],
    )
    assert placement_calls == ["cpu"]
    assert observed_devices == [("cpu", "cpu")]


def test_controller_evaluation_splits_oom_batches_in_order():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    controller.metadata = {}

    class _LimitedBatchModel(torch.nn.Module):
        def forward(self, data):
            if data.size(0) > 2:
                raise RuntimeError("out of memory")
            return data

    logits = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])
    metrics = controller._evaluate(
        _LimitedBatchModel(), [(logits, targets)]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 5
    assert controller.metadata["nas_evaluation_microbatch_size"] <= 2


def test_recipe_race_short_budget_falls_back_without_training():
    result, calls, _, _ = _run_recipe_race(
        {
            "stable": [(2, 0.90, 0.50)],
            "regularized": [(3, 0.91, 0.45)],
            "fast_fit": [(4, 0.89, 0.55)],
        },
        remaining=20.0,
        seconds_per_step=1.0,
    )
    assert calls == []
    assert abs(result["val_acc"] - 0.84) < 1e-12
    assert abs(result["model"].marker.item() - 1.0) < 1e-12


if __name__ == "__main__":
    test_label_aware_slots_are_family_balanced()
    test_label_aware_slots_reserve_feasible_anchor_endpoints()
    test_macro_seed_schedule_is_candidate_order_independent()
    test_capacity_anchor_gets_bounded_late_repechage()
    test_insured_anchor_is_parked_and_handed_off_as_one_challenger()
    test_architectures_are_not_confounded_with_recipes()
    test_promoted_pool_restores_high_capacity_size_strata()
    test_final_score_does_not_let_rank_override_final_accuracy()
    test_clear_confirmation_winner_cannot_be_overturned_by_history()
    test_rank_history_is_only_a_confirmation_tie_break()
    test_late_overfit_risk_breaks_only_a_confirmation_tie()
    test_risky_winner_retains_materially_smaller_challenger()
    test_refinement_checkpoint_restores_weights_and_optimizer()
    test_search_plateau_keeps_state_and_reduces_lr()
    test_recipe_race_preserves_a_stronger_incumbent()
    test_recipe_race_restores_best_stage_and_matching_optimizer()
    test_recipe_race_rejects_a_gain_inside_sampling_uncertainty()
    test_recipe_trial_score_uses_validation_loss_slope()
    test_controller_evaluation_reports_cross_entropy_loss()
    test_controller_evaluation_places_fresh_model_on_controller_device()
    test_controller_evaluation_splits_oom_batches_in_order()
    test_recipe_race_short_budget_falls_back_without_training()
    print("Controller refinement regression tests passed")
