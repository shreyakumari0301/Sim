"""Main simulation loop: layers 1–3 + latent state update."""

from __future__ import annotations

import copy
from typing import List

from layer1 import apply_layer1
from layer2 import apply_layer2
from layer3 import apply_layer3
from state import WorldState


def run_simulation(
    initial_state: WorldState,
    rule_tables: dict,
    n_timesteps: int = 90,
    verbose: bool = False,
) -> List[WorldState]:
    """
    Run the simulation loop.

    Returns a list of WorldState snapshots.
    state_history[0] = initial state BEFORE any updates.
    state_history[t+1] = state AFTER timestep t completes.
    """
    history: List[WorldState] = [initial_state]
    state = initial_state

    for t in range(n_timesteps):
        meta = state.meta.model_copy(
            update={
                "t": t,
                "trial_day": t,
            }
        )
        state = state.copy_updated(meta=meta)

        state = apply_layer1(state, rule_tables)
        state = apply_layer2(state, rule_tables)
        state = _update_latent(state, rule_tables)
        state = apply_layer3(state, rule_tables)

        history.append(copy.deepcopy(state))

        if verbose:
            print(
                f"t={t:3d} | conc={state.drug.plasma_conc:.1f} "
                f"| response={state.effects.clinical_response:.2f} "
                f"| tol={state.tolerance.tolerance_level:.3f} "
                f"| ae={state.toxicity.ae_severity} "
                f"| dose={state.treatment.dose_level:.0f} "
                f"| drug={'ON' if state.treatment.drug_active else 'OFF'}"
            )

    return history


def _update_latent(state: WorldState, rule_tables: dict) -> WorldState:
    """
    Cross-layer latent state update (between L2 and L3).
    Updates resistance_flag based on tolerance + pathway state.
    """
    tol = state.tolerance.model_copy()

    if (
        state.tolerance.tolerance_level > 0.7
        and state.biomarkers.pathway_activity > 0.5
        and not tol.resistance_flag
    ):
        tol.resistance_flag = True

    return state.copy_updated(tolerance=tol)
