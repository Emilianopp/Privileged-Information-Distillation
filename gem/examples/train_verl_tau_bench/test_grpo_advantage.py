"""
Standalone verification of GRPO advantage computation with KL.

Tests the exact same logic from Phase 4 of run_agent_env_loop()
against hand-computed expected values.
"""
import numpy as np
from collections import defaultdict

def compute_grpo_advantages(
    traj_episode_ids, traj_task_ids, traj_turn_indices,
    traj_turn_rewards, kl_per_turn, distill_coef, gamma
):
    """Exact copy of Phase 4 logic from train_tau_bench.py"""
    N = len(traj_episode_ids)

    ep_turn_raw = defaultdict(dict)
    ep_turn_kl = defaultdict(dict)
    ep_task = {}

    for i in range(N):
        ep_id = traj_episode_ids[i]
        turn_idx = traj_turn_indices[i]
        ep_turn_raw[ep_id][turn_idx] = traj_turn_rewards[i]
        if kl_per_turn is not None:
            ep_turn_kl[ep_id][turn_idx] = kl_per_turn[i]
        ep_task[ep_id] = traj_task_ids[i]

    ep_return = {}
    for ep_id, turn_rewards in ep_turn_raw.items():
        turns_desc = sorted(turn_rewards.keys(), reverse=True)
        G = 0.0
        for t in turns_desc:
            G = turn_rewards[t] + gamma * G

        kl_total = 0.0
        if ep_id in ep_turn_kl:
            kl_total = sum(ep_turn_kl[ep_id].values())

        ep_return[ep_id] = G - distill_coef * kl_total

    task_ep_returns = defaultdict(list)
    for ep_id, ret in ep_return.items():
        task_ep_returns[ep_task[ep_id]].append((ep_id, ret))

    adv_by_episode = {}
    for task_idx, ep_rets in task_ep_returns.items():
        returns = [r for _, r in ep_rets]
        mean_ret = np.mean(returns)
        for ep_id, ret in ep_rets:
            adv_by_episode[ep_id] = ret - mean_ret

    final_adv = [adv_by_episode[traj_episode_ids[i]] for i in range(N)]
    return final_adv, ep_return, adv_by_episode


def test_baseline_no_kl():
    """
    2 tasks, 2 repeats each. gamma=1.0 for easy math.

    Task 0:
      ep0: turns [0,1,2], rewards [0, 0, 1]  → G = 0 + 0 + 1 = 1.0
      ep1: turns [0,1],   rewards [0, 0]      → G = 0 + 0 = 0.0
    Task 1:
      ep2: turns [0,1,2], rewards [0, 0, 0.5] → G = 0.5
      ep3: turns [0,1,2], rewards [0, 0, 1]   → G = 1.0

    GRPO for task 0: mean=0.5, adv_ep0=0.5, adv_ep1=-0.5
    GRPO for task 1: mean=0.75, adv_ep2=-0.25, adv_ep3=0.25
    """
    #                    ep0        ep0        ep0        ep1       ep1       ep2        ep2        ep2        ep3        ep3        ep3
    traj_episode_ids  = [0,         0,         0,         1,        1,        2,         2,         2,         3,         3,         3]
    traj_task_ids     = [0,         0,         0,         0,        0,        1,         1,         1,         1,         1,         1]
    traj_turn_indices = [0,         1,         2,         0,        1,        0,         1,         2,         0,         1,         2]
    traj_turn_rewards = [0.0,       0.0,       1.0,       0.0,      0.0,      0.0,       0.0,       0.5,       0.0,       0.0,       1.0]

    adv, ep_ret, adv_ep = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn=None, distill_coef=0.0, gamma=1.0
    )

    # Check episode returns
    assert abs(ep_ret[0] - 1.0) < 1e-9, f"ep0 return: {ep_ret[0]} != 1.0"
    assert abs(ep_ret[1] - 0.0) < 1e-9, f"ep1 return: {ep_ret[1]} != 0.0"
    assert abs(ep_ret[2] - 0.5) < 1e-9, f"ep2 return: {ep_ret[2]} != 0.5"
    assert abs(ep_ret[3] - 1.0) < 1e-9, f"ep3 return: {ep_ret[3]} != 1.0"

    # Check advantages
    assert abs(adv_ep[0] - 0.5) < 1e-9,   f"adv ep0: {adv_ep[0]} != 0.5"
    assert abs(adv_ep[1] - (-0.5)) < 1e-9, f"adv ep1: {adv_ep[1]} != -0.5"
    assert abs(adv_ep[2] - (-0.25)) < 1e-9, f"adv ep2: {adv_ep[2]} != -0.25"
    assert abs(adv_ep[3] - 0.25) < 1e-9,   f"adv ep3: {adv_ep[3]} != 0.25"

    # All turns in same episode get same advantage
    assert adv[0] == adv[1] == adv[2], "ep0 turns should share advantage"
    assert adv[3] == adv[4], "ep1 turns should share advantage"

    # Per-task advantages sum to ~0
    task0_advs = [adv_ep[0], adv_ep[1]]
    task1_advs = [adv_ep[2], adv_ep[3]]
    assert abs(sum(task0_advs)) < 1e-9, f"task0 advs don't sum to 0: {task0_advs}"
    assert abs(sum(task1_advs)) < 1e-9, f"task1 advs don't sum to 0: {task1_advs}"

    print("PASS test_baseline_no_kl")


def test_with_kl_penalty():
    """
    Same setup as above but with KL and distill_coef=0.1, gamma=1.0.

    Task 0:
      ep0: turns [0,1,2], rewards [0, 0, 1], kl=[0.5, 0.3, 0.2]
        G = 1.0, kl_total = 1.0, R = 1.0 - 0.1*1.0 = 0.9
      ep1: turns [0,1],   rewards [0, 0],    kl=[0.1, 0.4]
        G = 0.0, kl_total = 0.5, R = 0.0 - 0.1*0.5 = -0.05

    mean_task0 = (0.9 + (-0.05)) / 2 = 0.425
    adv_ep0 = 0.9 - 0.425 = 0.475
    adv_ep1 = -0.05 - 0.425 = -0.475
    """
    traj_episode_ids  = [0,    0,    0,    1,    1]
    traj_task_ids     = [0,    0,    0,    0,    0]
    traj_turn_indices = [0,    1,    2,    0,    1]
    traj_turn_rewards = [0.0,  0.0,  1.0,  0.0,  0.0]
    kl_per_turn       = [0.5,  0.3,  0.2,  0.1,  0.4]

    adv, ep_ret, adv_ep = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn, distill_coef=0.1, gamma=1.0
    )

    assert abs(ep_ret[0] - 0.9) < 1e-9,   f"ep0 R: {ep_ret[0]} != 0.9"
    assert abs(ep_ret[1] - (-0.05)) < 1e-9, f"ep1 R: {ep_ret[1]} != -0.05"
    assert abs(adv_ep[0] - 0.475) < 1e-9,  f"adv ep0: {adv_ep[0]} != 0.475"
    assert abs(adv_ep[1] - (-0.475)) < 1e-9, f"adv ep1: {adv_ep[1]} != -0.475"

    # All turns in ep0 share the same advantage
    assert adv[0] == adv[1] == adv[2]

    print("PASS test_with_kl_penalty")


def test_gamma_discounting():
    """
    Verify gamma discounting on raw rewards (NOT on KL).
    gamma=0.9, distill_coef=0.5

    ep0: turns [0,1,2], rewards [0, 0.5, 1.0], kl=[0.1, 0.2, 0.3]
      G = 0 + 0.9*(0.5 + 0.9*1.0) = 0 + 0.9*1.4 = 1.26
      kl_total = 0.1 + 0.2 + 0.3 = 0.6  (NO discounting on KL)
      R = 1.26 - 0.5*0.6 = 1.26 - 0.3 = 0.96

    ep1: turns [0,1,2], rewards [0, 0, 0.5], kl=[0.4, 0.1, 0.1]
      G = 0 + 0.9*(0 + 0.9*0.5) = 0 + 0.9*0.45 = 0.405
      kl_total = 0.4 + 0.1 + 0.1 = 0.6
      R = 0.405 - 0.5*0.6 = 0.405 - 0.3 = 0.105
    """
    traj_episode_ids  = [0,    0,    0,    1,    1,    1]
    traj_task_ids     = [0,    0,    0,    0,    0,    0]
    traj_turn_indices = [0,    1,    2,    0,    1,    2]
    traj_turn_rewards = [0.0,  0.5,  1.0,  0.0,  0.0,  0.5]
    kl_per_turn       = [0.1,  0.2,  0.3,  0.4,  0.1,  0.1]

    adv, ep_ret, adv_ep = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn, distill_coef=0.5, gamma=0.9
    )

    assert abs(ep_ret[0] - 0.96) < 1e-9,  f"ep0 R: {ep_ret[0]} != 0.96"
    assert abs(ep_ret[1] - 0.105) < 1e-9, f"ep1 R: {ep_ret[1]} != 0.105"

    mean_ret = (0.96 + 0.105) / 2  # 0.5325
    assert abs(adv_ep[0] - (0.96 - mean_ret)) < 1e-9
    assert abs(adv_ep[1] - (0.105 - mean_ret)) < 1e-9

    print("PASS test_gamma_discounting")


def test_kl_zero_coef_matches_baseline():
    """When distill_coef=0, KL values should have no effect."""
    traj_episode_ids  = [0,   0,   1,   1]
    traj_task_ids     = [0,   0,   0,   0]
    traj_turn_indices = [0,   1,   0,   1]
    traj_turn_rewards = [0.0, 1.0, 0.0, 0.5]
    kl_per_turn       = [99., 99., 99., 99.]  # huge KL, but coef=0

    adv_with_kl, ep_ret_kl, _ = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn, distill_coef=0.0, gamma=1.0
    )
    adv_no_kl, ep_ret_no, _ = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn=None, distill_coef=0.0, gamma=1.0
    )

    for ep_id in ep_ret_kl:
        assert abs(ep_ret_kl[ep_id] - ep_ret_no[ep_id]) < 1e-9, \
            f"ep{ep_id}: {ep_ret_kl[ep_id]} != {ep_ret_no[ep_id]}"

    for a, b in zip(adv_with_kl, adv_no_kl):
        assert abs(a - b) < 1e-9

    print("PASS test_kl_zero_coef_matches_baseline")


def test_single_episode_per_task():
    """
    Edge case: only 1 episode per task (n_repeat=1).
    Advantage should be 0 since mean = the only return.
    """
    traj_episode_ids  = [0,   0,   1,   1]
    traj_task_ids     = [0,   0,   1,   1]  # different tasks!
    traj_turn_indices = [0,   1,   0,   1]
    traj_turn_rewards = [0.0, 1.0, 0.0, 0.5]

    adv, _, adv_ep = compute_grpo_advantages(
        traj_episode_ids, traj_task_ids, traj_turn_indices,
        traj_turn_rewards, kl_per_turn=None, distill_coef=0.0, gamma=1.0
    )

    assert abs(adv_ep[0]) < 1e-9, f"single-ep task should have 0 advantage, got {adv_ep[0]}"
    assert abs(adv_ep[1]) < 1e-9, f"single-ep task should have 0 advantage, got {adv_ep[1]}"

    print("PASS test_single_episode_per_task")


if __name__ == "__main__":
    test_baseline_no_kl()
    test_with_kl_penalty()
    test_gamma_discounting()
    test_kl_zero_coef_matches_baseline()
    test_single_episode_per_task()
    print("\n✓ All 5 tests passed!")
