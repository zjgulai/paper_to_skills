"""Smoke test for thompson_sampling."""
from .thompson_sampling_mab import BernoulliThompsonSampling


def test_bernoulli_thompson_runs():
    bandit = BernoulliThompsonSampling(n_arms=3)
    for _ in range(50):
        arm = bandit.select_arm()
        reward = 1 if arm == 0 else 0
        bandit.update(arm, reward)

    assert hasattr(bandit, 'alpha') or hasattr(bandit, 'successes')


if __name__ == "__main__":
    test_bernoulli_thompson_runs()
    print("OK")
