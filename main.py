import argparse
import random
import time
import numpy as np

from environment.utils import compare_results, compare_scores
from environment.rugby_env import RugbyEnv

from agents.random_agent import RandomAgent
from agents.dummy_greedy_agent import DummyGreedyAgent
from agents.greedy_agent import GreedyAgent
from agents.role_agent import RoleAgent

ROLES = 7
(
    BALL_CARRIER,
    RIGHT_SUPPORTER,
    LEFT_SUPPORTER,
    RIGHT_SUB_SUPPORTER,
    LEFT_SUB_SUPPORTER,
    RIGH_WING,
    LEFT_WING,
) = range(ROLES)


def run_multi_agent(environment, agents, n_episodes):
    steps_results = np.zeros(n_episodes)
    tackles_results = np.zeros(n_episodes)
    passes_results = np.zeros(n_episodes)
    scores_results = np.zeros((n_episodes,), dtype=np.dtype("i4, i4"))

    for episode in range(n_episodes):
        steps = 0
        score = (0, 0)
        n_tackles = 0
        n_passes = 0
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        while not all(terminals):
            steps += 1
            for observations, agent in zip(observations, agents):
                agent.see(observations)
            actions = [agent.action() for agent in agents]
            next_observations, rewards, terminals, info = environment.step(actions)
            score = info["score"]
            n_tackles = info["n_tackles"]
            n_passes = info["n_passes"]

            if opt.render_ui:
                environment.render()
                
            time.sleep(opt.render_sleep_time)

            observations = next_observations

        steps_results[episode] = steps
        tackles_results[episode] = n_tackles
        scores_results[episode] = score
        passes_results[episode] = n_passes

        environment.close()

    return steps_results, tackles_results, scores_results, passes_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n_agents", type=int, default=7)
    parser.add_argument("--n_opponents", type=int, default=7)
    parser.add_argument("--render-sleep-time", type=float, default=0)
    parser.add_argument("--render-ui", type=bool, default=False)
    parser.add_argument("--max-steps", type=int, default=1000)
    opt = parser.parse_args()

    # Setup the environment
    env = RugbyEnv(
        grid_shape=(21, 11),
        n_agents=opt.n_agents,
        n_opponents=opt.n_opponents,
        max_steps=opt.max_steps,
    )

    # Set seeds.
    random.seed(3)
    np.random.seed(3)
    env.seed(3)

    ACTIONS = 6
    DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

    ROLES = 7
    (
        BALL_CARRIER,
        RIGHT_SUPPORTER,
        LEFT_SUPPORTER,
        RIGHT_SUB_SUPPORTER,
        LEFT_SUB_SUPPORTER,
        RIGH_WING,
        LEFT_WING,
    ) = range(ROLES)
    (
        FORWARD_DEFENDER_1,
        FORWARD_DEFENDER_2,
        FORWARD_DEFENDER_3,
        FORWARD_DEFENDER_4,
        BACK_DEFENDER_1,
        BACK_DEFENDER_2,
        BACK_DEFENDER_3,
    ) = range(ROLES)

    attacking_roles = [
        BALL_CARRIER,
        RIGHT_SUPPORTER,
        LEFT_SUPPORTER,
        RIGHT_SUB_SUPPORTER,
        LEFT_SUB_SUPPORTER,
        RIGH_WING,
        LEFT_WING,
    ]
    defending_roles = [
        FORWARD_DEFENDER_1,
        FORWARD_DEFENDER_2,
        FORWARD_DEFENDER_3,
        FORWARD_DEFENDER_4,
        BACK_DEFENDER_1,
        BACK_DEFENDER_2,
        BACK_DEFENDER_3,
    ]

    random_games = {
        "Random Team vs Random Team": [
            # Attacker team
            RandomAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RandomAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Dummy Greedy Team vs Random Team": [
            # Attacker team
            DummyGreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RandomAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Greedy Team vs Random Team": [
            # Attacker team
            GreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RandomAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Roles Team vs Random Team": [
            # Attacker team
            RoleAgent(
                id=0,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=1,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=2,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=3,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=4,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=5,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=6,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            # Defensive team
            RandomAgent(
                id=7,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=8,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=9,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=10,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=11,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=12,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            RandomAgent(
                id=13,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
    }

    dummy_greedy_games = {
        "Random Team vs Dummy Greedy Team": [
            # Attacker team
            RandomAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            DummyGreedyAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Dummy Greedy Team vs Dummy Greedy Team": [
            # Attacker team
            DummyGreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            DummyGreedyAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Greedy Team vs Dummy Greedy Team": [
            # Attacker team
            GreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            DummyGreedyAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
        "Roles Team vs Dummy Greedy Team": [
            # Attacker team
            RoleAgent(
                id=0,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=1,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=2,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=3,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=4,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=5,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=6,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            # Defensive team
            DummyGreedyAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            DummyGreedyAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
    }

    greedy_games = {
        # "Random Team vs Greedy Team": [
        #     # Attacker team
        #     RandomAgent(
        #         id=0,
        #         n_actions=env.action_space[0].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=1,
        #         n_actions=env.action_space[1].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=2,
        #         n_actions=env.action_space[2].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=3,
        #         n_actions=env.action_space[3].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=4,
        #         n_actions=env.action_space[4].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=5,
        #         n_actions=env.action_space[5].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     RandomAgent(
        #         id=6,
        #         n_actions=env.action_space[6].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     # Defensive team
        #     GreedyAgent(
        #         id=7,
        #         n_actions=env.action_space[7].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=8,
        #         n_actions=env.action_space[8].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=9,
        #         n_actions=env.action_space[9].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=10,
        #         n_actions=env.action_space[10].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=11,
        #         n_actions=env.action_space[11].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=12,
        #         n_actions=env.action_space[12].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=13,
        #         n_actions=env.action_space[13].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        # ],
        # "Dummy Greedy Team vs Greedy Team": [
        #     # Attacker team
        #     DummyGreedyAgent(
        #         id=0,
        #         n_actions=env.action_space[0].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=1,
        #         n_actions=env.action_space[1].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=2,
        #         n_actions=env.action_space[2].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=3,
        #         n_actions=env.action_space[3].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=4,
        #         n_actions=env.action_space[4].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=5,
        #         n_actions=env.action_space[5].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     DummyGreedyAgent(
        #         id=6,
        #         n_actions=env.action_space[6].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     # Defensive team
        #     GreedyAgent(
        #         id=7,
        #         n_actions=env.action_space[7].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=8,
        #         n_actions=env.action_space[8].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=9,
        #         n_actions=env.action_space[9].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=10,
        #         n_actions=env.action_space[10].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=11,
        #         n_actions=env.action_space[11].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=12,
        #         n_actions=env.action_space[12].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=13,
        #         n_actions=env.action_space[13].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        # ],
        # "Greedy Team vs  Greedy Team": [
        #     # Attacker team
        #     GreedyAgent(
        #         id=0,
        #         n_actions=env.action_space[0].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=1,
        #         n_actions=env.action_space[1].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=2,
        #         n_actions=env.action_space[2].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=3,
        #         n_actions=env.action_space[3].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=4,
        #         n_actions=env.action_space[4].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=5,
        #         n_actions=env.action_space[5].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     GreedyAgent(
        #         id=6,
        #         n_actions=env.action_space[6].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=0,
        #     ),
        #     # Defensive team
        #     GreedyAgent(
        #         id=7,
        #         n_actions=env.action_space[7].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=8,
        #         n_actions=env.action_space[8].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=9,
        #         n_actions=env.action_space[9].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=10,
        #         n_actions=env.action_space[10].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=11,
        #         n_actions=env.action_space[11].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=12,
        #         n_actions=env.action_space[12].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        #     GreedyAgent(
        #         id=13,
        #         n_actions=env.action_space[13].n,
        #         n_agents=opt.n_agents,
        #         n_opponents=opt.n_opponents,
        #         team=1,
        #     ),
        # ],
        "Roles Team vs  Greedy Team": [
            # Attacker team
            RoleAgent(
                id=0,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=1,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=2,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=3,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=4,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=5,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=6,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            # Defensive team
            GreedyAgent(
                id=7,
                n_actions=env.action_space[7].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=8,
                n_actions=env.action_space[8].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=9,
                n_actions=env.action_space[9].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=10,
                n_actions=env.action_space[10].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=11,
                n_actions=env.action_space[11].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=12,
                n_actions=env.action_space[12].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
            GreedyAgent(
                id=13,
                n_actions=env.action_space[13].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
            ),
        ],
    }

    roles_games = {
        "Random Team vs Roles Team": [
            # Attacker team
            RandomAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            RandomAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RoleAgent(
                id=7,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=8,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=9,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=10,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=11,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=12,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=13,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
        ],
        "Dummy Greedy Team vs Roles Team": [
            # Attacker team
            DummyGreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            DummyGreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RoleAgent(
                id=7,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=8,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=9,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=10,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=11,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=12,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=13,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
        ],
        "Greedy Team vs Roles Team": [
            # Attacker team
            GreedyAgent(
                id=0,
                n_actions=env.action_space[0].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=1,
                n_actions=env.action_space[1].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=2,
                n_actions=env.action_space[2].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=3,
                n_actions=env.action_space[3].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=4,
                n_actions=env.action_space[4].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=5,
                n_actions=env.action_space[5].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            GreedyAgent(
                id=6,
                n_actions=env.action_space[6].n,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
            ),
            # Defensive team
            RoleAgent(
                id=7,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=8,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=9,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=10,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=11,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=12,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=13,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
        ],
        "Roles Team vs Roles Team": [
            # Attacker team
            RoleAgent(
                id=0,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=1,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=2,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=3,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=4,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=5,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=6,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=0,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            # Defensive team
            RoleAgent(
                id=7,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=8,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=9,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=10,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=11,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=12,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
            RoleAgent(
                id=13,
                n_agents=opt.n_agents,
                n_opponents=opt.n_opponents,
                team=1,
                attack_roles=attacking_roles,
                defense_roles=defending_roles,
            ),
        ],
    }

    # Evaluate teams
    steps_results = {}
    tackles_results = {}
    passes_results = {}
    scores_results = {}

    # step_results_random = {}
    # tackle_results_random = {}
    # pass_results_random = {}
    # score_results_random = {}
    # for game, agents in random_games.items():
    #     print(f"Running {game}.")
    #     steps_res, tackles_res, scores_res, passes_res = run_multi_agent(env, agents, opt.episodes)
    #     step_results_random[game] = steps_res
    #     tackle_results_random[game] = tackles_res
    #     pass_results_random[game] = passes_res
    #     score_results_random[game] = scores_res
    # steps_results["Random Team Tournament Performance"] = step_results_random
    # tackles_results["Random Team Tournament Performance"] = tackle_results_random
    # passes_results["Random Team Tournament Performance"] = pass_results_random
    # scores_results["Random Team Tournament Performance"] = score_results_random

    step_results_greedy = {}
    tackle_results_greedy = {}
    pass_results_greedy = {}
    score_results_greedy = {}
    for game, agents in greedy_games.items():
        print(f"Running {game}.")
        steps_res, tackles_res, scores_res, passes_res = run_multi_agent(env, agents, opt.episodes)
        step_results_greedy[game] = steps_res
        tackle_results_greedy[game] = tackles_res
        pass_results_greedy[game] = passes_res
        score_results_greedy[game] = scores_res
    steps_results["Greedy Team Tournament Performance"] = step_results_greedy
    tackles_results["Greedy Team Tournament Performance"] = tackle_results_greedy
    passes_results["Greedy Team Tournament Performance"] = pass_results_greedy
    scores_results["Greedy Team Tournament Performance"] = score_results_greedy
    
    # step_results_dummy_greedy = {}
    # tackle_results_dummy_greedy = {}
    # pass_results_dummy_greedy = {}
    # score_results_dummy_greedy = {}
    # for game, agents in dummy_greedy_games.items():
    #     print(f"Running {game}.")
    #     steps_res, tackles_res, scores_res, passes_res = run_multi_agent(env, agents, opt.episodes)
    #     step_results_dummy_greedy[game] = steps_res
    #     tackle_results_dummy_greedy[game] = tackles_res
    #     pass_results_dummy_greedy[game] = passes_res
    #     score_results_dummy_greedy[game] = scores_res
    # steps_results["Dummy Greedy Team Tournament Performance"] = step_results_dummy_greedy
    # tackles_results["Dummy Greedy Team Tournament Performance"] = tackle_results_dummy_greedy
    # passes_results["Dummy Greedy Team Tournament Performance"] = pass_results_dummy_greedy
    # scores_results["Dummy Greedy Team Tournament Performance"] = score_results_dummy_greedy
    
    # step_results_roles = {}
    # tackle_results_roles = {}
    # pass_results_roles = {}
    # score_results_roles = {}
    # for game, agents in roles_games.items():
    #     print(f"Running {game}.")
    #     steps_res, tackles_res, scores_res, passes_res = run_multi_agent(env, agents, opt.episodes)
    #     step_results_roles[game] = steps_res
    #     tackle_results_roles[game] = tackles_res
    #     pass_results_roles[game] = passes_res
    #     score_results_roles[game] = scores_res
    # steps_results["Roles-based Team Tournament Performance"] = step_results_roles
    # tackles_results["Roles-based Team Tournament Performance"] = tackle_results_roles
    # passes_results["Roles-based Team Tournament Performance"] = pass_results_roles
    # scores_results["Roles-based Team Tournament Performance"] = score_results_roles


    # Compare results
    for name, res in steps_results.items():
        compare_results(res, title=name, colors=["orange", "blue", "green", "gray"])

    for name, res in tackles_results.items():
        compare_results(
            res,
            title=name,
            metric="Tackles Per Episode",
            colors=["orange", "blue", "green", "gray"],
        )
    
    for name, res in passes_results.items():
        compare_results(
            res,
            title=name,
            metric="Passes Per Episode",
            colors=["orange", "blue", "green", "gray"],
        )

    # compare_scores(scores_results)
