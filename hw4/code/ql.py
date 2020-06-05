from environment import *
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilons = [0.9]
alphas = [0.1]
gammas = [1]
max_e = 100000
test_game_num = 100000

states = [d * 100 + p for d in range(1, 11) for p in range(1, 22)]
states.append("terminal")
actions = [True, False]

q_table = pd.DataFrame(data=[[0 for _ in actions] for _ in states],
                       index=states, columns=actions)


def train(e, a, g):
    global q_table

    for s in states:
        for ac in actions:
            q_table.loc[s, ac] = 0
    es, vsum, cumes, cumr = [], [], [], []
    rs = 0

    for i in range(max_e):
        current_s = init_state()

        if (i + 1) % 400 == 0:
            cumr.append(rs)
            cumes.append(i + 1)
            rs = 0

        while current_s != "terminal":
            if (random.uniform(0, 1) > e) or (q_table.loc[current_s] == 0).all():
                current_a = random.choice(actions)
            else:
                current_a = q_table.loc[current_s].idxmax()

            next_s, reward = step(current_s, current_a)
            rs += reward
            next_state_q_values = q_table.loc[next_s, actions]
            q_table.loc[current_s, current_a] += a * (
                    reward + g * next_state_q_values.max() - q_table.loc[current_s, current_a])
            current_s = next_s

        es.append(i)
        vsum.append(np.sum(np.array(q_table.max(axis=1))))
    return es, vsum, cumes, cumr


def draw_learning_curve(res):
    fig = plt.figure()
    eps, alp, gam = 0, 0, 0
    for i in res:
        x = res[i][0]
        y = res[i][1]
        plt.plot(x, y, label="epsilon={}, alpha={}, gamma={}".format(i[0], i[1], i[2]))
        plt.legend()
        eps, alp, gam = i[0], i[1], i[2]
    plt.savefig("{}e{}a{}g_{}.png".format(eps, alp, gam, max_e))


# plt.show()


def draw_learning_curve1(res):
    fig = plt.figure()
    eps, alp, gam = 0, 0, 0
    for i in res:
        x = res[i][2]
        y = res[i][3]
        plt.plot(x, y, label="epsilon={}, alpha={}, gamma={}".format(i[0], i[1], i[2]))
        plt.legend()
        eps, alp, gam = i[0], i[1], i[2]
    # plt.show()
    plt.savefig("{}e{}a{}g_{}_.png".format(eps, alp, gam, max_e))


def draw_3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    dealers = np.arange(1, 11)
    players = np.arange(1, 22)

    z = np.array(q_table.max(axis=1)[:-1]).reshape(21, 10)
    d, p = np.meshgrid(dealers, players)

    ax.plot_surface(d, p, z, cmap='rainbow')
    plt.show()


def win_rate(game_num):
    global q_table
    win, lose, fair = 0, 0, 0

    for i in range(game_num):
        current_s = init_state()

        while current_s != "terminal":
            if (random.uniform(0, 1) > e) or (q_table.loc[current_s] == 0).all():
                current_a = random.choice(actions)
            else:
                current_a = q_table.loc[current_s].idxmax()

            next_s, reward = step(current_s, current_a)
            current_s = next_s

        if reward == 1.0:
            win += 1
        elif reward == -1.0:
            lose += 1
        else:
            fair += 1

    print("total={}, win={}, lose={}, fair={}, win rate={}"
          .format(game_num, win, lose, fair, win * 1.0 / game_num))


def stategy_win_rate(game_num, stg):
    if stg == "random":
        q_table_ = pd.DataFrame(data=[[random.uniform(0, 1) for _ in actions] for _ in states],
                                  index=states, columns=actions)
    elif stg == "hit":
        q_table_ = pd.DataFrame(data=[[0 for _ in actions] for _ in states],
                               index=states, columns=actions)
        q_table_.loc[states, True] = 1
    elif stg == "stick":
        q_table_ = pd.DataFrame(data=[[0 for _ in actions] for _ in states],
                                 index=states, columns=actions)
        q_table_.loc[states, False] = 1

    win, lose, fair = 0, 0, 0

    for i in range(game_num):
        current_s = init_state()

        while current_s != "terminal":
            if (random.uniform(0, 1) > e) or (q_table_.loc[current_s] == 0).all():
                current_a = random.choice(actions)
            else:
                current_a = q_table_.loc[current_s].idxmax()

            next_s, reward = step(current_s, current_a)
            current_s = next_s

        if reward == 1.0:
            win += 1
        elif reward == -1.0:
            lose += 1
        else:
            fair += 1

    print("total={}, win={}, lose={}, fair={}, win rate={}"
          .format(game_num, win, lose, fair, win * 1.0 / game_num))


if __name__ == '__main__':
    res = {}
    for e in epsilons:
        for a in alphas:
            for g in gammas:
                es, vsum, cumes, cumr = train(e, a, g)
                res[(e, a, g)] = (es, vsum, cumes, cumr)
                print("epsilon={}, alpha={}".format(e, a))
            # win_rate(test_game_num)
    #draw_learning_curve(res)
    #draw_learning_curve1(res)
    #draw_3d()

    import pickle as pkl
    with open("q_table.pkl", 'wb') as f:
        pkl.dump(q_table, f)

    print("use random strategy.")
    stategy_win_rate(game_num=test_game_num, stg="random")
    print("use hit strategy.")
    stategy_win_rate(game_num=test_game_num, stg="hit")
    print("use random stick.")
    stategy_win_rate(game_num=test_game_num, stg="stick")
