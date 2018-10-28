import logging
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import gym
import gym_tetris



from dqn import DQN

MAX_EPSILON = 0.9
MIN_EPSILON = 0.1
EPISODE = 10000
STEP = 2000
START_CHEK = 0
CHECK_DUR = 50
CHECK_DISPLAY = False
COST_CURVE = 1000
SAVE_DUR = 1000
MODEL_NAME = "tetris_two"
LOAD_NAME = ""
SET_DUR = 10
DISPLAY = False
TEST_MODE = False

def norm(a):
        b = np.mean(a, axis=2)
        c = np.zeros((20, 10))
        for i in range(20):
            for j in range(10):
                c[i][j]=(b[j*20+5][i*20+5] > 0)
        return c


m_state = np.zeros((20, 10,4))
def ad_s(ol_state, nw_state):
    nw_state = norm(nw_state)
    ol_state = np.delete(ol_state, 0, axis = 2)
    ol_state = np.insert(ol_state, 3, values = nw_state, axis = 2)
    return ol_state

if __name__ == '__main__':
    env = gym.make('Tetris-v0')
    gym.logger.set_level(40)
    best_reward = -20

    reward = 0
    done = False

    start = 0
    print(env.action_space)

    agent = DQN(env.observation_space.shape, 6)
    if not START_CHEK == 0:
        agent.load_model('tetris_fin')

    done = False
    ave_reward = 0
    reward_rec = []

    for e in range(START_CHEK, START_CHEK + EPISODE):
        state = env.reset()
        m_state = np.zeros((20, 10,4))
        state = ad_s(m_state, state)
        total_reward = 0  
        for _ in range(STEP):
            if DISPLAY:
                env.render()  
#            agent.display_Q(state)
            action = agent.e_greedy(state)
            observation, reward, done, _ = env.step(action)
#            print(reward)
            observation = ad_s(state, observation)
    #            if TEST and (e % TEST_EP == 0):
    #                agent.test_cnn(observation)
    #                system("pause")
    #            if reward >= 10:
    #                total_lines += 1
            total_reward += reward
            agent.memorize(state, action, reward, observation, done)
            state = observation
            if done:
                ave_reward += total_reward
                break
        if(e <= 1000):
            if (e) % SET_DUR == 0:
                print("ep [{}/{}] finished.".format(e + 1, EPISODE + START_CHEK))
                ave_reward = 0
                total_lines = 0
                new_ep = (MIN_EPSILON - MAX_EPSILON) / (1000) * (e) + MAX_EPSILON
                agent.set_epsilon(new_ep)
                print("set epsilon:",new_ep)
        else:
            if(e) % 100 == 0:
                print("ep [{}/{}] finished. keep 0.1".format(e + 1, EPISODE + START_CHEK))

        if (e+1) % CHECK_DUR == 0:
            rew = 0
            for r in range(2):
                state = env.reset()
                m_state = np.zeros((20, 10,4))
                state = ad_s(m_state, state)
                total_reward = 0
                for _ in range(STEP):
                    if CHECK_DISPLAY:
                        env.render()
                        agent.display_Q(state)
        #            time.sleep(0.1)
                    action = agent.greedy_action(state)
                    state_, reward, done, info= env.step(action)
                    state = ad_s(state,state_)
                    total_reward += reward
                    if done:
                        break
                rew += total_reward
                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.save_model(MODEL_NAME + "_best_{}".format(best_reward))
            rew /= 2
            reward_rec.append(rew)
            print("ep {} got ave reward: {} \nbest: {}".format(e + 1, rew, best_reward))
            
            
        if (e + 1)% COST_CURVE == 0:
            plt.plot(np.arange(1, len(agent.cost_tmp) + 1), agent.cost_tmp)
            plt.ylabel('cost')
            plt.xlabel('traning steps')
#            plt.show()
            plt.savefig('cost_fig/{}_cost_{}_to_{}.jpg'.format(MODEL_NAME, e - COST_CURVE + 2, e + 1))
            plt.clf()
            agent.cost_tmp = []
        """ if (e + 1) % SAVE_DUR == 0:
            agent.save_model("{}_{}".format(MODEL_NAME, e+1))"""
    if TEST_MODE:
        plt.plot(np.arange(1, len(agent.cost_tmp) + 1), agent.cost_tmp)
        plt.ylabel('cost')
        plt.xlabel('traning steps')
        plt.show()
    plt.plot(np.arange(1, len(agent.cost_board) + 1), agent.cost_board)
    plt.ylabel('cost')
    plt.xlabel('traning steps(x1000)')
#   plt.show()
    plt.savefig('cost_fig/{}_cost_total_{}_to_{}.jpg'.format(MODEL_NAME, START_CHEK + 1, START_CHEK + EPISODE))
    plt.clf()
    plt.plot(np.arange(1, len(reward_rec) + 1) * 10, reward_rec)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.savefig("reward_2.jpg")
    agent.save_model(MODEL_NAME + "_fin")
    f = open("reward_2.out","w")
    print(reward_rec, file = f)
    f.close()
