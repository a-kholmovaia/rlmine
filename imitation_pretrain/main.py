from demonstrations_learning import Pretrainer

if __name__ == '__main__':
    imitation_learner = Pretrainer(env_name='MineRLNavigateDense-v0')
    imitation_learner.train(epochs=50)