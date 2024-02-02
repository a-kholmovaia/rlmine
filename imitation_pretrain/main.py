from pretrainer import Pretrainer
if __name__ == '__main__':
    imitation_learner = Pretrainer()
    imitation_learner.train(epochs=100)