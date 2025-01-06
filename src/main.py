from q_learning import QLearning


def main():
    q = QLearning()
    q.train(episodes=300000)
    q.plot()
    q.q_table.dump_csv()
    q.test()


if __name__ == "__main__":
    main()
