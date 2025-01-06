from sarsa import SARSA


def main():
    sarsa = SARSA()
    sarsa.train(episodes=300000)
    sarsa.plot()
    sarsa.q_table.dump_csv()
    sarsa.test()


if __name__ == "__main__":
    main()
