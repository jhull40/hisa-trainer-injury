import pandas as pd


def load_data():
    races = pd.read_csv("../../hisa-data/races_3sample.csv")

    return races


if __name__ == "__main__":
    data = load_data()
    print(data.head())
