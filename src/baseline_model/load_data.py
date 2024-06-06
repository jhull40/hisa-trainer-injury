import pandas as pd


def load_data():
    df = pd.read_csv('/users/jameshull/documents/github/hisa-data/races_2023.csv', nrows=1000)

    return df


if __name__ == "__main__":
    data = load_data()
    print(data.head())
