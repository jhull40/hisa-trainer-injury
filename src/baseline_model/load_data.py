import pandas as pd


def load_data():
    races = pd.read_csv('../../hisa-data/races_trainers_a.csv')
    races = races[races['trainer_name'].isin(races['trainer_name'].value_counts().index[:100])]

    return races


if __name__ == '__main__':
    data = load_data()
    print(data.head())