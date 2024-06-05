import pandas as pd


def load_data():
    races = pd.read_csv("../../hisa-data/races_trainers_a.csv")

    return races
