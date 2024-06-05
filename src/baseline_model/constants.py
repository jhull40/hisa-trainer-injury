# TODO: Add sex column
COLS_FOR_MODEL = [
    "age",
    "sex",
    "registration_number",
    "race_surface_Dirt",
    "race_surface_Synthetic",
    "race_surface_Turf",
    "race_condition_Fast",
    "race_condition_Firm",
    "race_condition_Good",
    "race_condition_Muddy",
    "race_condition_Sloppy",
    "race_condition_Wet Fast",
    "race_condition_Yielding",
    "race_type_Allowance",
    "race_type_Allowance Optional Claimer",
    "race_type_Claiming",
    "race_type_Maiden",
    "race_type_Maiden Claiming",
    "race_type_Starter Allowance",
]

TARGET = "dnf"

SEED = 524
TOP_CONDITIONS = ["Fast", "Firm", "Good", "Sloppy", "Muddy", "Wet Fast", "Yielding"]
TOP_RACE_TYPES = [
    "Claiming",
    "Maiden Claiming",
    "Maiden",
    "Allowance",
    "Allowance Optional Claimer",
    "Starter Allowance",
]
