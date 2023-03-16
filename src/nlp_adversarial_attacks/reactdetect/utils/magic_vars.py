SUPPORTED_TARGET_MODELS = {"roberta", "bert", "distilcamembert"}
SUPPORTED_TARGET_MODEL_DATASETS = {
    "fnc1",
    "civil_comments",
    "hatebase",
    "wikipedia",
    "sst",
    "imdb",
    "climate-change_waterloo",
    "nuclear_energy",
    "wikipedia_personal",
    "gab_dataset",
    "reddit_dataset",
    "allocine",
}
NUM_LABELS_LOOKUP = {
    "fnc1": 4,
    "civil_comments": 2,
    "hatebase": 2,
    "wikipedia": 2,
    "sst": 2,
    "imdb": 2,
    "climate-change_waterloo": 3,
    "nuclear_energy": 3,
    "gab_dataset": 2,
    "reddit_dataset": 2,
    "wikipedia_personal": 2,
    "allocine": 2,
}
SUPPORTED_ATTACKS = {
    "bae",
    "bert",
    "checklist",
    "clare",
    "deepwordbug",
    "faster_genetic",
    "genetic",
    "hotflip",
    "iga_wang",
    "input_reduction",
    "kuleshov",
    "pruthi",
    "pso",
    "pwws",
    "textbugger",
    "textfooler",
    "clean",
}
PRIMARY_KEY_FIELDS = sorted(
    [
        "attack_name",
        "attack_toolchain",
        "scenario",
        "target_model",
        "target_model_dataset",
        "test_index",
    ]
)
