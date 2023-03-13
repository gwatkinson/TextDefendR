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
        "scenario",
        "target_model_dataset",
        "target_model",
        "attack_toolchain",
        "attack_name",
        "test_index",
        "original_text_identifier",
    ]
)
