import hashlib


def hash_pk_tuple(tup):
    assert len(tup) == 7, "react pk does not have 7 items!"
    str_tup = "_".join(str(i) for i in tup)
    id_ = str(int(hashlib.md5(str_tup.encode("ascii")).hexdigest(), 16))
    return id_


def get_pk_tuple(df, index):
    _pk = sorted(
        [
            "attack_name",
            "attack_toolchain",
            "original_text_identifier",
            "scenario",
            "target_model",
            "target_model_dataset",
            "test_index",
        ]
    )
    _pk = [df.at[index, i] for i in _pk]
    return _pk
