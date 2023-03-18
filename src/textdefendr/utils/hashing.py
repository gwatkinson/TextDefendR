import hashlib

from textdefendr.utils.magic_vars import PRIMARY_KEY_FIELDS


def hash_pk_tuple(tup):
    assert len(tup) == len(PRIMARY_KEY_FIELDS)
    str_tup = "_".join(str(i) for i in tup)
    id_ = str(int(hashlib.md5(str_tup.encode("ascii")).hexdigest(), 16))
    return id_


def get_pk_tuple(df, index):
    pk = [df.at[index, i] for i in PRIMARY_KEY_FIELDS]
    return pk


def get_pk_tuple_from_pandas_row(pandas_row):
    return tuple([pandas_row[i] for i in PRIMARY_KEY_FIELDS])
