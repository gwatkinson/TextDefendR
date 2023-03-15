import json
import os
from pathlib import Path


def grab_joblibs(root_dir):
    """
    grab all joblibs in a root dir
    """
    root_dir = Path(root_dir)
    files = list(root_dir.glob("**/*.joblib"))
    return files


def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print("mkfile warning: making fdir since DNE:", fpath)
        os.makedirs(os.path.dirname(fpath))
    else:
        pass


def mkdir_if_dne(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)


def vim_write_zz(fpath, string):
    """
    write as if vim write zz
    """
    mkfile_if_dne(fpath)
    with open(fpath, "w") as ofp:
        ofp.write(string + "\n")


def load_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def dump_json(data, fpath):
    with open(fpath, "w") as outfile:
        json.dump(data, outfile, indent=4)
