import os


def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print("mkfile warning: making fdir since DNE:", fpath)
        os.makedirs(os.path.dirname(fpath))
    else:
        pass
