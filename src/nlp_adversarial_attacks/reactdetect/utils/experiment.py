import datetime
from pathlib import Path

import pandas as pd

from nlp_adversarial_attacks.reactdetect.utils.file_io import (
    dump_json,
    mkdir_if_dne,
    vim_write_zz,
)
from nlp_adversarial_attacks.reactdetect.utils.pandas_ops import show_df_stats


class Experiment:
    def __init__(
        self,
        train_df,
        val_df,
        test_df_release,
        test_df_hidden,
        name,
        create_subfolder=False,
        aux=None,
    ):
        if aux is None:
            aux = {}

        assert (
            isinstance(train_df, pd.DataFrame)
            and isinstance(val_df, pd.DataFrame)
            and isinstance(test_df_release, pd.DataFrame)
            and isinstance(test_df_hidden, pd.DataFrame)
        )

        # again if same src is attacked 10 times it should all just go to either train OR test data
        s1 = set(train_df["unique_src_instance_identifier"])
        s2 = set(val_df["unique_src_instance_identifier"])
        s3 = set(test_df_release["unique_src_instance_identifier"])
        s4 = set(test_df_hidden["unique_src_instance_identifier"])

        assert len(s1.intersection(s2)) == 0
        assert len(s1.intersection(s3)) == 0
        assert len(s1.intersection(s4)) == 0
        assert len(s2.intersection(s3)) == 0
        assert len(s2.intersection(s4)) == 0
        assert len(s4.intersection(s3)) == 0

        assert len(train_df["attack_name"].unique()) > 1
        assert len(val_df["attack_name"].unique()) > 1
        assert len(test_df_release["attack_name"].unique()) > 1
        assert len(test_df_hidden["attack_name"].unique()) > 1

        (
            self.train_df,
            self.val_df,
            self.test_df_release,
            self.test_df_hidden,
            self.name,
            self.create_subfolder,
        ) = (train_df, val_df, test_df_release, test_df_hidden, name, create_subfolder)
        self.aux = aux

    def stats(self):
        out = """\n\n pre-splitted training-testing\n\n```\n"""
        out = out + "train\n" + show_df_stats(self.train_df)
        out = out + "\nval\n" + show_df_stats(self.val_df)
        out = out + "\ntest release\n" + show_df_stats(self.test_df_release)
        out = out + "\ntest hidden\n" + show_df_stats(self.test_df_hidden)
        out = out + """\n```"""
        return out

    def str_now(self):
        return str(datetime.datetime.now())

    def to_markdown(self):
        md_info = (
            """### """
            + self.name
            + """\ngenerated on """
            + self.str_now()
            + """\n### datastats\n"""
        )
        md_info = md_info + self.stats() + """\n```"""
        return md_info

    def dump(self, exp_root_dir):
        odir = exp_root_dir
        if self.create_subfolder:
            odir = Path(odir, self.name)
        print(" ***", self.name, "saving to", odir)
        mkdir_if_dne(odir)
        self.train_df.to_csv(Path(odir, "train.csv"), index=False)
        self.val_df.to_csv(Path(odir, "val.csv"), index=False)
        self.test_df_release.to_csv(Path(odir, "test_release.csv"), index=False)
        self.test_df_release.to_csv(Path(odir, "test_hidden.csv"), index=False)
        md_info = self.to_markdown()
        vim_write_zz(Path(odir, "README.md"), md_info)

        exp_info = {}

        if "clean_vs" in self.name:
            exp_info["is_binary"] = True
        else:
            exp_info["is_binary"] = False

        exp_info["setting"] = self.name

        exp_info.update(self.aux)

        dump_json(exp_info, Path(odir, "settings.json"))

        return self
