from random import Random
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    pairwise_distances,
    make_scorer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    f1_score,
)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.svm import LinearSVC

from baycomp import two_on_single

from LANDMark import LANDMarkClassifier

from multiprocessing import Pool

from joblib import Parallel, delayed, parallel_backend

from imblearn.under_sampling import RandomUnderSampler

from Bio import SeqIO

from scipy.sparse import issparse
from scipy.sparse import hstack as sp_hstack

# The VarWrapper() class simply removes binary features with zero variance.
class VarWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, comp_type):
        self.comp_type = comp_type

    def fit_resample(self, X, y):
        if self.comp_type == "Combined":
            # Split the dataset
            X_spec = X[:, 0:12]
            X_pa = X[:, 12:]

            # Remove zero variance features
            self.var_thresh = VarianceThreshold().fit(X_pa)

            X_trf = self.var_thresh.transform(X_pa)

            # Recombine
            if issparse(X):
                X_trf = sp_hstack((X_spec, X_trf))

            else:
                X_trf = np.hstack((X_spec, X_trf))

            return X_trf, y

        elif self.comp_type == "SNVs_AAMut":
            # Remove zero variance features
            self.var_thresh = VarianceThreshold().fit(X)

            X_trf = self.var_thresh.transform(X)

            return X_trf, y

        return X, y

    def transform(self, X):
        if self.comp_type == "Combined":
            # Split the dataset
            X_spec = X[:, 0:12]
            X_pa = X[:, 12:]

            X_trf = self.var_thresh.transform(X_pa)

            # Recombine
            X_trf = np.hstack((X_spec, X_trf))

            return X_trf

        elif self.comp_type == "SNVs_AAMut":
            # Remove zero variance features
            X_trf = self.var_thresh.transform(X)

            return X_trf

        return X

def calc_bacc(classifier, cv, X, y, resample, resample_method):
    """
    Calculates the Balanced Accuracy Score
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object
        X: Feature Matrix
        y: Class labels

    """

    baccs = []

    counter = 0

    for train, test in cv.split(X, y):
        print(counter + 1)
        counter = counter + 1

        X_train = X[train]
        y_train = y[train]

        if resample:
            X_re, y_re = resample_method.fit_resample(X_train, y_train)

        else:
            X_re, y_re = X_train, y_train

        clf = classifier.fit(X_re, y_re)

        p = clf.predict(X[test])

        bacc = balanced_accuracy_score(y[test], p)

        print(type(classifier), bacc)

        baccs.append(bacc)

    mu = np.mean(baccs)
    std = np.std(baccs, ddof=1)

    return mu, std, baccs

if __name__ == "__main__":

    MIN_MUTATIONS = 10

    # Get a dataframe of mutations for each sample and the metadata for each sample
    with open("Deer GISAID/nextclade/nextclade.csv", "r") as file:
        data = file.readlines()

    mut_data_new = []
    meta_data = []
    total_seq = 0
    selected_seq = 0
    for entry in data[1:]:
        total_seq += 1

        subs = entry.strip("\n").split(";")[24].split(",")

        lab = entry.strip("\n").split(";")

        meta = lab[1].split("|")

        meta.append(lab[2])  # Clade
        meta.append(entry.strip("\n").split(";")[30])  # aaInsertions
        meta.append(entry.strip("\n").split(";")[28])  # aaSubstitutions
        meta.append(entry.strip("\n").split(";")[29])  # aaDeletions
        meta.append(entry.strip("\n").split(";")[24])  # ntSubstitutions
        meta.append(entry.strip("\n").split(";")[38])  # missing

        meta_data.append(meta)

    meta_data = np.asarray(meta_data)

    # Create mutation spectrum w.r.t original refernece strain
    mutation_types = set()
    sub_list = []
    for i, nt_muts in enumerate(meta_data[:, 11]):

        row = nt_muts.split(",")

        if len(row) - 1 == 0:
            pass
        else:
            for mutation in row:
                first = mutation[0]
                last = mutation[-1]
                m_type = "%s > %s" % (first, last)
                mutation_types.add(m_type)

    mutation_types = {entry: i for i, entry in enumerate(mutation_types)}
    mutation_spectrum = np.zeros(shape=(meta_data.shape[0], len(mutation_types)))
    for i, nt_muts in enumerate(meta_data[:, 11]):
        row = nt_muts.split(",")

        if len(row) - 1 == 0:
            pass

        else:
            for entry in row:
                first = entry[0]
                last = entry[-1]
                m_type = "%s > %s" % (first, last)
                idx = mutation_types[m_type]

                mutation_spectrum[i, idx] += 1.0

    cols = [x for x in mutation_types.keys()]
    for i, entry in enumerate(cols):
        if "T" in entry:
            cols[i] = entry.replace("T", "U")

    # Filter out samples with to few mutations
    mut_count = mutation_spectrum.sum(axis=1)
    mut_count = np.asarray(
        [True if x >= MIN_MUTATIONS else False for x in mut_count]
    )

    # Convert to relative counts
    mutation_spectrum_rel = (
       mutation_spectrum / mutation_spectrum.sum(axis=1)[:, None]
     )

    # Create X and y
    X = np.copy(mutation_spectrum_rel, "C")
    y = pd.DataFrame(
        meta_data,
        columns=[
            "Strain",
            "Date",
            "Region",
            "Province",
            "Host",
            "Country",
            "Sample Type",
            "Clade",
            "aaInsertions",
            "aaSubstitutions",
            "aaDeletions",
            "ntSubstitutions",
            "missing",
        ],
    )

    # Filter out samples with few mutations
    X = X[mut_count]
    y = y[mut_count]

    # Pick human samples from Pickering et al. (2022)
    pickering_et_al = pd.read_csv("Deer GISAID/EPI_Pickering_et_al.csv").values[
        :, [1, 2, 3]
    ]

    pickering_data_w_mink_deer = []
    for row in y.values:
        tmp = "/".join(row[0].split("/")[1:])

        if (
            tmp in pickering_et_al
            and row[4] == "Human"
            and row[2] == "North America"
        ):
            pickering_data_w_mink_deer.append(True)

        elif (
            row[4] == "Neovison vison" or row[4] == "Odocoileus virginianus"
        ) and row[2] == "North America":
            pickering_data_w_mink_deer.append(True)

        else:
            pickering_data_w_mink_deer.append(False)

    pickering_data_w_mink_deer = np.asarray(pickering_data_w_mink_deer)

    X = X[pickering_data_w_mink_deer]
    y = y[pickering_data_w_mink_deer]

    # Get All Clades, But Filtered
    MIN_CLADE_COUNT = 20
    clade, clade_count = np.unique(y["Clade"], return_counts=True)
    selected_clades = clade[np.where(clade_count >= MIN_CLADE_COUNT, True, False)]
    clade_idx = []
    for entry in y["Clade"].values:
        if entry in selected_clades:
            clade_idx.append(True)

        else:
            clade_idx.append(False)

    clade_idx = np.asarray(clade_idx)

    X_all = X[clade_idx]
    y_all = y[clade_idx]

    # Get the 20C dataset
    selected_clades = ["20C"]
    clade_idx = []
    for entry in y["Clade"].values:
        if entry in selected_clades:
            clade_idx.append(True)

        else:
            clade_idx.append(False)

    clade_idx = np.asarray(clade_idx)

    X_20C = X[clade_idx]
    y_20C = y[clade_idx]

    # Get the 21J dataset
    selected_clades = ["21J"]
    clade_idx = []
    for entry in y["Clade"].values:
        if entry in selected_clades:
            clade_idx.append(True)

        else:
            clade_idx.append(False)

    clade_idx = np.asarray(clade_idx)

    X_21J = X[clade_idx]
    y_21J = y[clade_idx]

    datasets = {"All": [X_all, y_all], "20C": [X_20C, y_20C], "21J": [X_21J, y_21J]}

    # Add SNP and AA mutation data
    for dataset, X_y in datasets.items():
        y_tmp = X_y[1]

        mut_locations = set()

        X_aa_nts = y_tmp.ntSubstitutions.str.split(",").values
        for row in X_aa_nts:
            for entry in row:
                if entry != "":
                    mut_locations.add(entry)

        X_aa_mut = y_tmp.aaSubstitutions.str.split(",").values
        for row in X_aa_mut:
            for entry in row:
                if entry != "":
                    mut_locations.add(entry)

        X_aa_del = y_tmp.aaDeletions.str.split(",").values
        for row in X_aa_del:
            for entry in row:
                if entry != "":
                    mut_locations.add(entry)

        X_aa_ins = y_tmp.aaInsertions.str.split(",").values
        for row in X_aa_ins:
            for entry in row:
                if entry != "":
                    mut_locations.add(entry)

        mut_locations = {mut: i for i, mut in enumerate(mut_locations)}

        mut_cols = []
        for k, v in mut_locations.items():
            mut_cols.append(k)

        X_mutation = np.zeros(
            shape=(X_y[0].shape[0], len(mut_locations)), dtype=int
        )
        for i, row in enumerate(X_aa_nts):
            for entry in row:
                if entry == "":
                    pass
                else:
                    j = mut_locations[entry]

                    X_mutation[i, j] = 1

        for i, row in enumerate(X_aa_mut):
            for entry in row:
                if entry == "":
                    pass
                else:
                    j = mut_locations[entry]

                    X_mutation[i, j] = 1

        for i, row in enumerate(X_aa_del):
            for entry in row:
                if entry == "":
                    pass
                else:
                    j = mut_locations[entry]

                    X_mutation[i, j] = 1

        for i, row in enumerate(X_aa_ins):
            for entry in row:
                if entry == "":
                    pass
                else:
                    j = mut_locations[entry]

                    X_mutation[i, j] = 1

        high_freq = (
            X_mutation.sum(axis=0) >= 5 #0 for grabbing all 
        )  # 5 normally, 0 just for grabbing all mutations

        X_mutation = X_mutation[:, high_freq]
        cols_mutation = np.asarray(mut_cols)[high_freq]

        X_combined = np.hstack((X_y[0], X_mutation))
        cols_combined = np.hstack((cols, cols_mutation))

        X_y.extend([cols, "Spectrum"])
        datasets[dataset] = [
            X_y,
            [X_combined, y_tmp, cols_combined, "Combined"],
            [X_mutation, y_tmp, cols_mutation, "SNVs_AAMut"],
        ]

    for dataset_name, dataset in datasets.items():
        pd.DataFrame(dataset[2][0], index = dataset[2][1]["Strain"], columns = dataset[2][2]).to_csv("Figures Final/Positive Selection/%s_full_aa_snv.csv" %dataset_name)
        dataset[2][1].to_csv("Final_Results/Selection/%s_full_meta.csv" %dataset_name)
