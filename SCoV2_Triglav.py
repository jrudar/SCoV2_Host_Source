import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from triglav import Triglav, ETCProx

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder

from multiprocessing import Pool

from Bio import SeqIO

from baycomp import two_on_single

from joblib import Parallel, delayed, parallel_backend
    
if __name__ == "__main__":

    MIN_MUTATIONS = 10

    with open("Deer GISAID/nextclade/nextclade_csv.csv", "r") as file:
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
        print(i)

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

    # Classification
    # Filter on large number of sequences
    mut_count = mutation_spectrum.sum(axis=1)
    mut_count = np.asarray(
        [True if x >= MIN_MUTATIONS else False for x in mut_count]
    )

    # Convert to relative counts
    #mutation_spectrum_rel = (
        #   mutation_spectrum / mutation_spectrum.sum(axis=1)[:, None]
    #)

    mutation_spectrum_rel = mutation_spectrum 

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
    pickering_et_al = pd.read_csv("Deer_GISAID/EPI_Pickering_et_al.csv").values[
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

    #Get All Clades, But Filtered
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

    #Get the 20C dataset
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

    #Get the 21J dataset
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
        
    datasets ={"All": [X_all, y_all],
                "20C": [X_20C, y_20C],
                "21J": [X_21J, y_21J]}

    # Add SNP and AA mutation data
    add_snp_data = True
    if add_snp_data :
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

            X_mutation = np.zeros(shape=(X_y[0].shape[0], len(mut_locations)), dtype=int)
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

            high_freq = X_mutation.sum(axis=0) >= 5

            X_mutation = X_mutation[:, high_freq]
            cols_mutation = np.asarray(mut_cols)[high_freq]

            datasets[dataset] = [X_mutation, y_tmp, cols_mutation, "SNVs_AAMut"]

    splitter = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 20, random_state = 0)

    selected_features = []

    i = 0

    d_set = "All"
    X = datasets[d_set][0]
    y = datasets[d_set][1]
    cols_final = datasets[d_set][2]

    for train, test in splitter.split(X, y["Host"].values):
        X_train = X[train]
        y_train = y["Host"].values[train]

        X_test = X[test]
        y_test = y["Host"].values[test]

        print(np.unique(y_train, return_counts = True)[1])
        print(np.unique(y_train))
        print(X_train.shape, X_test.shape)

        #Select Features
        f_select_model = Triglav(per_class_imp = False,
                                    metric = ETCProx(),
                                    n_iter_fwer = 8,
                                    n_jobs = 35,
                                    alpha = 0.01,
                                    verbose = 3,
                                    n_iter = 60,
                                    estimator = LogisticRegressionCV(max_iter = 2000, scoring = make_scorer(balanced_accuracy_score), multi_class="multinomial"),
                                    run_stage_2 = False,
                                    sampler = RandomUnderSampler(sampling_strategy = {0: 600, 1: 150, 2: 375})
                                    )
                                         
        #Select flat cluster cutpoint based on dendrogram
        f_select_model.thresh = 0.5

        f_select_model.fit(X_train, y_train)

        #Transform data
        X_train_trf = f_select_model.transform(X_train)
        X_test_trf = f_select_model.transform(X_test)
                    
        #Save selected features
        idx = f_select_model.selected_
        selected_features.append(idx)

        i += 1

    stability_df = pd.DataFrame(np.where(np.asarray(selected_features) == True, 1, 0), columns = np.asarray(cols_final))
    stability_df.to_csv("Final_Results/Triglav/%s/Stability_Results.csv" %d_set)
        
