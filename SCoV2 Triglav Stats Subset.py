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

from scipy.stats import binom_test
from statsmodels.stats.multitest import multipletests

from LANDMark import LANDMarkClassifier

from multiprocessing import Pool

from joblib import Parallel, delayed, parallel_backend

from imblearn.under_sampling import RandomUnderSampler

from Bio import SeqIO

def getNewick(node, newick, parentdist, leaf_names):
    """
    Converts SciPy Linkage matrix to Newick format
    From: https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format/31878514#31878514
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick


def print_nwk(dataset_name, tree_nwk):
    with open(
        "Figures Final/Positive Selection/%s/X_triglav.nwk" % dataset_name, "w"
    ) as file:
        print(tree_nwk, file=file)


if __name__ == "__main__":

    # This block of code will take in each dataset (20C, 21J, All)
    # It will report results of feature selection performance (generalization and F1 Score)
    # It will generate a subset of sequences and metadata for constructing phylogenies/clustermaps
    # It will calculate SAGE importance scores
    get_cluster_map_stability_stats = True
    if get_cluster_map_stability_stats:
        dataset_name = "All"

        # Load the stability data
        X_stability = pd.read_csv(
            "Selection/%s/Stability_Results.csv"
            %dataset_name
        ).astype(int)

        X_cols = X_stability.columns.values[1:]

        X = pd.read_csv(
            "Final_Results/Selection/%s_full_aa_snv.csv" % dataset_name
        )
        y_meta = pd.read_csv(
            "Figures Final/Positive Selection/%s_full_meta.csv" % dataset_name
        ).astype(str)[["Host", "Strain", "Province", "Clade", "Country"]]

        # Remove Rare Clade-Province combinations if dataset is All Clades
        if dataset_name == "All":
            combos = []
            locations = []
            for i, x in enumerate(y_meta[["Clade", "Host", "Province"]].values):
                combos.append("-".join(x))

            combo_name, combo_num = np.unique(combos, return_counts=True)
            combo_num = combo_num >= 10
            combo_name = combo_name[combo_num]

            for i, x in enumerate(y_meta[["Clade", "Host", "Province"]].values):
                tmp = "-".join(x)

                if tmp in combo_name:
                    locations.append(i)

            y_meta = y_meta.loc[locations, :]
            X = X.loc[locations, :]

        # Remove Rare Host-Geographic combinations in 21J
        elif dataset_name == "21J":
            combos = []
            locations = []
            for i, x in enumerate(y_meta[["Host", "Province"]].values):
                combos.append("-".join(x))

            combo_name, combo_num = np.unique(combos, return_counts=True)
            combo_num = combo_num >= 3
            combo_name = combo_name[combo_num]

            for i, x in enumerate(y_meta[["Host", "Province"]].values):
                tmp = "-".join(x)

                if tmp in combo_name:
                    locations.append(i)

            y_meta = y_meta.loc[locations, :]

            X = X.loc[locations, X.columns.values[1:]]

        # Remove Rare Host-Geographic combinations in 20C
        elif dataset_name == "20C":
            combos = []
            locations = []
            for i, x in enumerate(y_meta[["Host", "Province"]].values):
                combos.append("-".join(x))

            combo_name, combo_num = np.unique(combos, return_counts=True)
            combo_num = combo_num >= 10
            combo_name = combo_name[combo_num]

            for i, x in enumerate(y_meta[["Host", "Province"]].values):
                tmp = "-".join(x)

                if tmp in combo_name:
                    locations.append(i)

            y_meta = y_meta.loc[locations, :]
            X = X.loc[locations, X.columns.values[1:]]

        # Re-Write Province Column to Reflect Country if 'nans' occur or the threshold for occurance is too small
        prov, prov_sz = np.unique(y_meta["Province"], return_counts=True)
        prov = {p: prov_sz[i] for i, p in enumerate(prov)}

        prov_small_num = []
        for p in y_meta[["Country", "Province"]].values:
            p_sz = prov[p[1]]

            if p[1] == "nan":
                prov_small_num.append("%s (Minor)" % p[0])

            else:
                prov_small_num.append(p[1])

        y_meta["ProvOrig"] = np.asarray(prov_small_num)

        X.index = y_meta["Strain"].values
        y_meta.index = y_meta["Strain"].values

        y = y_meta["Host"]

        X_cols = X_stability.columns.values[1:]

        # Find the best dataset
        scores_sel = []
        scores_not = []
        scores_f1_sel = []
        scores_f1_not = []
        best_cols = None
        best_score = -9999
        best_data = None
        for i, row_vals in enumerate(X_stability.values):
            col_subset = X_stability.columns.values[1:][
                np.where(row_vals[1:] > 0, True, False)
            ]
            not_col_subset = X_stability.columns.values[1:][
                np.where(row_vals[1:] > 0, False, True)
            ]

            if dataset_name != "21J":
                if dataset_name == "20C":
                    # Split data into training and testing samples
                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        y_m_train,
                        y_m_test,
                    ) = train_test_split(
                        X.values,
                        y.values,
                        y_meta,
                        test_size=0.2,
                        random_state=i,
                        stratify=y_meta[["Host", "ProvOrig"]],
                    )

                    # Determine locations of all human and animal samples in the training set
                    only_human = np.where(y_train == "Human", True, False)
                    only_animal = np.where(y_train != "Human", True, False)

                    # Extract locations
                    animal = X_train[only_animal]
                    human = X_train[only_human]

                    # Random stratified subsample of human data
                    X_tr_h, _, y_tr_h, _ = train_test_split(
                        human,
                        y_train[only_human],
                        train_size=300,
                        random_state=i,
                        stratify=y_m_train.loc[only_human, ["ProvOrig"]],
                    )

                    # Prepare targets
                    y_tr_an = y_train[only_animal]

                    # Prepare dataset
                    X_re = np.vstack((X_tr_h, animal))
                    y_re = np.hstack((y_tr_h, y_tr_an))

                    # Divide dataset into Triglav subset and those excluded by Triglav
                    X_re_not = pd.DataFrame(X_re, columns=X.columns.values).loc[
                        :, not_col_subset
                    ]
                    X_test_not = pd.DataFrame(X_test, columns=X.columns.values).loc[
                        :, not_col_subset
                    ]

                    X_re = pd.DataFrame(X_re, columns=X.columns.values).loc[
                        :, col_subset
                    ]
                    X_test = pd.DataFrame(X_test, columns=X.columns.values).loc[
                        :, col_subset
                    ]

                else:
                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        y_m_train,
                        y_m_test,
                    ) = train_test_split(
                        X.values,
                        y.values,
                        y_meta,
                        test_size=0.2,
                        random_state=i,
                        stratify=y_meta[["Clade", "Host", "ProvOrig"]],
                    )

                    only_human = np.where(y_train == "Human", True, False)
                    only_animal = np.where(y_train != "Human", True, False)

                    animal = X_train[only_animal]
                    human = X_train[only_human]

                    X_tr_h, _, y_tr_h, _ = train_test_split(
                        human,
                        y_train[only_human],
                        train_size=600,
                        random_state=i,
                        stratify=y_m_train.loc[only_human, ["Clade", "ProvOrig"]],
                    )

                    y_tr_an = y_train[only_animal]

                    X_re = np.vstack((X_tr_h, animal))
                    y_re = np.hstack((y_tr_h, y_tr_an))

                    X_re_not = pd.DataFrame(X_re, columns=X.columns.values).loc[
                        :, not_col_subset
                    ]
                    X_test_not = pd.DataFrame(X_test, columns=X.columns.values).loc[
                        :, not_col_subset
                    ]

                    X_re = pd.DataFrame(X_re, columns=X.columns.values).loc[
                        :, col_subset
                    ]
                    X_test = pd.DataFrame(X_test, columns=X.columns.values).loc[
                        :, col_subset
                    ]

            else:
                # Subset by geographic region
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_m_train,
                    y_m_test,
                ) = train_test_split(
                    X.values,
                    y.values,
                    y_meta,
                    test_size=0.2,
                    random_state=i,
                    stratify=y_meta[["Host", "ProvOrig"]],
                )

                # Prepare dataset
                X_re = pd.DataFrame(X_train, columns=X.columns.values).loc[
                    :, col_subset
                ]
                y_re = y_train

                X_re_not = pd.DataFrame(X_train, columns=X.columns.values).loc[
                    :, not_col_subset
                ]
                X_test_not = pd.DataFrame(X_test, columns=X.columns.values).loc[
                    :, not_col_subset
                ]

                X_test = pd.DataFrame(X_test, columns=X.columns.values).loc[
                    :, col_subset
                ]

            # Train model, record scores, save best score and associated data
            model = LogisticRegressionCV(
                multi_class="multinomial", max_iter=2000, n_jobs=8
            ).fit(X_re, y_re)
            model_not = LogisticRegressionCV(
                multi_class="multinomial", max_iter=2000, n_jobs=8
            ).fit(X_re_not, y_re)

            score = balanced_accuracy_score(y_test, model.predict(X_test))
            score_not = balanced_accuracy_score(y_test, model_not.predict(X_test_not))

            score_f1 = f1_score(y_test, model.predict(X_test), average="macro")
            score_f1_not = f1_score(
                y_test, model_not.predict(X_test_not), average="macro"
            )

            print(score, score_not, score_f1, score_f1_not)

            scores_sel.append(score)
            scores_not.append(score_not)

            scores_f1_sel.append(score_f1)
            scores_f1_not.append(score_f1_not)

            if score > best_score:
                best_score = score
                best_cols = col_subset
                best_data = [(X_re, X_test, y_re, y_test)]

        # Statistical analysis
        from baycomp import two_on_single

        mu_sel = np.mean(scores_sel)
        std_sel = np.std(scores_sel, ddof=1)

        mu_not = np.mean(scores_not)
        std_not = np.std(scores_not, ddof=1)

        mu_sel_f = np.mean(scores_f1_sel)
        std_sel_f = np.std(scores_f1_sel, ddof=1)

        mu_not_f = np.mean(scores_f1_not)
        std_not_f = np.std(scores_f1_not, ddof=1)

        left, rope, right = two_on_single(
            np.asarray(scores_sel), np.asarray(scores_not), rope=0.05
        )
        left_f, rope_f, right_f = two_on_single(
            np.asarray(scores_f1_sel), np.asarray(scores_f1_not), rope=0.05
        )

        with open(
            "Final_Results/Triglav/%s_clf_perf_results.txt" % dataset_name,
            "w",
        ) as file:
            print(
                "Model (Selected Features)",
                "Model (Removed Features)",
                "Mean (Selected)",
                "Std Dev (Selected)",
                "Mean (Removed)",
                "Std Dev (Removed)",
                "P(S>R)",
                "P(S=R)",
                "P(S<R)",
                file=file,
            )
            print(
                "Balanced Accuracy",
                "Logistic Regression CV",
                "Logistic Regression CV",
                mu_sel,
                std_sel,
                mu_not,
                std_not,
                left,
                rope,
                right,
                file=file,
            )
            print(
                "F1-Score",
                "Logistic Regression CV",
                "Logistic Regression CV",
                mu_sel_f,
                std_sel_f,
                mu_not_f,
                std_not_f,
                left_f,
                rope_f,
                right_f,
                file=file,
            )

        # Get the best columns and divide into AA and NT mutations
        X_cols = best_cols
        X_cols_aa = np.asarray([x for x in X_cols if ":" in x])[1:]
        X_cols_nt = np.asarray([x for x in X_cols if ":" not in x])

        # Sub-sample based on host/province/(clade) for clustering
        if dataset_name != "21J":
            if dataset_name == "20C":
                # Get subset of samples associated with MI mink / B.1.641
                important_human = [
                    "hCoV-19/USA/MI-MDHHS-SC22125/2020",
                    "hCoV-19/USA/MI-MDHHS-SC22140/2020",
                    "hCoV-19/USA/MI-MDHHS-SC23517/2020",
                    "hCoV-19/USA/MI-MDHHS-SC22669/2020",
                    "hCoV-19/USA/MI-UM-10037594993/2020",
                    "hCoV-19/Canada/ON-PHL-21-44225/2021",
                ]

                important_human = np.asarray(important_human)

                # Get locations of remaining human samples and animal samples
                only_human = []
                for x in y_meta.values:
                    if x[0] == "Human" and x[1] not in important_human:
                        only_human.append(True)

                    else:
                        only_human.append(False)

                only_human = np.asarray(only_human)

                only_animal = np.where(y != "Human", True, False)

                # Get animal and human (not including MI samples) subset of X
                animal = X.loc[only_animal, best_cols]
                human = X.loc[only_human, best_cols]

                # Get random stratified sample of human samples (by geography)
                X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
                    human,
                    y[only_human],
                    train_size=300,
                    random_state=0,
                    stratify=y_meta.loc[only_human, ["ProvOrig"]],
                )

                # Prepare targets
                y_tr_an = y[only_animal]

                # Prepare data
                X_re = np.vstack((X_tr_h, animal, X.loc[important_human, best_cols]))
                y_re = np.hstack((y_tr_h, y_tr_an, y.loc[important_human]))

                idx = np.hstack(
                    (X_tr_h.index.values, animal.index.values, important_human)
                )

                X_re = pd.DataFrame(X_re, index=idx, columns=best_cols)
                y_re = pd.Series(y_re, index=idx)

            else:
                only_human = np.where(y == "Human", True, False)
                only_animal = np.where(y != "Human", True, False)

                animal = X.loc[only_animal, best_cols]
                human = X.loc[only_human, best_cols]

                X_tr_h, _, y_tr_h, _ = train_test_split(
                    human,
                    y[only_human],
                    train_size=600,
                    random_state=0,
                    stratify=y_meta.loc[only_human, ["Clade", "Host", "Province"]],
                )

                y_tr_an = y[only_animal]

                X_re = np.vstack((X_tr_h, animal))
                y_re = np.hstack((y_tr_h, y_tr_an))

                idx = np.hstack((X_tr_h.index.values, animal.index.values))

                X_re = pd.DataFrame(X_re, index=idx, columns=best_cols)
                y_re = pd.Series(y_re, index=idx)

        else:
            X_re = X.loc[:, X.columns.values[1:]][best_cols]
            y_re = y_meta["Host"]

        # Clustering and row dendrogram
        locs = np.sum(X_re, axis=0).values
        locs_final = []
        for x in locs:
            if 10 <= x <= (np.max(locs) - 5):
                locs_final.append(True)

            else:
                locs_final.append(False)

        X_final = X_re.loc[:, locs_final]
        X_final_cols_aa = set([x for x in X_final.columns.values if ":" in x])
        X_final_cols_nt = [x for x in X_final.columns.values if ":" not in x]

        X_aa = dict()
        for key in X_final_cols_aa:
            tmp = key.split(":")
            gene = tmp[0]
            loc = tmp[1][1:-1]
            tmp = "%s:%s" % (gene, loc)
            X_aa[tmp] = key

        from math import floor

        gene_positions = []
        translation_dict = {}
        translation_dict_inv = {}
        for mutation in X_final_cols_nt:
            loc = int(mutation[1:-1])

            gene_name = None

            if loc < 266:
                gene_name = "5'-UTR (%s)" % mutation

            elif 266 <= loc <= 13468:
                gene_name = "ORF1a"

                aa = floor((loc - 266) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 13468 <= loc <= 21555:
                gene_name = "ORF1b"

                aa = floor((loc - 13468) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 21563 <= loc <= 25384:
                gene_name = "S"

                aa = floor((loc - 21563) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 25393 <= loc <= 26220:
                gene_name = "ORF3a"

                aa = floor((loc - 25393) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 26245 <= loc <= 26472:
                gene_name = "E"

                aa = floor((loc - 26245) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 26523 <= loc <= 27191:
                gene_name = "M"

                aa = floor((loc - 26523) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 27202 <= loc <= 27387:
                gene_name = "ORF6"

                aa = floor((loc - 27202) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 27394 <= loc <= 27759:
                gene_name = "ORF7a"

                aa = floor((loc - 27394) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 27894 <= loc <= 28259:
                gene_name = "ORF8"

                aa = floor((loc - 27894) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 28274 <= loc <= 29533:
                gene_name = "N"

                aa = floor((loc - 28274) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif 29558 <= loc <= 29674:
                gene_name = "ORF10"

                aa = floor((loc - 29558) / 3) + 1

                gene_name = "%s:%d (%s)" % (gene_name, aa, mutation)

            elif loc > 29674:
                gene_name = "3'-UTR (%s)" % mutation

            else:
                gene_name = "Intergenic (%s)" % mutation

            gene_positions.append(gene_name)

            if gene_name not in translation_dict:
                translation_dict[gene_name] = []

            translation_dict[gene_name].append(mutation)
            translation_dict_inv[mutation] = gene_name

        X_cols_ss = []
        X_cols_final_lab = []
        for mutation in X_final.columns.values:
            if ":" in mutation:
                X_cols_ss.append(mutation)
                X_cols_final_lab.append(mutation)

                tmp = mutation.split(":")
                gene = tmp[0]
                pos = tmp[1][1:-1]

            else:
                tmp = translation_dict_inv[mutation].split(" ")[0]

                if tmp in X_aa:
                    # X_cols_final_lab.append(X_aa[tmp])
                    pass

                else:
                    X_cols_ss.append(mutation)

                    if mutation == "C29666T":
                        tmp = translation_dict_inv[mutation]
                        tmp = tmp.split(" ")
                        gene_name = "ORF10:L37F %s" % tmp[1]

                        X_cols_final_lab.append(gene_name)

                    else:
                        X_cols_final_lab.append(translation_dict_inv[mutation])

        X_final = X_final[X_cols_ss]
        X_final.columns = X_cols_final_lab

        # SAGE Scores Plots
        import sage as sg

        if dataset_name != "21J":
            y_tr_c = np.where(best_data[0][2] == "Human", 0, best_data[0][2])
            y_tr_c = np.where(y_tr_c == "Neovison vison", 1, y_tr_c)
            y_tr_c = np.where(y_tr_c == "Odocoileus virginianus", 2, y_tr_c).astype(int)

            y_te_c = np.where(best_data[0][3] == "Human", 0, best_data[0][3])
            y_te_c = np.where(y_te_c == "Neovison vison", 1, y_te_c)
            y_te_c = np.where(y_te_c == "Odocoileus virginianus", 2, y_te_c).astype(int)

        else:
            y_tr_c = np.where(best_data[0][2] == "Human", 0, best_data[0][2])
            y_tr_c = np.where(y_tr_c == "Odocoileus virginianus", 1, y_tr_c).astype(int)

            y_te_c = np.where(best_data[0][3] == "Human", 0, best_data[0][3])
            y_te_c = np.where(y_te_c == "Odocoileus virginianus", 1, y_te_c).astype(int)

        model = LogisticRegressionCV(
            multi_class="multinomial", max_iter=2000, n_jobs=8
        ).fit(best_data[0][0][X_cols_ss].values, y_tr_c)

        I = sg.MarginalImputer(model, best_data[0][0][X_cols_ss].values)
        E = sg.SignEstimator(I)

        if dataset_name != "21J":
            loc_human = np.where(y_te_c == 0, True, False)
            loc_mink = np.where(y_te_c == 1, True, False)
            loc_deer = np.where(y_te_c == 2, True, False)

            SV_H = E(
                best_data[0][1][X_cols_ss].values[loc_human],
                y_te_c[loc_human],
                bar=False,
            )
            SV_M = E(
                best_data[0][1][X_cols_ss].values[loc_mink], y_te_c[loc_mink], bar=False
            )
            SV_D = E(
                best_data[0][1][X_cols_ss].values[loc_deer], y_te_c[loc_deer], bar=False
            )

            SV_H.plot_sign(
                feature_names=X_cols_final_lab,
                title="Feature Importance - Human",
                max_features=20,
            )
            SV_M.plot_sign(
                feature_names=X_cols_final_lab,
                title="Feature Importance - Mink",
                max_features=20,
            )
            SV_D.plot_sign(
                feature_names=X_cols_final_lab,
                title="Feature Importance - Deer",
                max_features=20,
            )

        else:
            loc_human = np.where(y_te_c == 0, True, False)
            loc_deer = np.where(y_te_c == 1, True, False)

            SV_H = E(
                best_data[0][1][X_cols_ss].values[loc_human],
                y_te_c[loc_human],
                bar=False,
            )
            SV_D = E(
                best_data[0][1][X_cols_ss].values[loc_deer], y_te_c[loc_deer], bar=False
            )

            SV_H.plot_sign(
                feature_names=X_cols_final_lab,
                title="Feature Importance - Human",
                max_features=20,
            )
            SV_D.plot_sign(
                feature_names=X_cols_final_lab,
                title="Feature Importance - Deer",
                max_features=20,
            )

        # Cluster X_final and get cluster numbers
        from scipy.cluster.hierarchy import fcluster, to_tree, dendrogram

        g = sns.clustermap(X_final, metric="hamming", method="average")
        col_reorder = g.dendrogram_col.reordered_ind

        Z = g.dendrogram_row.linkage
        clus = fcluster(Z, 9, "maxclust")
        y_meta = y_meta.loc[X_re.index.values, :]
        y_meta["Cluster"] = clus

        # Print X_final to CSV
        X_cols_final_lab = np.asarray(X_cols_final_lab)[col_reorder]

        X_final = X_final[X_cols_final_lab]

        # Convert dendrograms to Newick format
        tree = to_tree(Z, False)
        tree_nwk = getNewick(tree, "", tree.dist, X_final.index.values)
        print_nwk(dataset_name, tree_nwk)

        # Save results
        for column in X_final.columns.values:
            if ":" in column:
                tmp = column.split(":")[0]
            else:
                tmp = column.split(" ")[0]

            data = np.where(X_final[column].values > 0, tmp, "-")
            X_final.loc[:, column] = data

        X_final.to_csv(
            "Final_Results/Triglav/X_triglav_%s.csv" % dataset_name
        )

        # Print metadata to CSV (Clusters were used for visualization purposes, not in final manuscript)
        cluster_data = y_meta["Cluster"].values.astype(str)
        cluster_data = np.where(cluster_data == "1", "Cluster One", cluster_data)
        cluster_data = np.where(cluster_data == "2", "Cluster Two", cluster_data)
        cluster_data = np.where(cluster_data == "3", "Cluster Three", cluster_data)
        cluster_data = np.where(cluster_data == "4", "Cluster Four", cluster_data)
        cluster_data = np.where(cluster_data == "5", "Cluster Five", cluster_data)
        cluster_data = np.where(cluster_data == "6", "Cluster Six", cluster_data)
        cluster_data = np.where(cluster_data == "7", "Cluster Seven", cluster_data)
        cluster_data = np.where(cluster_data == "8", "Cluster Eight", cluster_data)
        cluster_data = np.where(cluster_data == "9", "Cluster Nine", cluster_data)
        cluster_data = np.where(cluster_data == "10", "Cluster Ten", cluster_data)
        y_meta["Cluster"] = cluster_data
        y_meta["Sub-Division"] = y_meta["Province"].values
        y_meta.to_csv(
            "Final_Results/Triglav/y_meta_triglav_%s.csv" % dataset_name
        )

        fdfd = 5
