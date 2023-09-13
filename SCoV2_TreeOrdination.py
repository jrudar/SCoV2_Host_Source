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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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

from TreeOrdination import TreeOrdination, CLRClosureTransformer

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

    # Do not convert to relative counts
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

    # Pick human, deer, and mink samples from Pickering et al. (2022)
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

    snv_sv_aa_analysis = False
    if snv_sv_aa_analysis:
        y = datasets["20C"][2][1]

        X_pa = datasets["20C"][2][0]
        col_info_pa = datasets["20C"][2][2]

        # Remove the Ontario Human Sequence - This will be our Query to see if other similar sequences can be detected
        y_no_ont = np.where(
            y["Strain"] != "hCoV-19/Canada/ON-PHL-21-44225/2021", True, False
        )
        y_ont = np.where(
            y["Strain"] == "hCoV-19/Canada/ON-PHL-21-44225/2021", True, False
        )

        # Get the embedding
        clf = TreeOrdination(
            feature_names=col_info_pa,
            proxy_model=ExtraTreesRegressor(
                16, n_jobs=1
            ),  # Not used for this experiment but required. Lower number just to speed up training of the model
            supervised_clf=ExtraTreesClassifier(
                16, n_jobs=1
            ),  # Not used for this experiment but required. Lower number just to speed up training of the model
            landmark_model=LANDMarkClassifier(
                n_estimators=128,
                use_nnet=False,
                etc_max_depth=8,
                minority_sz_lm=6,
                n_jobs=6,
            ),
            resampler=RandomUnderSampler(
                sampling_strategy={
                    "Human": 400,
                    "Neovison vison": 125,
                    "Odocoileus virginianus": 38,
                }
            ),
            n_iter_unsup=20,
            exclude_col=[False, -1],
            n_components=6,
        ).fit(X_pa[y_no_ont].astype(np.int8), y["Host"].values[y_no_ont])

        # High Dimensional Transformation
        X_query = clf.emb_transform(X_pa[y_ont], "UMAP")

        # Find nearest neighbors to unique human sequence from Ontario
        nn = NearestNeighbors(metric="euclidean").fit(clf.UMAP_emb)
        distance, idx = nn.kneighbors(X_query, 50)

        df_nn = y.values[y_no_ont][idx[0]]
        df_nn = np.vstack((df_nn, y.values[y_ont]))
        df_nn = pd.DataFrame(df_nn, columns=y.columns)
        df_nn.to_csv("Final_Results/TreeOrdination/20C_SNP_SV_AA.csv")

    spectrum_analysis = True
    if spectrum_analysis:
        y = datasets["20C"][0][1]

        X_pa = datasets["20C"][0][0]
        col_info_pa = datasets["20C"][0][2]

        # Remove the Ontario Human Sequence - This will be our Query to see if other similar sequences can be detected
        y_no_ont = np.where(
            y["Strain"] != "hCoV-19/Canada/ON-PHL-21-44225/2021", True, False
        )
        y_ont = np.where(
            y["Strain"] == "hCoV-19/Canada/ON-PHL-21-44225/2021", True, False
        )

        # Get the embedding
        clf = TreeOrdination(
            feature_names=col_info_pa,
            proxy_model=ExtraTreesRegressor(
                16, n_jobs=1
            ),  # Not used for this experiment but required. Lower number just to speed up training of the model
            supervised_clf=ExtraTreesClassifier(
                16, n_jobs=1
            ),  # Not used for this experiment but required. Lower number just to speed up training of the model
            landmark_model=LANDMarkClassifier(
                n_estimators=128,
                use_nnet=False,
                etc_max_depth=8,
                minority_sz_lm=6,
                n_jobs=6,
            ),
            resampler=RandomUnderSampler(
                sampling_strategy={
                    "Human": 400,
                    "Neovison vison": 125,
                    "Odocoileus virginianus": 38,
                }
            ), 
            n_iter_unsup=20,
            exclude_col=[False, -1],
            transformer = CLRClosureTransformer(),
            n_components=6,
        ).fit(X_pa[y_no_ont].astype(np.int8), y["Host"].values[y_no_ont])

        # High Dimensional Transformation
        X_query = clf.emb_transform(X_pa[y_ont], "UMAP")

        # Find nearest neighbors to unique human sequence from Ontario
        nn = NearestNeighbors(metric="euclidean").fit(clf.UMAP_emb)
        distance, idx = nn.kneighbors(X_query, 50)

        df_nn = y.values[y_no_ont][idx[0]]
        df_nn = np.vstack((df_nn, y.values[y_ont]))
        df_nn = pd.DataFrame(df_nn, columns=y.columns)
        df_nn.to_csv("Final_Results/TreeOrdination/20C_Spectrum.csv")

    prep_phylo = True
    if prep_phylo:
        #Much of the following code was taken and adapted from:
        #Pickering, B., Lung, O., Maguire, F. et al. Divergent SARS-CoV-2 variant emerges in white-tailed deer with deer-to-human transmission. Nat Microbiol 7, 2011–2024 (2022). https://doi.org/10.1038/s41564-022-01268-9
        #This can be found here: https://github.com/fmaguire/on_deer_spillback_analyses/tree/master/analyses/maximum_likelihood_phylogenies

        df_nn = pd.read_csv("Final_Results/TreeOrdination/20C_SNP_SV_AA.csv")
        df_nn.index = df_nn.Strain

        # Create a subset of data (Data From Figure 3 in Pickering et al. (2022) and NNs from above)
        df_gff = pd.read_table(
            "Deer GISAID/genemap.gff", comment="#", header=None
        )
        df_gff = df_gff.loc[:, [4, 5, 9]]
        df_gff.columns = ["start", "end", "gene"]
        df_gff.gene = df_gff.gene.str.replace("gene_name=", "").str.strip()
        gff = {row.gene: (row.start, row.end) for row in df_gff.itertuples()}

        import re
        from typing import Dict, Tuple, List, Set
        from collections import defaultdict

        regex_gene_aa_pos_simple = re.compile(r"(\w+):[a-zA-Z]+(\d+).*")
        regex_gene_aa_pos_insertion = re.compile(r"(\w+):(\d+).*")

        def split_mut(aamut: str):
            m = regex_gene_aa_pos_simple.match(aamut)
            if m:
                gene, aa_pos = m.groups()
                return gene, int(aa_pos)
            else:
                m = regex_gene_aa_pos_insertion.match(aamut)
                if m:
                    gene, aa_pos = m.groups()
                    return gene, int(aa_pos)
                else:
                    raise ValueError(
                        f"Could not parse gene and AA position from {aamut}"
                    )

        def get_gene_nt_coords(
            gff: Dict[str, Tuple[int, int]], gene: str, aa_pos: int
        ):
            if gene not in gff:
                raise ValueError(f"No gene {gene} in gene coordinate dict: {gff}")
            start, end = gff.get(gene)
            end_nt = aa_pos * 3 + start
            start_nt = end_nt - 3
            return start_nt, end_nt

        def range_str_to_int_tuple(nt_range: str):
            if "-" in nt_range:
                start, end = nt_range.split("-")
                return int(start), int(end)
            else:
                return int(nt_range), int(nt_range)

        def gene_ordered_mutations(muts) -> Dict[str, List[str]]:
            out = defaultdict(list)
            for m in muts:
                gene, mut = m.split(":", maxsplit=1)
                out[gene].append(mut)
            for k, v in out.items():
                v.sort(key=lambda x: int(re.sub(r"(\D+)?(\d+)\D+", r"\2", x)))
            return out

        def parse(nextclade_csv) -> pd.DataFrame:
            # df_nextclade = pd.read_table(nextclade_csv, sep=';', dtype=str, index_col=1)
            nextclade_csv.index = nextclade_csv["Strain"]
            aa_subs: pd.Series = nextclade_csv["aaSubstitutions"].str.split(",")
            aa_dels: pd.Series = nextclade_csv["aaDeletions"].str.split(",")
            aa_ins: pd.Series = nextclade_csv["aaInsertions"].str.split(",")
            nt_subs: pd.Series = nextclade_csv["ntSubstitutions"].str.split(",")
            sample_aas = sample_to_aa_mutations(aa_subs, aa_dels, aa_ins)
            sample_subs = sample_to_nt_mutations(nt_subs)
            samples = list(sample_aas.keys())
            unique_aas = get_sorted_aa_mutations(sample_aas)
            unique_subs = get_sorted_nt_subs(sample_subs)
            arr_aas = fill_aa_mutation_matrix(sample_aas, samples, unique_aas)
            arr_subs = fill_aa_mutation_matrix(sample_subs, samples, unique_subs)
            dfaa = pd.DataFrame(arr_aas, index=samples, columns=unique_aas)
            dfsubs = pd.DataFrame(arr_subs, index=samples, columns=unique_subs)
            return dfaa, dfsubs

        def sample_to_nt_mutations(nt_subs: pd.Series) -> Dict[str, Set[str]]:
            sample_subs = {}
            for sample, nt_sub in zip(nt_subs.index, nt_subs):
                subs = [] if not isinstance(nt_sub, list) else nt_sub
                sample_subs[sample] = set(subs)
            return sample_subs

        def sample_to_aa_mutations(
            aa_subs: pd.Series, aa_dels: pd.Series
        ) -> Dict[str, Set[str]]:
            sample_aas = {}
            for sample, aa_sub, aa_del in zip(aa_subs.index, aa_subs, aa_dels):
                aas = [] if not isinstance(aa_sub, list) else aa_sub
                aad = [] if not isinstance(aa_del, list) else aa_del
                sample_aas[sample] = set(aas) | set(aad)
            return sample_aas

        def fill_aa_mutation_matrix(
            sample_aas: Dict[str, Set[str]],
            samples: List[str],
            unique_aas: List[str],
        ) -> np.ndarray:
            """Fill AA mutation matrix with 1 when AA mutation present in sample"""
            arr_aas = np.zeros((len(sample_aas), len(unique_aas)), dtype="uint8")
            for i, sample in enumerate(samples):
                aas = sample_aas[sample]
                for j, aa in enumerate(unique_aas):
                    if aa in aas:
                        arr_aas[i, j] = 1
            return arr_aas

        def get_sorted_aa_mutations(sample_aas: Dict[str, Set[str]]) -> List[str]:
            unique_aas = set()
            for aas in sample_aas.values():
                unique_aas |= aas
            unique_aas = list(unique_aas)
            unique_aas.sort()
            return unique_aas

        def get_sorted_nt_subs(sample_subs: Dict[str, Set[str]]) -> List[str]:
            out = set()
            for subs in sample_subs.values():
                out |= subs
            out = list(out)
            out.sort()
            return out

        def sample_to_aa_mutations(
            aa_subs: pd.Series,
            aa_dels: pd.Series,
            aa_insertions: pd.Series,
        ) -> Dict[str, Set[str]]:
            sample_aas = {}
            for sample, aa_sub, aa_del, aa_ins in zip(
                aa_subs.index, aa_subs, aa_dels, aa_insertions
            ):
                if aa_sub == [""]:
                    aa_sub = np.NaN
                if aa_del == [""]:
                    aa_del = np.NaN
                if aa_ins == [""]:
                    aa_ins = np.NaN

                aas = aa_sub if isinstance(aa_sub, list) else []
                aad = aa_del if isinstance(aa_del, list) else []
                aai = aa_ins if isinstance(aa_ins, list) else []
                sample_aas[sample] = set(aas) | set(aad) | set(aai)
            return sample_aas

        aa_nt_muts = """
                        ORF1a:V23D (T333A),
                        ORF1a:E159A (A741C),
                        ORF1a:T265I (C1059T),
                        ORF1a:M297V (A1154G),
                        ORF1a:H325Q (T1240A),
                        ORF1a:T619S (C2121G),
                        ORF1a:T708I (C2388T),
                        ORF1a:A735 (T2463TA [FRAMESHIFT]),
                        ORF1a:D1289 (GA4130G [FRAMESHIFT]),
                        ORF1a:V1290A (T4134C),
                        ORF1a:A1314V (C4206T),
                        ORF1a:A1809 (G5690GA [FRAMESHIFT]),
                        ORF1a:L3116F (C9611T),
                        ORF1a:S3149F (CC9711TT),
                        ORF1a:K3353R (A10323G),
                        ORF1a:D3972V (A12180T),
                        ORF1a:V3976 (G12188GTT [FRAMESHIFT]),
                        ORF1a:S3983F (C12213T),
                        ORF1a:C4326R (T13241C),
                        ORF1a:M4390T (T13434C),
                        ORF1b:R524C (C15037T),
                        ORF1b:V1271L (G17278T),
                        ORF1b:M1693I (G18546T),
                        ORF1b:P1727S (C18646T),
                        ORF1b:I2303V (A20374G),
                        ORF1b:A2469del (GTGC20870G [disruptive_inframe_deletion]),
                        ORF1b:K2579R (A21203G),
                        S:F486L (T23020G),
                        S:N501T (A23064C),
                        S:D614G (A23403G),
                        S:S640F (C23481T),
                        S:S1003 (CA24566C [FRAMESHIFT]),
                        S:M1237 (GTA25269G [FRAMESHIFT]),
                        ORF3a:T12I (C25427T),
                        ORF3a:L219V (T26047G),
                        E:P71S (C26455T),
                        ORF8:D35Y (G27996T),
                        ORF8:E106* (G28209T),
                        N:P168S (C28775T),
                        N:S206P (T28889C),
                        N:T391I (C29445T),
                        ORF1a:S3983F (C12213T),
                        S:L1004S (T24573C),
                        S:A1070 (G24770GA [FRAMESHIFT]),
                        S:A1070E (C24771A),
                        S:Q1071K (C24773A),
                        S:E1072K (G24776A),
                        E:P71S (C26455T),
                        ORF8:E106* (G28209T),
                        N:P168S (C28775T),
                        ORF1a:T265I (C1059T),
                        ORF1a:Y369 (TA1370T [FRAMESHIFT]),
                        ORF1a:T708I (C2388T),
                        """.strip().split(
            ","
        )

        aa_nt_muts = [x.strip() for x in aa_nt_muts]

        for aant_mut in aa_nt_muts:
            if ":" not in aant_mut:
                continue
            print(aant_mut)
            aamut, ntmut = aant_mut.split(" ", maxsplit=1)
            gene, aapos = split_mut(aamut)
            print(gene, aapos)
            print(get_gene_nt_coords(gff, gene, aapos))

        missing_series = df_nn.missing.str.split(",").values
        for i, row in enumerate(missing_series):
            if row == [""]:
                missing_series[i] = np.NaN

        missing_series_w_sample = [
            (df_nn["Strain"].values[i], x) for i, x in enumerate(missing_series)
        ]

        sample_missing_regions = {}
        for row in missing_series_w_sample:
            sample = row[0]
            range_missing = row[1]
            if not isinstance(range_missing, list):
                print(f"sample {sample} has no missing regions")
                continue
            sample_missing_regions[sample] = [
                range_str_to_int_tuple(r) for r in range_missing
            ]

        dfaa, dfsubs = parse(df_nn)

        dfaa_sub = dfaa
        mut_mask = dfaa_sub.values.sum(axis=0) > 0
        dfaa_sub = dfaa_sub.loc[:, mut_mask]

        ordered_genes = """
        ORF1a
        ORF1b
        S
        ORF3a
        E
        M
        ORF7a
        ORF7b
        ORF8
        ORF9b
        N
        """.strip().split()

        gmuts = gene_ordered_mutations(dfaa_sub.columns)
        ordered_gmuts = [f"{g}:{x}" for g in ordered_genes for x in gmuts[g]]

        dfaa_sub = dfaa_sub[ordered_gmuts]

        for col in dfaa_sub:
            gene, _ = col.split(":", maxsplit=1)
            series = dfaa_sub[col]
            series[series == 0] = "-"
            series[series == 1] = gene

        for sample_full in dfaa_sub.index.values:
            sample = sample_full.split("|")[0]

            if sample in sample_missing_regions:
                for missing_ranges in sample_missing_regions[sample]:
                    for gmut in ordered_gmuts:
                        gene, aapos = split_mut(gmut)
                        start, end = get_gene_nt_coords(gff, gene, aapos)
                        missing_start = missing_ranges[0]
                        missing_end = missing_ranges[1]
                        if (
                            missing_end >= start >= missing_start
                            or missing_end >= end >= missing_start
                        ):
                            print(gmut)
                            print(start, end)
                            print(missing_start, missing_end)
                            print("=" * 80)
                            dfaa_sub.loc[sample, gmut] = "*No Coverage"

        dfaa_sub.to_csv("Final_Results/TreeOrdination/aa-matrix.tsv", sep="\t")

        df_loc = np.where(dfaa_sub.values != "-", True, False).sum(axis=0) >= 3

        df_sub_ss = dfaa_sub.values[:, df_loc]
        df_sub_ss = pd.DataFrame(
            df_sub_ss,
            index=dfaa_sub.index.values,
            columns=dfaa_sub.columns.values[df_loc],
        )
        df_sub_ss.to_csv("Final_Results/TreeOrdination/aa-matrix_sub.tsv", sep="\t")

        fasta_seq = {}
        from Bio import SeqIO

        with open("Deer GISAID/nextclade/all.fasta", "r") as file:
            for record in SeqIO.parse(file, format="fasta"):
                fasta_seq[record.id.split("|")[0]] = str(record.seq)

        sub_set = set(dfaa_sub.index.values)
        fasta_final = []
        host = []
        idx = 0
        for i, line in enumerate(sub_set):
            seq_id = line.split("|")[0]

            if seq_id in fasta_seq:
                host.append((seq_id, df_nn.loc[seq_id]["Host"]))

                fasta_final.append(">%s" % seq_id)
                fasta_final.append(fasta_seq[seq_id])

        host_series = pd.DataFrame(host, columns=["SeqID", "Host"])
        host_series.to_csv("Final_Results/TreeOrdination/subset_host.csv")

        dates = []
        with open("Final_Results/TreeOrdination/subset.fasta", "w") as file:
            for line in fasta_final:
                print(line, file=file)

        date_file = y.loc[y["Strain"].isin(host_series["SeqID"])][
            ["Strain", "Date"]
        ]
        date_file.to_csv("Final_Results/TreeOrdination/subset_dates.csv")

