# SCoV2_Host_Source
Sequence Signatures Within the Genome of SARS-CoV-2 Can be Used to Predict Host Source

This repository contains datasets and code which were used to generate the results reported in the main manuscript.
To use the code, you may need to update the paths located in each file. If there are issues or you need help using
the code, please contact Joe Rudar at joe.rudar@inspection.gc.ca.

# Files

SCoV2_Table_1_2_Supp1_10.py - This file runs the generalization performance analysis and will produce data for
Tables 1 and 2, and Supplementary Tables 1 to 10.

SCoV2_Selection_All_Features.py - This file will return a metadata file and a file of amino acid, SNP, and structural
variants for each sample. The output of this file is used later to generate figures and analyze the FUBAR/Triglav data.

SCoV2_Triglav.py - This file will run the Triglav analysis and return a CSV file containing data on which features were
selected across all Triglav runs.

SCoV2 Triglav Stats Subset.py - This file will run the analysis which looks at how well Triglav performed w.r.t the
rejected features and generate data to produce Figures 1 to 5.

SCoV2_TreeOrdination.py - This file runs the TreeOrdination/NN Search and will generate data to produce Figure 8.

# Directories

Deer_GISAID - This directory contains the raw data used for this analysis (sequences, GFF file, etc.).

Final_Results - This directory is used to store the output files. Currently, it only contains data which was
used to generate the results reported in the manuscript.

Selection_Results - This directory contains the data and trees used to generate Figures 5 and 7.
S
