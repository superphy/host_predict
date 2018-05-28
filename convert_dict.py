#!/usr/bin/env python

import numpy as np
from math import floor
import os

"""
Convert dictionaries of genomeid:row_index and kmerseq:col_index
into arrays, rows[row_index]=genomeid and cols[col_index]=kmerseq.
With testing this seemed faster than putting them in an array
during create_matrix.py.
"""
print("starting conversion")

# Load the matrix to get its dimensions
# Slow but better than explicitly stating the dimensions
matrix = np.load("unfiltered/kmer_matrix.npy")

# Get the dimensions of the matrix
num_rows = np.shape(matrix)[0]
num_cols = np.shape(matrix)[1]

# Load the dictionaries
kmer_rows = np.load("unfiltered/kmer_rows.npy")
kmer_cols = np.load("unfiltered/kmer_cols.npy")

# Prepare the new np arrays
row_names = np.empty([num_rows], dtype='S11')
col_names = np.empty([num_cols], dtype='S11')	

# Walk through row dictionary, place genome in correct index
i = 0
for key in kmer_rows.item():
	row_names[i] = key
	i+=1

# Walk through col dictionary, place sequence in correct index
j=0
for key in kmer_cols.item():
	col_names[j] = key
	j+=1

print("ending conversion\n")

# Save the np arrays
np.save('unfiltered/kmer_rows.npy', row_names)
np.save('unfiltered/kmer_cols.npy', col_names)


