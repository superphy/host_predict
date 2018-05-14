#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import sys
import lmdb
import numpy as np



if __name__ == "__main__":
	"""
	Fills a numpy matrix one row (genome) at a time.
	"""
	# Open environements to get row and col indices
	colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)
	rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)

	numcols = colenv.stat()['entries']
	numcols = int(numcols)

	# The path is in the form results/{species}/filename.fa
	filename = sys.argv[1]
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	# Using the genome ID, lookup the row# using rowenv.
	with rowenv.begin() as txn:
		rowindex = txn.get(genomeid.encode('ascii'))
		rowindex = rowindex.decode('utf-8')
		rowindex = int(rowindex)

	content = [0]*numcols
	with colenv.begin() as txn:
		for record in SeqIO.parse(filename, "fasta"):
			kmercount = record.id
			kmercount = int(kmercount)

			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)	

			colindex = txn.get(kmerseq.encode('ascii'))
			colindex = colindex.decode('utf-8')
			colindex = int(colindex)

			content[colindex] = kmercount

	# Open the numpy matrix and put the genome info into the correct row & save.
	kmermatrix = np.load('kmermatrix.npy')
	kmermatrix[rowindex,:] = content
	np.save('kmermatrix.npy', kmermatrix)



