#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import sys
import lmdb
import zarr
import numpy as np



if __name__ == "__main__":
	"""
	Given a .fa file, fills in its appropriate cells in a boolean .zarr matrix.
	Uses row and col environments created in createmaster.py.

	eg:
				AAAA	AAAC	...
	genome1		  1      0		...
	genome2       0      1		...
	...

	Note that the script readzarr.py will print out the contents of the matrix;
	use for testing.
	"""

	# Open row and col environments for index lookup.
	rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)
	colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)

	# Find the total number of rows and columns for creating the .zarr matrix.
	numrows = rowenv.stat()['entries']
	numrows = int(numrows)
	numcols = colenv.stat()['entries']
	numcols = int(numcols)


	# Define the .zarr matrix, mode is a = read & write.
	z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=(1,numcols), dtype='?')
	#z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=(1,numcols), dtype='?', synchronizer=zarr.ThreadSynchronizer())
	#z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=True, dtype='?', synchronizer=zarr.ThreadSynchronizer())

	# The path is in the form results/{species}/filename.fa
	filename = sys.argv[1]
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	# Using the genome ID, lookup the row# using rowenv.
	with rowenv.begin() as txn:
		rowindex = txn.get(genomeid.encode('ascii'))
		rowindex = rowindex.decode('utf-8')
		rowindex = int(rowindex)

	# Update the matrix after every 1,000,000 entries in the buffer.
	# Using a buffer and updating in batches increases efficiency.
	update_num = 1000000
	colbuffer = []

	# Can lookup the col# using a kmer sequence, using colenv.
	with colenv.begin() as txn:
		# Iterate through each record in the .fa
		for record in SeqIO.parse(filename, "fasta"):
			kmercount = record.id
			kmercount = int(kmercount)

			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)

			# Lookup col# from colenv, using the kmer sequence
			colindex = txn.get(kmerseq.encode('ascii'))
			colindex = colindex.decode('utf-8')
			colindex = int(colindex)

			#If the genome has the kmer, add it to the buffer for batch entry later
			if kmercount>0:
				colbuffer.append(colindex)

			# When the col buffer has 1,000,000 entires, do a batch update of the matrix
			s = len(colbuffer)
			if s >= update_num:
				z.set_coordinate_selection(([rowindex]*s, colbuffer), [1]*s)
				colbuffer = []
		# At the end of the file, batch update any remaining entries in the buffer
		s = len(colbuffer)	
		if s>0:
			z.set_coordinate_selection(([rowindex]*s, colbuffer), [1]*s)

