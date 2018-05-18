#!/usr/bin/env python

from Bio import SeqIO
from pathlib import *
import os
import sys
import lmdb


def print_env(env):
	"""
	Prints contents of a given environment.
	Used for testing.
	"""
	with env.begin() as txn:
		cursor = txn.cursor()
		for (index, (key, value)) in enumerate(cursor):
			test = value
			test.decode('ascii')
			print(index,(key, value))
	print(env.stat())
	

if __name__ == "__main__":
	"""
	Makes two environments for col and row lookup.
	Col stores kmer sequence and column number.
	Row stores genome ID and row number.
	"""
	# Create environments
	colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)
	rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)

	p = Path('./results')
	for filename in p.iterdir():
		# Get the genomeid from the filepath
		genomeid = os.path.basename(filename)


		rowindex = 0
		# Fill in the row environment.
		with rowenv.begin(write=True) as txn:
			# key = GenomeID & value = row index
			# Note that files passed in alphabetical order
			strindex = str(rowindex)
			txn.put(genomeid.encode('ascii'), strindex.encode('ascii'), overwrite=True)
			rowindex+=1

		# Fill in the colenv
		with colenv.begin(write=True) as txn:
			for record in SeqIO.parse(filename, "fasta"):
				# Add the sequence as the key. Initially the value is 0.
				# Files aren't given to this script in alphabetical order,
				# so the value (col#) will be input later (updatemaster.py)
				kmerseq = record.seq
				kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
				txn.put(kmerseq.encode('ascii'), '0'.encode('ascii'), overwrite=True)

	# After all sequences have been inserted in alphabetical order, assign their index
	with colenv.begin(write=True) as txn:
		cursor = txn.cursor()
		for (index,(seq, value)) in enumerate(cursor):
			seq = seq.decode('utf-8')
			index = str(index)
			txn.put(seq.encode('ascii'), index.encode('ascii'), overwrite=True)

'''
	# Initialize the kmermatrix
	numrows = rowenv.stat()['entries']
	numrows = int(numrows)
	numcols = colenv.stat()['entries']
	numcols = int(numcols)

	kmermatrix = np.zeros((numrows,numcols))
	np.save('kmermatrix.npy', kmermatrix)
'''

	#print_env(rowenv)
	#print_env(colenv)




