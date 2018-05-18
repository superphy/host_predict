#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import numpy as np
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
	

def create_master(rowenv,colenv):
	"""
	Create databases (lmdb) for row and column indices for the matrix
	"""
	p = Path('./results')
	rowindex = 0
	# For each fasta file in the results
	for filename in p.iterdir():
		# Get the genomeid from the filepath

		genomeid = filename.name

		#genomeid = str(filename)
		#genomeid = filename.split('/')[2] # Get filename.fa
		#genomeid = genomeid.split('.')[0] # Get filename






		# Fill in the row environment.
		with rowenv.begin(write=True) as txn:
			# key = GenomeID & value = row index; note that files
			# are passed in alphabetical order, so we can add the
			# index as we go.
			strindex = str(rowindex)
			txn.put(genomeid.encode('ascii'), strindex.encode('ascii'), overwrite=True)
			rowindex+=1
	
	# Fill in the column environment; note that sequences arent passed in
	# alphabetical order so we will have to initialize the index to 0 and
	# update it later.
	with colenv.begin(write=True) as txn:
		sequences={}
		# Walk through the master fasta file and keep track of the sequences
		# which we've already seen
		for seq_record in SeqIO.parse("master_fasta.fa", "fasta"):
			sequence = str(seq_record.seq).upper()
        	# If we haven't seen the sequence before, put it in the database
			if sequence not in sequences:
				sequences[sequence] = seq_record.id
				txn.put(sequence.encode('ascii'), '0'.encode('ascii'), overwrite=True)		

	# After all sequences have been inserted in alphabetical order, assign their index
		cursor = txn.cursor()
		for (index,(seq, value)) in enumerate(cursor):
			seq = seq.decode('utf-8')
			index = str(index)
			txn.put(seq.encode('ascii'), index.encode('ascii'), overwrite=True)


def create_matrix(rowenv,colenv):

	numrows = rowenv.stat()['entries']
	numrows = int(numrows)
	numcols = colenv.stat()['entries']
	numcols = int(numcols)

	kmermatrix = np.zeros((numrows,numcols))

	p = Path('./results')
	for filename in p.iterdir():
		# Get the genomeid from the filepath
		genomeid = filename.name

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
		kmermatrix[rowindex,:] = content
		
	# Save the matrix
	np.save('kmermatrix.npy', kmermatrix)


if __name__ == "__main__":
	"""
	Makes two environments for col and row lookup.
	Col stores kmer sequence and column number.
	Row stores genome ID and row number.
	"""
	# Create environments
	colenv = lmdb.Environment(b'coldb', max_dbs=1, map_size=5000000000)
	rowenv = lmdb.Environment(b'rowdb', max_dbs=1, map_size=5000000000)

	create_master(rowenv,colenv)
	create_matrix(rowenv,colenv)

	#print_env(rowenv)
	#print_env(colenv)
