#!/usr/bin/env python

from Bio import SeqIO
import sys
import lmdb
import zarr

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

	filename = sys.argv[1]            # The filepath is given in the first argument
	genomeid = filename.split('/')[2] # Get filename.fa
	genomeid = genomeid.split('.')[0] # Get filename


	# Fill in the row environment.
	with rowenv.begin(write=True) as txn:
		# Add the genome ID as the key. Initially the value is 0.
		# Files aren't given to this script in alphabetical order,
		# so the value (row#) will be input later (updatemaster.py)
		txn.put(genomeid.encode('ascii'), '0'.encode('ascii'), overwrite=True)

	# Fill in the colenv
	with colenv.begin(write=True) as txn:
		for record in SeqIO.parse(filename, "fasta"):
			# Add the sequence as the key. Initially the value is 0.
			# Files aren't given to this script in alphabetical order,
			# so the value (col#) will be input later (updatemaster.py)
			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
			txn.put(kmerseq.encode('ascii'), '0'.encode('ascii'), overwrite=True)

	#print_env(rowenv)
	#print_env(colenv)


