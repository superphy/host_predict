#!/usr/bin/env python

from Bio import SeqIO
import sys
import lmdb


def print_env(env):
	"""
	Prints contents of a given environment
	"""
	with env.begin() as txn:
		cursor = txn.cursor()
		for (index, (key, value)) in enumerate(cursor):
			test = value
			test.decode('ascii')
			print(index,(key, value))
	print(env.stat())
	

if __name__ == "__main__":
	colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)
	rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)
	filename = sys.argv[1]
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	# create the rowenv
	with rowenv.begin(write=True) as txn:
		txn.put(genomeid.encode('ascii'), '0'.encode('ascii'), overwrite=True)
	# after all data is in the rowenv (in alphabetical order),
	# update the value of the key to be the index, for easy lookup later
	with rowenv.begin(write=True) as txn:
		cursor = txn.cursor()
		for (index,(genomeid, value)) in enumerate(cursor):
			genomeid = genomeid.decode('utf-8')
			index = str(index)
			txn.put(genomeid.encode('ascii'), index.encode('ascii'), overwrite=True)

	# create the colenv
	with colenv.begin(write=True) as txn:
		for record in SeqIO.parse(filename, "fasta"):
			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
			txn.put(kmerseq.encode('ascii'), '0'.encode('ascii'), overwrite=True)
			#print(txn.stat(master_db))
	# after all data is in the colenv (in alphabetical order),
	# update the value of the key to be the index, for easy lookup later
	with colenv.begin(write=True) as txn:
		cursor = txn.cursor()
		for (index,(seq, value)) in enumerate(cursor):
			seq = seq.decode('utf-8')
			index = str(index)
			txn.put(seq.encode('ascii'), index.encode('ascii'), overwrite=True)

	#print_env(rowenv)
	#print_env(colenv)


