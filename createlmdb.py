#!/usr/bin/env python
from Bio import SeqIO
from Bio import SeqIO
from pathlib import Path
import sys
import lmdb


def fill_master(filename, env):
	"""
	Adds genome > kmer sequence > kmer count to the kmerdb
	Also adds each kmer sequence to the master db (no duplicates) with a count of 0
	"""
	master_db = env.open_db('Master'.encode('ascii'), dupsort=True)
	with env.begin(db=master_db, write=True) as txn:
		for record in SeqIO.parse(filename, "fasta"):
			kmercount = record.id
			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
			# Add the kmerseq to master db wiith kmercount 0; overwrite=True means no duplicates
			txn.put(kmerseq.encode('ascii'), '0'.encode('ascii'), overwrite=True)
			#print(txn.stat(master_db))
	return master_db


def fill_genome(filename, env):
	"""
	Adds genome > kmer sequence > kmer count to the kmerdb
	Also adds each kmer sequence to the master db (no duplicates) with a count of 0
	"""
	# The path is in the form results/{species}/filename.fa
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename
	genome_db = env.open_db(genomeid.encode('ascii'), dupsort=True)
	with env.begin(db=genome_db, write=True) as transaction:
		for record in SeqIO.parse(filename, "fasta"):
			kmercount = record.id
			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
			transaction.put(kmerseq.encode('ascii'), kmercount.encode('ascii'))
	return



if __name__ == "__main__":
	env = lmdb.Environment(b'kmerdb', max_dbs=100, map_size=5000000000)
	master_db = fill_master(sys.argv[1], env)
	fill_genome(sys.argv[1], env)

'''
# Prints the contents of the master db after it is created/filled
# Used to confirm lack of duplicates
	with env.begin(db=master_db) as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			test = value
			test.decode('ascii')
			print((key, value))
	print(env.stat())
'''



