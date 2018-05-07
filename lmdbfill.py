#!/usr/bin/env python
from Bio import SeqIO
from Bio import SeqIO
from pathlib import Path
import sys
import lmdb


def file_len(filename):
	"""
	Return the number of lines in the file.
	"""
	count = 0
	with open(filename) as file:
		for i, l in enumerate(file):
			count += 1
	return count

"""
def db_fill(directory):
	#Currently prints filepath and filename for every file in directory
	#Not needed, done via snakefile
	pathlist = Path(directory).glob('**/*.fa')
	for path in pathlist:
		pathstr = str(path)
		file = pathstr.split('/')[2]
		print(pathstr, file)
		add_genome(pathstr)
"""

def addto_master(kmerseq):
	"""
	Called by add_genome.
	As the kmerdb is filled in, add sequences to a masterdb without duplicates
	A faster version is included in add_genome
	"""
	'''
	with env.begin(write=True) as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			if key == kmerseq:
				return
		txn.put(kmerseq.encode('ascii'), kmercount.encode('ascii'))
	'''

def update_all():

	master_env = lmdb.Environment(b'masterdb')
	kmer_env = lmdb.Environment(b'kmerdb')


	with master_env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			test = value
			test.decode('ascii')
			print((key, value))

	with kmer_env.begin() as txn:
		cursor = txn.cursor()
		for key, value in enumerate(cursor):
			test = value
			#test.decode('ascii')
			print(key, value)
			with kmer_env.begin(db=key.encode('ascii'), write=True) as transaction:
				cur = txn.cursor()
				for k, v in enumerate(cur):
					t = v
					#test.decode('ascii')
					print(k, v)

	return

def add_genome(filename):
	"""
	Adds genome > kmer sequence > kmer count to the kmerdb
	Also adds each kmer sequence to the master db (no duplicates) with a count of 0
	"""

	#path in form results/{species}/filename.fa
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	env = lmdb.Environment(b'kmerdb', max_dbs=5, map_size=5000000000)
	masterenv = lmdb.Environment(b'masterdb', max_dbs=5, map_size=5000000000)
	genome_db = env.open_db(genomeid.encode('ascii'), dupsort=True)
	#master_db = env.open_db('Master'.encode('ascii'), dupsort=True)
	with env.begin(db=genome_db, write=True) as transaction:
		for record in SeqIO.parse(filename, "fasta"):
			kmercount = record.id
			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)
			transaction.put(kmerseq.encode('ascii'), kmercount.encode('ascii'))
			#addto_master(kmerseq)


			# Add the kmerseq to master db wiith kmercount 0; overwrite=True means no duplicates
			with masterenv.begin(write=True) as txn:
			#with env.begin(db=master_db, write=True) as txn:
				txn.put(kmerseq.encode('ascii'), '0'.encode('ascii'), overwrite=True)

		#print_lmdb(env, genome_db, genomeid)


def print_lmdb(env, genome_db, genomeid):
	"""
	Given an environment, genome database, and genoome id, prints out
	the kmer sequences and kmer counts of the genome database
	"""
	with env.begin(db=genome_db) as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			test = value
			test.decode('ascii')
			print((key, value))  # (kmerseq, kmercount)


if __name__ == "__main__":
	add_genome(sys.argv[1])
	update_all()

'''
# Prints the contents of the master db after it is created/filled
# Used to confirm lack of duplicates
	env = lmdb.Environment(b'masterdb', max_dbs=5, map_size=5000000000)
	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			test = value
			test.decode('ascii')
			print((key, value))
'''



