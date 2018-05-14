from Bio import SeqIO
import sys
import lmdb
import zarr
import numpy as np

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

	with rowenv.begin(write=True) as txn:
		cursor = txn.cursor()
		for (index,(genomeid, value)) in enumerate(cursor):
			genomeid = genomeid.decode('utf-8')
			index = str(index)
			txn.put(genomeid.encode('ascii'), index.encode('ascii'), overwrite=True)

	with colenv.begin(write=True) as txn:
		cursor = txn.cursor()
		for (index,(seq, value)) in enumerate(cursor):
			seq = seq.decode('utf-8')
			index = str(index)
			txn.put(seq.encode('ascii'), index.encode('ascii'), overwrite=True)

	#print_env(rowenv)
	#print_env(colenv)

	# Open row and col environments for index lookup.
	#rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)
	#colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)

	# Find the total number of rows and columns for creating the .zarr matrix.
	numrows = rowenv.stat()['entries']
	numrows = int(numrows)
	numcols = colenv.stat()['entries']
	numcols = int(numcols)

	kmermatrix = np.zeros((numrows,numcols))
	np.save('kmermatrix.npy', kmermatrix)