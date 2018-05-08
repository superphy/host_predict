import lmdb
import zarr
import numpy as np



def create_matrix(env, master_db):
	# Define number of rows and columns for the matrix
	numrows = env.stat()['entries'] 
	numcols = 0
	with env.begin(write=False) as txn:
		numcols = txn.stat(master_db)['entries']

	# Initialize the matrix
	z = zarr.open('kmermatrix.zarr', mode='w', shape=(numrows,numcols), chunks=True, dtype='?')
	print(z[:])


	# Create a dict of row names
	rowdict = {}
	with env.begin(write=False) as txn:
		cursor = txn.cursor()
		for (index,(genome, value)) in enumerate(cursor):
			genome = genome.decode('utf-8')
			rowdict[index]=genome
	#print(rowdict)

	# Create a dict of col names
	coldict = {}
	with env.begin(db=master_db, write=False) as txn:
		cursor = txn.cursor()
		index = 0
		for (sequence, count) in cursor:
			coldict[sequence.decode('utf-8')]=index
			index+=1
	#print(coldict)

	#Fill in the aray
	# Iterate through the database
	with env.begin(write=False) as big_txn:
		big_cursor = big_txn.cursor()
		for (rowindex, (genome, value)) in enumerate(big_cursor):
			genome = genome.decode('utf-8')
			genome_db = env.open_db(genome.encode('ascii'))
			print(genome)
			# Iterate through genome's info
			with env.begin(db=genome_db, write=False) as sub_txn:
				sub_cursor = sub_txn.cursor()
				for kmerseq, kmercount in sub_cursor:
					kmercount = int(kmercount)
					kmerseq = kmerseq.decode('utf-8')
					colindex = coldict[kmerseq]
					if kmercount > 0: z[rowindex,colindex]=1


					#print((kmerseq,kmercount))
			#print(filecount) #output number of entries traversed (expected 34)
		#print(big_txn.stat(master_db))
	#print(env.stat())
	return



def traverse_lmdb(env, master_db):
	'''
	Iterates through the environment and iterating through each entry
	'''
	# Iterate through environment
	with env.begin(write=False) as big_txn:
		big_cursor = big_txn.cursor()
		filecount = 0
		for genome, value in big_cursor:
			genome = genome.decode('utf-8')
			genome_db = env.open_db(genome.encode('ascii'))
			# Iterate through entries
			with env.begin(db=genome_db, write=False) as sub_txn:
				sub_cursor = sub_txn.cursor()
				for kmerseq, kmercount in sub_cursor:
					kmercount = int(kmercount)
					#print((kmerseq,kmercount))
			filecount+=1
			#print(filecount) #output number of entries traversed (expected 34)
		#print(big_txn.stat(master_db))
	#print(env.stat())
	return filecount


if __name__ == "__main__":
	env = lmdb.Environment(b'kmerdb', max_dbs=100, map_size=5000000000)
	master_db = env.open_db('Master'.encode('ascii'))

	create_matrix(env, master_db)


	#check = traverse_lmdb(env, master_db)
	#print(check)