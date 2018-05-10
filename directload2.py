#!/usr/bin/env python

from Bio import SeqIO
from pathlib import Path
import sys
import lmdb
import zarr
import numpy as np

if __name__ == "__main__":

	rowenv = lmdb.Environment(b'rowdb', max_dbs=100, map_size=5000000000)
	colenv = lmdb.Environment(b'coldb', max_dbs=100, map_size=5000000000)

	numrows = rowenv.stat()['entries']
	numrows = int(numrows)
	numcols = colenv.stat()['entries']
	numcols = int(numcols)


	#z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=True, dtype='?')
	z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=(1,numcols), dtype='?', synchronizer=zarr.ThreadSynchronizer())


	filename = sys.argv[1]

	# The path is in the form results/{species}/filename.fa
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	with rowenv.begin() as txn:
		#print(genomeid, type(genomeid))
		rowindex = txn.get(genomeid.encode('ascii'))
		rowindex = rowindex.decode('utf-8')
		rowindex = int(rowindex)

	with colenv.begin() as txn:
		for record in SeqIO.parse(filename, "fasta"):
			
			kmercount = record.id
			kmercount = int(kmercount)

			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)

			colindex = txn.get(kmerseq.encode('ascii'))
			colindex = colindex.decode('utf-8')
			colindex = int(colindex)

			#print(rowindex,colindex, kmercount)
			if kmercount>0:
				z[rowindex,colindex]=True
				#print(z[rowindex,colindex])
			#print(genomeid,kmercount,kmerseq)
		#print(genomeid)
	#for i in range(33):
		#rint(genomeid, rowindex)
		#print(z[i,:])
      #print(genomeid,kmercount,kmerseq)
    #print(genomeid)

