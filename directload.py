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

	#z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=(1,numcols), dtype='?', synchronizer=zarr.ThreadSynchronizer())
	#z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=True, dtype='?', synchronizer=zarr.ThreadSynchronizer())
	z = zarr.open('kmermatrix.zarr', mode='a', shape=(numrows,numcols), chunks=(1,numcols), dtype='?')

	filename = sys.argv[1]

	# The path is in the form results/{species}/filename.fa
	genomeid = filename.split('/')[2] # get filename.fa
	genomeid = genomeid.split('.')[0] # get filename

	with rowenv.begin() as txn:
		rowindex = txn.get(genomeid.encode('ascii'))
		rowindex = rowindex.decode('utf-8')
		rowindex = int(rowindex)

	update_num = 1000000
	#current_num = 0
	colbuffer = []

	with colenv.begin() as txn:
		for record in SeqIO.parse(filename, "fasta"):
      
			kmercount = record.id
			kmercount = int(kmercount)

			kmerseq = record.seq
			kmerseq = kmerseq._get_seq_str_and_check_alphabet(kmerseq)

			colindex = txn.get(kmerseq.encode('ascii'))
			colindex = colindex.decode('utf-8')
			colindex = int(colindex)

			if kmercount>0:
				colbuffer.append(colindex)
				#print(colbuffer)

			s = len(colbuffer)

			if s >= update_num:
				#print(([rowindex*s, colbuffer]))
				z.set_coordinate_selection(([rowindex]*s, colbuffer), [1]*s)
				#z.set_coordinate_selection(colbuffer, [1]*s)
				colbuffer = []

		s = len(colbuffer)	
		if s>0:
			z.set_coordinate_selection(([rowindex]*s, colbuffer), [1]*s)

	#print(genomeid)
	#print(z[0,:])
      #print(genomeid,kmercount,kmerseq)
    #print(genomeid)

