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

	z = zarr.load('kmermatrix.zarr')

	for i in range(33):
		print(i)
		print(z[i,:])