






import pandas as pd

if __name__ == '__main__':

	genome_list = []

	mic_df = pd.ExcelFile('amr_data/Updated_GenotypicAMR_Master.xlsx').parse('GenotypicAMR_Master') #you could add index_col=0 if there's an index
	df_rows = mic_df.index.values
	num_rows = len(df_rows)
	for index, row in mic_df.iterrows():
		if row["genus"] == "Escherichia":
			genome_list.append(row["run"])

	print(genome_list)
	print(len(genome_list))