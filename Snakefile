
#################################################################

# Location of the raw genomes
RAW_GENOMES_PATH = "/mnt/moria/rylan/phenores_data/raw/genomes/"
#RAW_GENOMES_PATH = "genomes/raw/"

# Location of the MIC data file (excel spreadsheet)
MIC_DATA_FILE = "amr_data/Updated_GenotypicAMR_Master.xlsx" # location of MIC data file

# The number of input genomes. The number of rows must match the
# nubmer of rows in the MIC data file. The names of the genomes
# must also be consistent, but need not be in the same order.
NUM_INPUT_FILES = 2552

# Kmer length that you want to count 
KMER_SIZE = 11

# Data type of the resulting kmer matrix. Use uint8 if counts are
# all under 256. Else use uint16 (kmer counts under 65536)
MATRIX_DTYPE = 'uint8'

#################################################################

ids, = glob_wildcards(RAW_GENOMES_PATH+"{id}.fasta")

rule all:
  input:
    "touchfile.txt"

#rule clean:
#  input:
#    RAW_GENOMES_PATH
#  output:
#    "genomes/clean/{id}.fasta"
#  run:
#    shell("python clean.py {input} genomes/clean/")

rule kmer_count:
  input:
    #expand("genomes/clean/{id}.fasta", id=ids)
    "genomes/clean/{id}.fasta"
  output:
    temp("results/{id}.jf")
  threads:
    2
  shell:
    "jellyfish count -m {KMER_SIZE} -s 100M -t {threads} {input} -o {output}"

rule fa_dump:
  input:
    "results/{id}.jf"
  output:
    "results/{id}.fa"
  shell:
    "jellyfish dump {input} > {output}"

rule make_matrix: 
  input:
    expand("results/{id}.fa", id=ids)
  output:
    touch("touchfile.txt")
  run:
    #shell("python tally.py")
    shell("python parallel_matrix.py {NUM_INPUT_FILES} {KMER_SIZE} {MATRIX_DTYPE} {input}")
    shell("python convert_dict.py")
    shell("python filter.py")
    shell("python bin_mics.py {MIC_DATA_FILE}")
    shell("python amr_prep.py")
    shell("python amr_split.py")