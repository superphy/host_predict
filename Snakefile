
RAW_GENOMES_PATH = "genomes/raw/"
MIC_DATA_FILE = "amr_data/Updated_GenotypicAMR_Master.xlsx"

ids, = glob_wildcards(RAW_GENOMES_PATH+"{id}.fasta")

rule all:
  input:
    "touchfile.txt"

rule clean:
  input:
    RAW_GENOMES_PATH
  output:
    "genomes/clean/{id}.fasta"
  run:
    shell("python clean.py {input} genomes/clean")

rule kmer_count:
  input:
    expand("genomes/clean/{id}.fasta", id=ids)
  output:
    temp("results/{id}.jf")
  threads:
    2
  shell:
    "jellyfish count -m 11 -s 100M -t {threads} {input} -o {output}"

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
    shell("python create_matrix.py {input}")
    shell("python convert_dict.py")
    shell("python filter.py")
    shell("python bin_mics.py {MIC_DATA_FILE}")
    shell("python amr_prep.py")
    shell("python amr_split.py")