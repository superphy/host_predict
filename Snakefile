ids, = glob_wildcards("genomes/{id}.fasta")

rule all:
  input:
    "touchfile.txt"

rule clean_input:
  output:
    "genomes/{id}.fasta"
  shell:
    "python clean.py ../SSRminiTest/genomes genomes/"

rule kmer_count:
  input:
    "genomes/{id}.fasta"
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

