ids, = glob_wildcards("genomes/{id}.fasta")

rule all:
  input:
    "touchfile.txt"

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


rule make_master: 
  input:
    expand("results/{id}.fa", id=ids)
  output:
    touch("touchfile.txt")
  shell:
   "python createmaster.py --threads {threads} {input}"


