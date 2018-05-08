configfile: "config.yaml"

rule all:
   input:
       expand("results/human/.{human_samples}", human_samples=config["human_samples"]),
       expand("results/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"]),

rule kmer_count:
   input:
       "data/{species}/{sample}.fasta"
   output:
       temp("results/{species}/{sample}.jf")
   threads:
       10
   shell:
       "jellyfish count -m 11 -s 100M -t {threads} {input} -o {output}"


rule fa_dump:
   input:
       "results/{species}/{sample}.jf"
   output:
       "results/{species}/{sample}.fa"
   shell:
       "jellyfish dump {input} > {output}"


rule fill_db: 
   input:
       "results/{species}/{sample}.fa"
   output:
       touch("results/{species}/.{sample}")
   shell:
       #"python lmdbfill.py {input}"
       "python createlmdb.py {input}"
