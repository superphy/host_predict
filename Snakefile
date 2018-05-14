configfile: "config.yaml"

ruleorder: all > kmer_count > fa_dump > make_master > update_master > make_matrix

rule all:
  input:
    expand("results/human/.{human_samples}", human_samples=config["human_samples"]),
    expand("results/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"]),
    expand("touch/human/.{human_samples}", human_samples=config["human_samples"]),
    expand("touch/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"])


rule kmer_count:
   input:
       "data/{species}/{sample}.fasta"
   output:
       temp("results/{species}/{sample}.jf")
   shell:
       "jellyfish count -m 11 -s 100M {input} -o {output}"


rule fa_dump:
  input:
    "results/{species}/{sample}.jf"
  output:
    "results/{species}/{sample}.fa"
  shell:
    "jellyfish dump {input} > {output}"


rule make_master: 
  input:
    "results/{species}/{sample}.fa"
  output:
    touch("touch/{species}/.{sample}")
  shell:
    "python createmaster.py {input}"


rule update_master: 
  input:
    #"touch/human/.h-ECI-0008"
    #"touch/{species}/.{sample}"
    expand("touch/human/.{human_samples}", human_samples=config["human_samples"]),
    expand("touch/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"])
  output:
    #"touch/touch/{species}/.{sample}"
    touch("touch/touchfile")
  shell:
    "python updatemaster.py"


rule make_matrix: 
  input:
    a = "results/{species}/{sample}.fa",
    #b = "touch/touch/{species}/.{sample}",
    c = "touch/touchfile"
  output:
    touch("results/{species}/.{sample}")
  threads: 1000000000
  shell:
    "python directload.py {input.a}"
