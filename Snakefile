configfile: "config.yaml"

ruleorder: all > kmer_count > fa_dump > make_master > make_zarr

rule all:
   input:
       expand("results/human/.{human_samples}", human_samples=config["human_samples"]),
       expand("results/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"]),
       expand("test/human/.{human_samples}", human_samples=config["human_samples"]),
       expand("test/bovine/.{bovine_samples}", bovine_samples=config["bovine_samples"])
       

rule kmer_count:
   input:
       "data/{species}/{sample}.fasta"
   output:
       temp("results/{species}/{sample}.jf")
   #threads: 100
   shell:
       "jellyfish count -m 11 -s 100M -t {threads} {input} -o {output}"


rule fa_dump:
   input:
       "results/{species}/{sample}.jf"
   output:
       "results/{species}/{sample}.fa"
   #threads: 100
   shell:
       "jellyfish dump {input} > {output}"

rule make_master: 
   input:
       "results/{species}/{sample}.fa"
   output:
       touch("test/{species}/.{sample}")
   #threads: 1
   shell:
       "python createmaster.py {input}"
      
rule make_zarr: 
   input:
      a = "results/{species}/{sample}.fa",
      b = "test/{species}/.{sample}"
   output:
       touch("results/{species}/.{sample}")
   threads: 1000000000000
   shell:
       "python directload.py {input.a}"
