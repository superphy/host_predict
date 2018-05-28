#!/usr/bin/env python

import sys
import os
import re
from Bio import SeqIO

def get_files_to_analyze(file_or_directory):
    """
    :param file_or_directory: name of either file or directory of files
    :return: a list of all files (even if there is only one)
    """

    # This will be the list we send back
    files_list = []

    if os.path.isdir(file_or_directory):
        # Using a loop, we will go through every file in every directory
        # within the specified directory and create a list of the file names
        for root, dirs, files in os.walk(file_or_directory):
            for filename in files:
                files_list.append(os.path.join(root, filename))
    else:
        # We will just use the file given by the user
        files_list.append(os.path.abspath(file_or_directory))

    # We will sort the list when we send it back, so it always returns the
    # same order
    return sorted(files_list)

def format_files(files_list, output_dir):
    """
    Print to new directory the re-formatted fasta files.
    We will remove anything smaller than 500bp, and under 5x coverage
    :param files_list: list of fasta files to format headers
    :return: success
    """

    for f in files_list:
        file_name = f
        with open(os.path.join(output_dir, file_name), "w") as oh:
            contig_number = 1
            with open(f, "r") as fh:
                for record in SeqIO.parse(fh, "fasta"):
                    if len(record.seq) < 500:
                        print("Skipping {}, less than 500bp".format(record.id), file=sys.stderr)
                        continue

                    m = re.search(r"_cov_([\d\.]+)", record.id)
                    if m:
                        if float(m.group(1)) < 5:
                            print("Skipping {}, low coverage {}".format(record.id, m.group(1)), file=sys.stderr)
                            continue
                    else:
                        print("Could not find coverage for {}".format(record.id), file=sys.stderr)
                    SeqIO.write(record, oh, "fasta")


if __name__ == "__main__":
    all_files = get_files_to_analyze(sys.argv[1])
    format_files(all_files, sys.argv[2])
