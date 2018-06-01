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

def find_recurring_char(record, start, end):
    length = len(record)
    window = end - start
    recur_char = 'X'
    perc_cut = 0.8
    A_count = (record).count("A",start,end)
    T_count = (record).count("T",start,end)
    G_count = (record).count("G",start,end)
    C_count = (record).count("C",start,end)
    N_count = (record).count("N",start,end)
    if(((A_count)/window)>perc_cut):
        recur_char = 'A'
    elif(((T_count)/window)>perc_cut):
        recur_char = 'T'
    elif(((G_count)/window)>perc_cut):
        recur_char = 'G'
    elif(((C_count)/window)>perc_cut):
        recur_char = 'C'
    elif(((N_count)/window)>perc_cut):
        recur_char = 'N'
    return recur_char

def format_files(files_list, output_dir):
    """
    Print to new directory the re-formatted fasta files.
    We will remove anything smaller than 500bp, under 5x coverage
    :param files_list: list of fasta files to format headers
    :return: success
    """
    max_score=25
    for f in files_list:
        file_name = f
        #with open(os.path.join(output_dir, file_name), "w") as oh:
        with open(os.path.join(output_dir, file_name.split('/')[-1]), "w") as oh:
            contig_number = 1
            with open(f, "r") as fh:
                node_counter = 0
                for record in SeqIO.parse(fh, "fasta"):
                    node_counter +=1
                    if len(record.seq) < 500:
                        #print("Skipping {}, less than 500bp".format(record.id), file=sys.stderr)
                        continue

                    m = re.search(r"_cov_([\d\.]+)", record.id)
                    if m:
                        if float(m.group(1)) < 5:
                            #print("Skipping {}, low coverage {}".format(record.id, m.group(1)), file=sys.stderr)
                            continue
                    #else:
                        #print("Could not find coverage for {}".format(record.id), file=sys.stderr)
                    length = len(record.seq)
                    str1 = record.seq
                    #print(record.description)
                    #searching end of file for garbage
                    recur_char ="X"
                    window_size =0
                    for i in range (30,310,20):
                        recur_char = find_recurring_char(record.seq,length-i, length)
                        if(recur_char != 'X'):
                            window_size = i
                            break
                    if(recur_char !='X'):
                        index = length-window_size+1
                        score = max_score
                        while(score != 0):
                            index -=1
                            curr_char = record.seq[index]
                            if(curr_char==recur_char and score != max_score):
                                score+=1
                            elif(curr_char!=recur_char):
                                score-=1
                            if(score == max_score):
                                window_size = length - index
                        str1 = record.seq[0:(length-window_size+1)]
                        #print("Deleting", length-len(str),"bases from node", node_counter, "in",f)
                    #searching front of file for garbage
                    length = len(str1)
                    recur_char ="X"
                    window_size =0
                    for i in range (30,310,20):
                        recur_char = find_recurring_char(record.seq,0,i)
                        if(recur_char != 'X'):
                            window_size = i
                            break
                    if(recur_char !='X'):
                        index = window_size-1
                        score = max_score
                        while(score != 0):
                            index +=1
                            curr_char = record.seq[index]
                            if(curr_char==recur_char and score != max_score):
                                score+=1
                            elif(curr_char!=recur_char):
                                score-=1
                            if(score == max_score):
                                window_size = index
                        str1 = record.seq[window_size+1:length-1]


                    record.description = record.description+"_filteredlen_"+str(len(str1))

                    record.seq=str1
                    SeqIO.write(record, oh, "fasta")


if __name__ == "__main__":
    all_files = get_files_to_analyze(sys.argv[1])
    format_files(all_files, sys.argv[2])