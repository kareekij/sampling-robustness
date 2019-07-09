from __future__ import print_function
import csv
import numpy as np
import log

dataset = 'undergrad_edges'
file = open('./log/' + dataset + '_n.txt', 'r')
rows = file.readlines()

output_fname = './log/' + dataset + '_rate.txt'


for i, row in enumerate(rows):
    if i == 0:
        tmp = row.replace('\n','').split(',')
        tmp.append('budget2')
    elif i != 0 and i != len(rows) - 1:
        rows_a = rows[i].replace('\n','').split(', ')
        rows_b = rows[i+1].replace('\n', '').split(', ')
        budget_a = int(rows_a[3])
        budget_b = int(rows_b[3])

        if budget_a > budget_b:
            continue

        a = np.array(rows_a, dtype="int")
        b = np.array(rows_b, dtype="int")
        tmp = (b - a).tolist()
        tmp.append(budget_a)

    log.save_to_file_line(output_fname, tmp)


file_adv = open('./log/' + dataset + '_rand_e_n.txt', 'r')
rows = file_adv.readlines()

output_adv_fname = './log/' + dataset + '_rate_adv.txt'

for i, row in enumerate(rows):
    if i != len(rows) - 1:
        rows_a = rows[i].replace('\n', '').split(', ')
        rows_b = rows[i+1].replace('\n', '').split(', ')
        budget_a = int(rows_a[0])
        budget_b = int(rows_b[0])
        p_a = float(rows_a[3])
        p_b = float(rows_b[3])
        diff = int(rows_b[1]) - int(rows_a[1])
        method = rows_a[2]

        if budget_a > budget_b or p_a != p_b:
            continue
        print('write {} to {} '.format(i, output_adv_fname))
        log.save_to_file_line(output_adv_fname, [str(budget_a), str(diff), method, str(p_a)])