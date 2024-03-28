scaled_cells_avg_list = ['Trained_cells/scaled_cells_avg_output_0.txt', 'Trained_cells/scaled_cells_avg_output_1.txt',
                         'Trained_cells/scaled_cells_avg_output_2.txt', 'Trained_cells/scaled_cells_avg_output_3.txt',
                         'Trained_cells/scaled_cells_avg_output_4.txt']

scaled_cells_list = ['Trained_cells/scaled_cells_output_0.txt', 'Trained_cells/scaled_cells_output_1.txt',
                     'Trained_cells/scaled_cells_output_2.txt', 'Trained_cells/scaled_cells_output_3.txt',
                     'Trained_cells/scaled_cells_output_4.txt']

scaled_cells_mean_sub_list = ['Trained_cells/scaled_cells_mean_sub_output_0.txt', 'Trained_cells/scaled_cells_mean_sub_output_1.txt',
                              'Trained_cells/scaled_cells_mean_sub_output_2.txt', 'Trained_cells/scaled_cells_mean_sub_output_3.txt',
                              'Trained_cells/scaled_cells_mean_sub_output_4.txt']

scaled_cells_avg = [0.0]*6
scaled_cells = [0.0]*6
scaled_cells_mean_sub = [0.0]*6


scaled_lines_avg_list = ['Trained_lines/scaled_lines_avg_output_0.txt', 'Trained_lines/scaled_lines_avg_output_1.txt',
                         'Trained_lines/scaled_lines_avg_output_2.txt', 'Trained_lines/scaled_lines_avg_output_3.txt',
                         'Trained_lines/scaled_lines_avg_output_4.txt']

scaled_lines_list = ['Trained_lines/scaled_lines_output_0.txt', 'Trained_lines/scaled_lines_output_1.txt',
                     'Trained_lines/scaled_lines_output_2.txt', 'Trained_lines/scaled_lines_output_3.txt',
                     'Trained_lines/scaled_lines_output_4.txt']

scaled_lines_mean_sub_list = ['Trained_lines/scaled_lines_mean_sub_output_0.txt', 'Trained_lines/scaled_lines_mean_sub_output_1.txt',
                              'Trained_lines/scaled_lines_mean_sub_output_2.txt', 'Trained_lines/scaled_lines_mean_sub_output_3.txt',
                              'Trained_lines/scaled_lines_mean_sub_output_4.txt']

scaled_lines_avg = [0.0]*6
scaled_lines = [0.0]*6
scaled_lines_mean_sub = [0.0]*6

file_lists = [scaled_cells_list, scaled_cells_avg_list, scaled_cells_mean_sub_list, scaled_lines_list, scaled_lines_avg_list, scaled_lines_mean_sub_list]
save_lists = [scaled_cells_avg, scaled_cells, scaled_cells_mean_sub, scaled_lines_avg, scaled_lines, scaled_lines_mean_sub]

for pos, f in enumerate(file_lists):
    for ff in f:
        with open(ff, 'r') as f_open:
            for i in range(50):
                s = f_open.readline()
            s = f_open.readline().split()
            save_lists[pos][0] += float(s[3])/5.0
            save_lists[pos][1] += float(s[5])/5.0
            save_lists[pos][2] += float(s[7])/5.0
            save_lists[pos][3] += float(s[9])/5.0
            save_lists[pos][4] += float(s[11])/5.0
            save_lists[pos][5] += float(s[13])/5.0

for i in save_lists:
    print(i)
