from matplotlib import pyplot as plt
import numpy as np
plt.style.use('ostyle')


c_list = [
    "#f26241",
    "#f28444",
    "#7abfb8",
    "#205a8c",
    "#133654",
]


def get_avgs_and_weights(l_in):
    w_list = []
    avg_list = []
    err_list = []
    for cur_file in l_in:
        w_list.append(float(cur_file.split('/')[5]))
        cur_arr = np.genfromtxt(cur_file)
        avg_list.append(cur_arr.mean())
        err_list.append(cur_arr.std()/np.sqrt(len(cur_arr)))

    return w_list, avg_list, err_list


def main():

    # $RESULT_DIR/$Z3_LAT_SIZE/$Z3_WEIGHT/$Z3_NBINS/$Z3_NMEASURE/$Z3_NUPDATE/
    l4_file_list = [
        '/home/owen/z3_results/4/0.05/10000/500/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.1/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.2/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.3/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.5/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.6/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.8/10000/100/32/cluster_size_estimator.csv',
        '/home/owen/z3_results/4/0.9/10000/100/32/cluster_size_estimator.csv',
    ]
    l8_file_list = [
        '/home/owen/z3_results/8/0.05/10000/500/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.1/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.2/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.3/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.5/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.6/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.8/10000/100/128/cluster_size_estimator.csv',
        '/home/owen/z3_results/8/0.9/10000/100/128/cluster_size_estimator.csv',
    ]
    l16_file_list = [
        '/home/owen/z3_results/16/0.05/10000/1000/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.1/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.2/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.3/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.5/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.6/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.8/10000/500/512/cluster_size_estimator.csv',
        '/home/owen/z3_results/16/0.9/10000/500/512/cluster_size_estimator.csv',
    ]
    l24_file_list = [
        '/home/owen/z3_results/24/0.05/10000/1000/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.1/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.2/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.3/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.5/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.6/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.8/10000/500/1152/cluster_size_estimator.csv',
        '/home/owen/z3_results/24/0.9/10000/500/1152/cluster_size_estimator.csv',
    ]
    l32_file_list = [
        '/home/owen/z3_results/32/0.05/10000/1000/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.1/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.2/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.3/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.5/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.6/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.8/10000/500/2048/cluster_size_estimator.csv',
        '/home/owen/z3_results/32/0.9/10000/500/2048/cluster_size_estimator.csv',
    ]

    w, a, e = get_avgs_and_weights(l4_file_list)
    plt.errorbar(w, a, e, marker='o', c=c_list[0], label="L4")
    w, a, e = get_avgs_and_weights(l8_file_list)
    plt.errorbar(w, a, e, marker='o', c=c_list[1], label="L8")
    w, a, e = get_avgs_and_weights(l16_file_list)
    plt.errorbar(w, a, e, marker='o', c=c_list[2], label="L16")
    w, a, e = get_avgs_and_weights(l24_file_list)
    plt.errorbar(w, a, e, marker='o', c=c_list[3], label="L24")
    w, a, e = get_avgs_and_weights(l32_file_list)
    plt.errorbar(w, a, e, marker='o', c=c_list[4], label="L32")
    #plt.semilogy()
    #plt.show()
    plt.xlabel("Weights")
    plt.legend(loc='upper left')
    plt.ylabel("Size (In Number of Vertices)")
    plt.savefig('w_and_s.png', dpi=300)
    plt.tight_layout()
    exit()

    w_desired_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    l_list = [4, 8, 16, 24, 32]
    avg_list = [[], [], [], [], [], [], [], []]
    err_list = [[], [], [], [], [], [], [], []]
    for i, cur_w in enumerate(w_desired_list):
        w, a, e = get_avgs_and_weights(l4_file_list)
        for j in range(len(w)):
            if np.isclose(w[j], cur_w):
                avg_list[i].append(a[j])
                err_list[i].append(e[j])

        w, a, e = get_avgs_and_weights(l8_file_list)
        for j in range(len(w)):
            if np.isclose(w[j], cur_w):
                avg_list[i].append(a[j])
                err_list[i].append(e[j])

        w, a, e = get_avgs_and_weights(l16_file_list)
        for j in range(len(w)):
            if np.isclose(w[j], cur_w):
                avg_list[i].append(a[j])
                err_list[i].append(e[j])

        w, a, e = get_avgs_and_weights(l24_file_list)
        for j in range(len(w)):
            if np.isclose(w[j], cur_w):
                avg_list[i].append(a[j])
                err_list[i].append(e[j])

        w, a, e = get_avgs_and_weights(l32_file_list)
        for j in range(len(w)):
            if np.isclose(w[j], cur_w):
                avg_list[i].append(a[j])
                err_list[i].append(e[j])

    plt.errorbar(l_list, avg_list[0], err_list[0], c='r', label="w: {}".format(w_desired_list[0]))
    plt.errorbar(l_list, avg_list[1], err_list[1], c='b', label="w: {}".format(w_desired_list[1]))
    plt.errorbar(l_list, avg_list[2], err_list[2], c='g', label="w: {}".format(w_desired_list[2]))
    plt.errorbar(l_list, avg_list[3], err_list[3], c='y', label="w: {}".format(w_desired_list[3]))
    plt.errorbar(l_list, avg_list[4], err_list[4], c='k', label="w: {}".format(w_desired_list[4]))
    plt.errorbar(l_list, avg_list[5], err_list[5], c='c', label="w: {}".format(w_desired_list[5]))
    plt.errorbar(l_list, avg_list[6], err_list[6], c='c', label="w: {}".format(w_desired_list[6]))
    plt.errorbar(l_list, avg_list[7], err_list[7], c='c', label="w: {}".format(w_desired_list[7]))
    plt.legend()
    #plt.semilogy()
    plt.show()

if __name__ == "__main__":
    main()

