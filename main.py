from behaviordisc import *
from relationdisc import *
from forecasting import *
from sdl import Sdl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def relation(sd_log):

    #k_means_clustering(sd_log)
    res1 = grangers_causation_matrix(sd_log)
    print(res1)
    corr_plot(sd_log)
    corr_distance2(sd_log)

    plt.show()


def behaviour(sd_log):
    # sd_log.plot_all_with_cp()
    sd_log.plot_all()

    ar_points = sd_log.get_points(sd_log.arrival_rate)
    tmp = cp_detection_window_based(ar_points)
    svt_points = sd_log.get_points(sd_log.service_time)
    np_points = sd_log.get_points(sd_log.num_in_process)
    decompostion_STL(ar_points, period=4, title=sd_log.arrival_rate)
    changepoints = cp_detection_KSWIN(svt_points, period=sd_log.tw)
    subseqeuence_clustering(svt_points, changepoints, y_label=sd_log.service_time)

    plt.show()
    print('End behaviour')


def forcasting(sd_log):

    x = sd_log.get_points(sd_log.arrival_rate)

    res = multi_forecast(sd_log,
                         [sd_log.arrival_rate, sd_log.finish_rate],
                         10)
    print(res)
    # plot_acf()
    res2 = uni_forecast(x, 7, sd_log.tw)
    # fc.test2()


    print('End forecasting')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'logs/BPI2017Active_1H_sdlog.csv'
    sd_log = Sdl(path)
    #behaviour(sd_log)
    relation(sd_log)
    #forcasting(sd_log)
