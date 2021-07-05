from behaviordisc import *
from relationdisc import *
from forecasting import *
from sdl import Sdl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def relation(sd_log):

    #k_means_clustering(sd_log)
    non_linear_granger_causation(sd_log)
    df, relations, exogenous_factors = grangers_causation_matrix(sd_log)
    print(relations)
    print(exogenous_factors)
    corr_pearson(sd_log)
    corr_distance(sd_log)

    plt.show()


def behaviour(sd_log):
    sd_log.plot_all_with_cp()
    sd_log.plot_all()

    ar_points = sd_log.get_points(sd_log.avg_arrival_rate)
    changepoints = cp_detection_PELT(ar_points, show_plot=True)

    #svt_points = sd_log.get_points(sd_log.service_time)
    #np_points = sd_log.get_points(sd_log.num_in_process)
    #decompostion_STL(ar_points, period=7, title=sd_log.arrival_rate)
    #changepoints = cp_detection_KSWIN(ar_points, period=sd_log.tw)
    subseqeuence_clustering(ar_points, changepoints, y_label=sd_log.avg_arrival_rate)

    plt.show()
    print('End behaviour')


def forcasting(sd_log):
    res = multi_forecast(sd_log,
                         [sd_log.arrival_rate, sd_log.finish_rate],
                         7)
    #print(res)
    # plot_acf()
    x = sd_log.get_points(sd_log.num_in_process)
    res2 = uni_forecast(x, 14, sd_log.period)
    # fc.test2()


    print('End forecasting')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #path = 'logs/sd_logs/BPI2017Active_7D_sdlog.csv'
    path = "Outputs/act_vacc_1D/Checkout_sdlog.csv"
    path2 = "Outputs/act_vacc_1D/Vaccinate_sdlog.csv"
    #path = "Outputs/act_vacc_1D/Requestappointment_sdlog.csv"
    #path2 = "Outputs/act_vacc_1D/Appointmentgranted_sdlog.csv"
    sd_log1 = Sdl(path)
    sd_log2 = Sdl(path2)

    sd_log1.plot_all_with_cp()
    sd_log2.plot_all_with_cp()

    grangers_causation_matrix_2sdlogs(sd_log_one=sd_log1, sd_log_two=sd_log2, plot=True)
    corr_pearson(sd_log1, sd_log2, plot=True)
    #behaviour(sd_log)
    #stats_plot_pacf(sd_log.get_points(sd_log.columns[0]))
    #relation(sd_log)
    #forcasting(sd_log)


