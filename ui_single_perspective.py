import behaviordisc
import relationdisc
from forecasting import uni_forecast, multi_forecast
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import os

# thresholds to filter exogenous factors
PEAR_THRESHOLD = 0.5
GRANGER_THRESHOLD = 0.05


def init_res():
    res = dict()
    res['acf_img'] = None
    res['pacf_img'] = None
    res['period'] = None
    res['hm_granger_img'] = None
    res['hm_corr_img'] = None
    res['granger_causations'] = None
    res['granger_exog'] = None
    res['cp_img'] = None
    res['cp_val'] = None
    res['clust_img'] = None
    res['clust_clusters'] = None
    res['uni_forecast_img'] = None
    res['cp_type'] = None
    res['corr_type'] = None
    res['granger_type'] = None
    res['uni_model'] = None
    res['exog_factor'] = None

    return res


def calc_res(sd_log, aspect, checked_points):
    # seasonality, cp_pelt, cp_ks, forecasting, sub_clustering, granger, aspect, ks_win=100, ks_stat=40):
    aspect_points = sd_log.get_points(aspect)
    res = init_res()
    if checked_points['w_size'] == '' or checked_points['w_size'] is None:
        checked_points['w_size'] = 100
    else:
        checked_points['w_size'] = int(checked_points['w_size'])
    if checked_points['stat_size'] == '' or checked_points['stat_size'] is None:
        checked_points['stat_size'] = 40
    else:
        checked_points['stat_size'] = int(checked_points['stat_size'])

    # create acf_plot from arival rate
    if checked_points['check_season'] == 'on':
        # autocorrelation function
        ar_points = sd_log.get_points(sd_log.columns[0])
        plot_acf(ar_points)
        plt.savefig(os.path.join('static', 'images', 'acf_plot.png'))
        res['acf_img'] = os.path.join('static', 'images', 'acf_plot.png')

        # partial autocorellation function
        plot_pacf(ar_points)
        plt.savefig(os.path.join('static', 'images', 'pacf_plot.png'))
        res['pacf_img'] = os.path.join('static', 'images', 'pacf_plot.png')
        # get estimated period from sd_log
        res['period'] = sd_log.period

    if checked_points['granger'] == 'granger_linear':
        granger_hm_path = os.path.join('static', 'images', 'hm_granger_img.png')
        # create granger heatmap
        df_granger, relations, exogenous_factors = relationdisc.grangers_causation_matrix(sd_log=sd_log, plot=True,
                                                                                          save_hm=True,
                                                                                          outputpath=granger_hm_path)
        res['hm_granger_img'] = granger_hm_path
        res['granger_causations'] = relations
        res['granger_type'] = 'linear'
        res['granger_exog'] = None
    elif checked_points['granger'] == 'granger_non_linear':
        granger_hm_path = os.path.join('static', 'images', 'hm_granger_img.png')
        # create granger heatmap
        df_granger, relations, exogenous_factors = relationdisc.non_linear_granger_causation(sd_log=sd_log, plot=True,
                                                                                             save_hm=True,
                                                                                             outputpath=granger_hm_path)
        res['hm_granger_img'] = granger_hm_path
        res['granger_causations'] = relations
        res['granger_type'] = 'non-linear'
        res['granger_exog'] = None

    # correlation heatmap
    corr_hm_path = os.path.join('static', 'images', 'hm_corr_img.png')
    if checked_points['corr'] == 'pearson_corr':
        df_corr = relationdisc.corr_pearson(sd_log=sd_log, plot=True, save_hm=True, outputpath=corr_hm_path)
        res['hm_corr_img'] = corr_hm_path
        res['corr_type'] = 'Pearson'
    elif checked_points['corr'] == 'distance_corr':
        df_corr = relationdisc.corr_distance(sd_log=sd_log, plot=True, save_hm=True, outputpath=corr_hm_path)
        res['hm_corr_img'] = corr_hm_path
        res['corr_type'] = 'Distance'

    # check if aspect is exogenous factor
    if checked_points['corr'] and checked_points['granger']:
        # get aspects that have a significant correlation (above threshold) for selected aspect
        try:
            asp_corr = df_corr[df_corr[aspect] >= PEAR_THRESHOLD].index.values.tolist()
            asp_caus = df_granger[df_granger[aspect + '_x'] <= GRANGER_THRESHOLD].index.values.tolist()
        except:
            asp_corr = [1]
            asp_caus = []

        if asp_corr and asp_caus:
            res['exog_factor'] = False
        else:
            res['exog_factor'] = True

    # create changepoint plot
    cp_img_path = os.path.join('static', 'images', 'cp_img.png')
    if checked_points['cp_pelt'] == 'on':
        cp_values = behaviordisc.cp_detection_PELT(aspect_points, save_plot=True, outputpath=cp_img_path)
        res['cp_img'] = cp_img_path
        res['cp_val'] = cp_values
        res['cp_type'] = 'Pelt'
    elif checked_points['ks_test'] == 'on':
        cp_values = behaviordisc.cp_detection_KSWIN(aspect_points, window_size=checked_points['w_size'],
                                                    stat_size=checked_points['stat_size'],
                                                    save_plot=True, outputpath=cp_img_path)
        res['cp_img'] = cp_img_path
        res['cp_val'] = cp_values
        res['cp_type'] = 'Kolmogorov-Smirnov'

    # create subsequence plot
    if checked_points['sub_seq'] == 'on' and checked_points['cp_pelt']:
        clust_img_path = os.path.join('static', 'images', 'clust_img.png')
        clusters = behaviordisc.subseqeuence_clustering(aspect_points, cp_values, save_plot=True,
                                                        outputpath=clust_img_path, y_label=aspect)
        res['clust_img'] = clust_img_path  # path to cp img
        res['clust_clusters'] = clusters

    # forecasting (uni+multi)
    if checked_points['forecasting'] == 'on':
        uni_forecast_img_path = os.path.join('static', 'images', 'uni_for_img.png')
        forecast_val, model = uni_forecast(aspect_points, int(checked_points['forecast_n_period']), sd_log.period,
                                           save_plot=True, outputpath=uni_forecast_img_path)
        print(model)
        res['uni_model'] = str(model)
        res['uni_forecast_img'] = uni_forecast_img_path

        multi_forecast_img_path = os.path.join('static', 'images', 'mult_for_img.png')

        best_var = [key[1] for key, value in relations.items() if aspect == key[0]]
        best_var.insert(0, aspect)
        try:
            multi_forecast(sd_log=sd_log, variables=best_var, n_period=int(checked_points['forecast_n_period']),
                           save_plot=True, outputpath=multi_forecast_img_path)
        except:
            res['multi_forecast_img'] = multi_forecast_img_path

    return res
