"""
Event Log Object
Contains interesting behaviors. Should be used to make suggestion which sdLog to generate
Only accepts "ready event logs" - see PMSD Inside Event log - or xes files
"""

import json
import pandas as pd
import pm4py
import datetime

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import interval_lifecycle
from pm4py.statistics.sojourn_time.log import get as soj_time_get
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.attributes.log import get as attr_get
from pm4py.algo.organizational_mining.roles import algorithm as roles_discovery
from pm4py.visualization.sna import visualizer as sna_visualizer
from pm4py.algo.organizational_mining.sna import algorithm as sna


def create_handover_work(log):
    """
    @param log: event log
    @return: handover of work matrix
    """
    from pm4py.algo.organizational_mining.sna import algorithm as sna
    hw_values = sna.apply(log, variant=sna.Variants.HANDOVER_LOG)
    return hw_values


def get_handover(trans, roles):
    """
    @param trans: two activities which have a transition
    @param roles: roles detected in log by pm4py function
    @return: returns the resource role of the two activities in the transition
    """
    t1, t2 = trans
    t1_role = ''
    t2_role = ''
    for role in roles:
        if t1 in role[0]:
            t1_role = role[1]
        if t2 in role[0]:
            t2_role = role[1]
    return t1_role, t2_role


def check_for_lifecycle(log):
    """
    Checks if log contains a lifecycle (same start and complete timestamps)
    @param log: expects event log
    @return: boolean if event log has lifecylce
    """
    for trace in log:
        for event in trace:
            if event['time:timestamp'] != event['start_timestamp']:
                return False
    return True


def load_data(path):
    """
    Loads xes or csv file and converts it into interval with start and compelte timestamp
    Please note that only csv in "ready event logs" layout are accepted
    @param path: path to file
    @return: interval converted event log and original log as csv
    """

    log_format = path.split('.')[-1]

    if str(log_format) == 'csv':
        log_CSV = pd.read_csv(path)
        if {'Start Timestamp', 'Complete Timestamp', 'Activity', 'Resource', 'Case ID'}.issubset(log_CSV.columns):
            # convert timestamps into timestamp objects and convert resource and activity to strings
            log_CSV['Complete Timestamp'] = pd.to_datetime(log_CSV['Complete Timestamp'], errors='coerce')
            log_CSV['Start Timestamp'] = pd.to_datetime(log_CSV['Start Timestamp'], errors='coerce')
            log_CSV['Resource'] = log_CSV['Resource'].astype(str)
            log_CSV['Activity'] = log_CSV['Activity'].astype(str)
            # rename for columns for log_converter
            log_csv_re = log_CSV.rename(columns={'Case ID': 'case:concept:name',
                                                 'Complete Timestamp': 'time:timestamp',
                                                 'Start Timestamp': 'start_timestamp',
                                                 'Activity': 'concept:name',
                                                 'Resource': 'org:resource'})
            event_log = log_converter.apply(log_csv_re, variant=log_converter.Variants.TO_EVENT_LOG)
        else:
            raise Exception('Only accepts "ready event logs" CSV or XES, please see the PMSD Framework for format')

    elif str(log_format) == 'xes':
        event_log = pm4py.read_xes(path)
        log_CSV = None
    event_log = interval_lifecycle.to_interval(event_log)
    event_log = interval_lifecycle.to_lifecycle(event_log)
    event_log = interval_lifecycle.to_interval(event_log)
    return event_log, log_CSV


class El:
    """
    Class for event log to give an overview about interesting activities, resources and organizations
    """
    def __init__(self, filepath):
        self.log, self.log_csv = load_data(filepath)
        self.__first_timestamp = min(event['start_timestamp'] for trace in self.log for event in trace).timestamp()
        self.lifecycle = check_for_lifecycle(self.log)
        self.petri_net = None  # can be extended later

        self.soj_time = soj_time_get.apply(self.log,
                                           parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp",
                                                       soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp"})

        self.dfg = dfg_discovery.apply(self.log,
                                       parameters={dfg_discovery.Parameters.START_TIMESTAMP_KEY: "start_timestamp",
                                                   dfg_discovery.Parameters.TIMESTAMP_KEY: "time:timestamp"})
        self.dfg_perf = dfg_discovery.apply(self.log, variant=dfg_discovery.Variants.PERFORMANCE,
                                            parameters={dfg_discovery.Parameters.START_TIMESTAMP_KEY: "start_timestamp",
                                                        dfg_discovery.Parameters.TIMESTAMP_KEY: "time:timestamp"})

        # self.hw_values = sna.apply(self.log, variant=sna.Variants.HANDOVER_LOG)  # handover of work
        # self.sub_values = sna.apply(self.log, variant=sna.Variants.SUBCONTRACTING_LOG)  # subcontracting
        # self.ja_values = sna.apply(self.log, variant=sna.Variants.JOINTACTIVITIES_LOG)  # similar activities
        # self.wt_values = sna.apply(self.log, variant=sna.Variants.WORKING_TOGETHER_LOG)  # working together

        self.roles = roles_discovery.apply(self.log)

        self.act_recommendation = None
        self.trans_recommendation = None
        self.res_recommendation = None  # TODO add recommendations for resources/ organizations
        self.make_recommendation()

        # self.__best_time_window =

    def get_earliest_timestamp(self):
        return self.__start_timestamp

    def show_dfg(self, variant='Frequency'):
        from pm4py.visualization.dfg import visualizer as dfg_visualization
        if variant == 'Frequency':
            gviz = dfg_visualization.apply(self.dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        elif variant == 'Performance':
            gviz = dfg_visualization.apply(self.dfg_perf, log=self.log, variant=dfg_visualization.Variants.PERFORMANCE)
        else:
            raise Exception('Variant not supported. Choose Frequency or Performance')

        dfg_visualization.view(gviz)

    def show_handover(self):
        hw_values = self.hw_values
        gviz_hw_py = sna_visualizer.apply(hw_values, variant=sna_visualizer.Variants.PYVIS)
        sna_visualizer.view(gviz_hw_py, variant=sna_visualizer.Variants.PYVIS)

    def make_recommendation(self):
        act_recommendation = dict()
        activities_count = attr_get.get_attribute_values(self.log, 'concept:name')
        for act in self.soj_time:
            recom_entry = dict()
            recom_entry['type'] = 'bottleneck'
            recom_entry['mean_duration'] = datetime.timedelta(seconds=self.soj_time[act])
            recom_entry['total_duration'] = datetime.timedelta(seconds=(self.soj_time[act] * activities_count[act]))
            recom_entry['Resource Role'] = [role[1] for role in self.roles if act in role[0]]
            act_recommendation[act] = recom_entry
        self.act_recommendation = act_recommendation

        trans_recommendation = dict()
        for trans in self.dfg_perf:
            recom_entry = dict()
            recom_entry['type'] = 'bottleneck'
            recom_entry['mean_duration'] = datetime.timedelta(seconds=self.dfg_perf[trans])
            recom_entry['total_duration'] = datetime.timedelta(seconds=(self.dfg_perf[trans] * self.dfg[trans]))
            recom_entry['resource handover'] = get_handover(trans, self.roles)
            trans_recommendation[str(trans)] = recom_entry
        self.trans_recommendation = trans_recommendation

        res_recommendation = dict()
        recom_entry = dict()
        recom_entry['type'] = 'social network'
        recom_entry['role'] = self.roles
        # recom_entry['similar activities'] = None
        # recom_entry['working together'] = None
        # recom_entry['similar activities'] = None
        res_recommendation['organizational mining'] = recom_entry
        self.res_recommendation = res_recommendation

    def summary(self):
        """
        Prints a summary about detected bottlenecks in activities respectively transitions
        with corresponding resources in social networks
        """
        print('Activities: \n')
        print(json.dumps(self.act_recommendation, sort_keys=True, indent=4, default=str))
        # print bottleneck transitions
        print('\n')
        print('Transitions: \n')
        print(json.dumps(self.trans_recommendation, sort_keys=True, indent=4, default=str))
        # print organizational mining
        print('\n')
        print('Organizational mining: \n')
        print(json.dumps(self.res_recommendation, sort_keys=True, indent=4, default=str))


# path = "../PMSD/Outputs/ready_event_log.csv"
# path = "C:/Users/Firas/Downloads/BPI Challenge 2017.xes"
# path = "C:/Users/Firas/OneDrive - rwth-aachen.de/Desktop/Uni/Masterthesis/PMSD/SampleEventLog.csv"
# path = "C:/Users/Firas/OneDrive - rwth-aachen.de/Desktop/Uni/Masterthesis/PMSD/running-example.xes"
# el = El(path)
# el.summary()

