__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from Scripts.Scripts import *
import datetime
from scipy import signal
import cv2 as cv
import numpy as np
import os
import sys
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


################################
######   General Scripts  ######
################################

# Get size of variables in MB
def get_size_of(*var_list):
    total_size = 0
    size_list = ""
    for var in var_list:
        size_list += str(sys.getsizeof(var)//10e6) + 'MB, '
        total_size += sys.getsizeof(var)
    if len(var_list) > 1:
        print('# Total Size =',total_size//10e6,'MB ({})'.format(size_list[:-2]))
    else:
        print('# Total Size =',total_size//10e6,'MB')
    return

# Show Progress Bar for the work
def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    try:
        # You can't have a progress bar with zero or negative length.
        if BarLength <1:
            BarLength = 20
        # Use status variable for going to the next line after progress completion.
        Status = ""
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.:
            Progress = 1
            Status = "\r\n"    # Going to the next line
        # Calculating how many places should be filled
        Block = int(round(BarLength * Progress))
        # Show this
        Bar = "[{}] {:.0f}% {}".format(ProgressIcon * Block + BarIcon * (BarLength - Block), round(Progress * 100, 0), Status)
        return Bar
    except:
        return "ERROR"
def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


# Count time deltas between recall and recall
def runtime_clock(min_time_threshold = 1e-5):
    if not hasattr(runtime_clock, "clock_buffer"):
        runtime_clock.clock_buffer = datetime.datetime.now()  # it doesn't exist yet, so initialize it
    click = datetime.datetime.now()
    delta_sec = (click - runtime_clock.clock_buffer).seconds
    if delta_sec < min_time_threshold:
        runtime_clock.clock_buffer = click
        return

    if delta_sec > 3600:
        print('@ Process Time: {} Hours and {} Minutes and {} Seconds'.format(str(delta_sec//3600),str(delta_sec%3600//60),str(delta_sec%60)) )
    elif delta_sec >= 60:
        print('@ Process Time: {} Minutes and {} Seconds'.format(str(delta_sec//60),str(delta_sec%60)) )
    else:
        print('@ Process Time: {} Seconds'.format(str( delta_sec )) )
    runtime_clock.clock_buffer = click
    return delta_sec
runtime_clock() # First run to start counting



# colored text and background
class colors:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'



def create_folder(path, over_write = False):
    import os
    if not os.path.exists(path):
        print('Create ' + path + ' folder')
        os.mkdir(path)
        # os.system('mkdir ' + path)
    elif over_write:
        print('Delete ' + path + ' folder')
        # os.system('rm -r ' + path)
        os.system('del -r ' + path)
        print('Create ' + path + ' folder')
        os.mkdir(path)
        # os.system('mkdir ' + path)

    return



def write_to_log_file():
    # global logfile
    # class logfile():
    #     def __init__(self, filename):
    #         old_stdout = sys.stdout
    #         log_file = open(filename,"w")
    #         sys.stdout = log_file
    #         return
    #
    #     def __del__(self):
    #         sys.stdout = old_stdout
    #         log_file.close()
    #         return
    # logfile("message.log")

    log_folder_path = 'C:/Users/khen/PycharmProjects/dvs/log_files/'
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_folder_path+os.path.basename(sys.argv[0])[:-3]+'.log', 'w'))
    global print
    print = logger.info
    print('Log File')
    return
# write_to_log_file()
# print('test')



#######################################
######   Neural Network Scripts  ######
#######################################

# def show_network(model):
#     from tabulate import tabulate
#     print(tabulate([[name, param.size()] for name, param in model.named_parameters() if param.requires_grad], headers=['Name', 'Size'], tablefmt='orgtbl'))
#     return