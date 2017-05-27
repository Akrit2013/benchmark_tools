#!/usr/bin/python

# This script evaluate the serialize method by loading the mats from
# a datalist, and serialize them.
# The time will be recorded and the string length will be accumulated

import getopt
import sys
import txt_tools
import matlab_tools
import path_tools
import glog as log
import cPickle as pickle
import timer_lib
import caffe_tools
import ujson
import json
import yaml
import serialize_lib
import easyprogressbar as eb


def main(argv):
    I = log.info
    in_file = None
    in_path = None
    mat_num = float('inf')
    backend = 'pickle'
    var_name = None

    dump_func = None
    load_func = None

    help_msg = 'eval_serialize -i <datalist> -v [var] -p [path] -n [num] -b [backend]\n\
-i <datalist>       The input datalist\n\
-v [var]            The var name in mat files\n\
-p [path]           The path to replace the original path of the datalist\n\
-n [num]            The number of the mat to be tested\n\
-b [backend]        The method of the serialize method, e.g.: yaml, pickle, \
datum, ujson, json, numpy'

    try:
        opts, args = getopt.getopt(argv, 'hi:v:p:n:b:')
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt == '-i':
            in_file = arg
        elif opt == '-v':
            var_name = arg
        elif opt == '-p':
            in_path = arg
        elif opt == '-n':
            mat_num = int(arg)
        elif opt == '-b':
            backend = arg

    if in_file is None:
        print help_msg
        sys.exit()

    serialize = serialize_lib.serialize_numpy()

    # Select the backend func
    if backend.lower() == 'pickle':
        dump_func = pickle.dumps
        load_func = pickle.loads
    elif backend.lower() == 'yaml':
        dump_func = yaml.dump
        load_func = yaml.load
    elif backend.lower() == 'datum':
        dump_func = caffe_tools.load_array_im_to_datum_str
        load_func = caffe_tools.datum_str_to_array_im
    elif backend.lower() == 'ujson':
        dump_func = ujson.dumps
        load_func = ujson.loads
    elif backend.lower() == 'json':
        dump_func = json.dumps
        load_func = json.loads
    elif backend.lower() == 'numpy':
        dump_func = serialize.dumps
        load_func = serialize.loads
    else:
        log.error('ERROR: Unknow backend')

    datalist = txt_tools.read_lines_from_txtfile(in_file)
    I('%s contains %d lines' % (in_file, len(datalist)))

    I('Loading the mat files ...')
    mat_list = []
    if mat_num < len(datalist):
        datalist = datalist[:mat_num]

    bar = eb.EasyProgressBar()
    bar.set_end(len(datalist))
    bar.start()
    for matpath in datalist:
        if in_path is not None:
            matpath = path_tools.replace_path(matpath, in_path)
        mat = matlab_tools.load_mat(matpath, var_name)
        mat_list.append(mat)
        bar.update_once()
    bar.finish()

    I('Start to serialize the mat files')
    bar.start()

    timer = timer_lib.timer()
    timer.start()
    str_list = []

    length_accum = 0

    for mat in mat_list:
        data_str = dump_func(mat)
        bar.update_once()
        str_list.append(data_str)
        length_accum += len(data_str)
    bar.finish()
    timer.stop()
    I('The serialize time for %s is %s, length is %d'
      % (backend, timer.to_str(), length_accum))

    I('Start to de-serialize the files')
    bar.start()

    timer.start()
    for data_str in str_list:
        mat = load_func(data_str)
        bar.update_once()
    bar.finish()
    timer.stop()
    I('The de-serialize time for %s is %s' % (backend, timer.to_str()))


if __name__ == '__main__':
    main(sys.argv[1:])
