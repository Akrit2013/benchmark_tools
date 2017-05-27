#!/usr/bin/python

# This script evaluate the load speed of the lmdb and leveldb file

import getopt
import sys
import random
import glog as log
import lmdb_lib
import leveldb_lib
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
    backend = 'ser'
    db_type = 'lmdb'

    load_func = None
    iter_num = 1000

    help_msg = 'eval_db_read_speed -i <db> -n [num] --backend [backend] --db [dbtype]\n\
-i <db>             The input db\n\
-n [num]            The iter number, default is 1000.\n\
--backend [backend] The method of the serialize method, \
e.g.: datum, pickle, sel\n\
--db [dbtype]       The type of database, leveldb/lmdb'

    try:
        opts, args = getopt.getopt(argv, 'hi:n:', ['backend=', 'db='])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt == '-i':
            in_file = arg
        elif opt == '-n':
            iter_num = int(arg)
        elif opt == '--backend':
            backend = arg
        elif opt == '--db':
            db_type = arg

    if in_file is None:
        print help_msg
        sys.exit()

    serialize = serialize_lib.serialize_numpy()

    # Select the backend func
    if backend.lower() == 'pickle':
        load_func = pickle.loads
    elif backend.lower() == 'yaml':
        load_func = yaml.load
    elif backend.lower() == 'datum':
        load_func = caffe_tools.datum_str_to_array_im
    elif backend.lower() == 'ujson':
        load_func = ujson.loads
    elif backend.lower() == 'json':
        load_func = json.loads
    elif backend.lower() == 'ser':
        load_func = serialize.loads
    else:
        log.error('ERROR: Unknow backend')

    # Select the db type
    if db_type.lower() == 'lmdb':
        db_creater = lmdb_lib.lmdb
    elif db_type.lower() == 'leveldb':
        db_creater = leveldb_lib.leveldb

    # Init the db
    db = db_creater(in_file)
    db.set_val_parser(load_func)

    I('Get the key list')
    key_list = db.get_keylist()

    I('Start Testing...')
    bar = eb.EasyProgressBar()
    bar.set_end(iter_num)
    bar.start()

    timer = timer_lib.timer()
    timer.start()

    for idx in range(iter_num):
        randidx = random.randint(0, len(key_list))
        key = key_list[randidx]
        db.get(key)
        bar.update_once()

    bar.finish()
    timer.stop()
    I('Test %d iterations, Total time: %s' % (iter_num, timer.to_str()))


if __name__ == '__main__':
    main(sys.argv[1:])
