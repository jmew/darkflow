from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet
import time

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)
    log_file = open('log_file.txt', 'wb')

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        while True:
            try:
                print('Enter training ...'); tfnet.train()
            except Exception as e:
                print(e.message)
                if "NaN" in str(e.message):
                    FLAGS.load = int(str(e.message)[4:])
                    tfnet = TFNet(FLAGS)
                    print('NaN Exception')
                    output = str(time.strftime("%c")) + ': NaN exception at ckpt - %d' + '\n'
                    log_file.write(output)

            if not FLAGS.savepb:
                exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()
