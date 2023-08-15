# -*- coding: utf-8 -*-

import torch
import logging
import sys


def save_checkpoint(model, optimizer, filename):
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
    except:
        torch.save({'model': model.state_dict(), \
            'optimizer_tsvad': optimizer['tsvad'].state_dict(), \
            'optimizer_resnet': optimizer['resnet'].state_dict()}, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s', datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename)) 
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger
