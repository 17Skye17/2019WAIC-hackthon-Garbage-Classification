import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='oxford5k,paris6k,roxford5k,rparis6k')
    parser.add_argument('--features', type=str, default='/tmp')
    args = parser.parse_args()
    return args
