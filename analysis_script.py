from pathlib import Path
from time import time

from normi13_qa.normi13 import Normi13
from argparse import ArgumentParser
    
if __name__ == "__main__":
    start = time()
    # Input arguments
    parser = ArgumentParser(description='Automated Normi-13 phantom image analysis')
    parser.add_argument('--data_path', type=Path, default='IMG00058.dcm')
    parser.add_argument('--save_path', type=Path, default='Results')
    parser.add_argument('--mtf_mode', type=str, choices=['relative', 'moments'], default='relative',
                        help='Algorithm for calculating either a relative or moment-based MTF.')
    parser.add_argument('--plot', type=bool, default=True, help='Plot results')
    arg = parser.parse_args()
    
    # Image analysis
    IQ = Normi13(arg.data_path, debug=arg.plot, plot=arg.plot, fig_path=arg.save_path, mtf_mode=arg.mtf_mode)
    IQ.analyze(visibility_threshold=0.0025)
    res = IQ.results_data(as_dict=True)
    end = time()

    # Show results
    print(IQ.results_data)
    print(f'Analysis done in {int((end - start) // 60)} minutes, {int((end - start) % 60)} seconds')
    