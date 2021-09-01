import argparse

import sys

# Import the required libraries
import pandas as pd
import statistics
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import numpy as np

#Define some functions for plotting

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)


def plot_regression_line(x, b, axs):  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    axs.plot(x, y_pred, color = "g")


def main():
    """Console script."""
    parser = argparse.ArgumentParser('A CLI for analyzing Anscombe json files')
    # add the main argument of the cli
    parser.add_argument('user_input_file', help="filename for data file")
    # add an optional argument
    parser.add_argument('--show_plot', action='store_true', help="whether or not to show plots")
    args = parser.parse_args()

    if args.user_input_file:
        # Import the csv file
        df = pd.read_json(args.user_input_file)

        unique_series = df.Series.unique()

        fig, axs = plt.subplots(len(unique_series), figsize=(5,2.5*len(unique_series)))

        fig.suptitle('Anscombe Data')

        for i, series in enumerate(unique_series):
            # Convert pandas dataframe into pandas series
            list1 = df.loc[lambda df: df['Series'] == series, 'X']
            list2 = df.loc[lambda df: df['Series'] == series, 'Y']
            print(f"Series: {series}")
            # Calculating mean for x1
            print('X Mean: %.1f' % statistics.mean(list1))
              
            # Calculating standard deviation for x1
            print('X STDV: %.2f' % statistics.stdev(list1))
              
            # Calculating mean for y1
            print('Y Mean: %.1f' % statistics.mean(list2))
              
            # Calculating standard deviation for y1
            print('Y STDV: %.2f' % statistics.stdev(list2))
              
            # Calculating pearson correlation
            corr, _ = pearsonr(list1, list2)
            print('Pearson Correlation: %.3f\n\n' % corr)
            axs[i].scatter(list1, list2)
            b = estimate_coef(list1, list2)
            plot_regression_line(list1, b, axs[i])
    else:
        print('Please provide an input file')
        return 1
    if args.show_plot:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover