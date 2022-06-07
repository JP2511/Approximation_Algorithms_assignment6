import copy
import hyperloglog

import numpy as np

from typing import Generator

###############################################################################

def read_file(filename: str) -> Generator:
    """Reads a file line by line, while removing newline characters.

    Args:
        filename (str): path/name of the file whose lines are to be read.

    Yields:
        Generator: lines of the file.
    
    Requires:
        filename: should be a valid path/name of the file.
    """

    with open(filename, 'r', encoding='utf8') as datafile:
        for line in datafile:
            yield line.strip()


def obtain_substrings(line: str) -> Generator:
    """For each line, generate all possible substrings except the empty string.

    Args:
        line (str): word from which to generate the substrings.

    Yields:
        Generator: substrings obtained from the word given in the line.
    """

    size = len(line)
    for i in np.arange(1, size+1):
        for j in np.arange(0, i):
            yield line[j:i]


def find_substrings_from_file(filename: str) -> Generator:
    """Generates all the possible substrings from the words given in the file.

    Args:
        filename (str): path/name of the file with the words.

    Yields:
        Generator: substrings of each word of the file.
    
    Requires:
        filename: should be a valid path/name of a file.
    """

    for line in read_file(filename):
        for substring in obtain_substrings(line):
            yield substring


def create_hll(filename: str, relative_error: float) -> hyperloglog:
    """Creates a hyperloglog data structure populated with all the substrings of
    formed using the words in the given file.

    Args:
        filename (str): path/name of the file with the words.
        relative_error (float): relative error associated with the construction
            of the hyperloglog.

    Returns:
        hll (hyperloglog): data structure with the counts of each substring in
            the given file.
    
    Requires:
        filename: should be a valid path/name of a file.
    """

    hll = hyperloglog.HyperLogLog(relative_error)
    hll.add("")
    for substring in find_substrings_from_file(filename):
        hll.add(substring)
    return hll


def calculate_metrics(hll: hyperloglog, relative_error: float) -> tuple:
    """Calculates the metrics associated with the cardinality and the error of
    a hyperloglog data structure.

    Args:
        hll (hyperloglog): hyperloglog structure to evaluate.
        relative_error (float): relative error associated with the construction
            of the hll.

    Returns:
        lower_bound (float): lower bound of the estimate of the cardinality of
            the hll.
        substrings (int): estimate of the cardinality of the hll.
        upper_bound (float): upper bound of the estimate of the cardinality of
            the hll.
        abs_error (float): absolute error associated with the construction
            of the hyperloglog given its cardinality estimate.
    """

    substrings = len(hll)
    lower_bound = 1/(1+relative_error)*substrings
    upper_bound = 1/(1-relative_error)*substrings
    abs_error = relative_error/(1+relative_error)*substrings
    
    return lower_bound, substrings, upper_bound, abs_error


def print_estimates(name: str, lower_bound: float, substrings: float, 
                    upper_bound: float):
    """Prints the estimates in a more readable format.

    Args:
        name (str): name of the set
        lower_bound (float): lower bound on the estimate
        substrings (float): estimate
        upper_bound (float): upper bound on the estimate
    """

    print(f"Substrings of the {name} words:")
    print(f"\t Lower bound on unique substrings: {lower_bound}")
    print(f"\t Estimate on unique substrings:  {substrings}")
    print(f"\t Upper bound on substrings: {upper_bound}")
    print()


###############################################################################

if __name__ == '__main__':
    relative_error = 0.01
    hll_dk = create_hll('words_danish.txt', relative_error)
    hll_en = create_hll('words_english.txt', relative_error)

    *dk_metrics, dk_abs_err = calculate_metrics(hll_dk, relative_error)
    print_estimates("Danish", *dk_metrics)
    
    *en_metrics, en_abs_err = calculate_metrics(hll_en, relative_error)
    print_estimates("English", *en_metrics)

    hll_combined = copy.deepcopy(hll_en) # Create a copy of the HLL
    hll_combined.update(hll_dk) # Merge the other HLL into the combined one
    
    comb_abs_err = dk_abs_err + en_abs_err
    comb_low = len(hll_combined) - comb_abs_err
    comb_est = len(hll_combined)
    comb_upp = len(hll_combined) + comb_abs_err
    print_estimates("Combined", comb_low, comb_est, comb_upp)
    print()