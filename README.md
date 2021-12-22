# Bayesian Network Reasoner
This repository provides a BN reasoner, which parses Bayesian Networks from BIFXML files.

### Prerequisites
- **Python 3.9** (for the typing feature)
- **A BIFXML file** (examples in the /testing folder)
- **Dependencies** (from the `requirements.txt` file)

### Example usage
```Python
# TODO: remember to never use the same BNReasoner instance -> this can lead to unexpected behavior
if __name__ == '__main__':
    # Example usage
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').is_d_separated({'X'}, {'J'}, {'O'}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').compute_marginal_distribution({'J', 'I'}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').compute_marginal_distribution({'J', 'I'}, {'O': True}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').calculate_MPE({'O': True}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').calculate_MPE({'J': True, 'O': False}))
```