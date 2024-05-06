'''
**** WARNING ****
CODE HERE IS TEMPORARY AND LIKELY OBSOLETE.
IT WAS JUST AN HELPER FOR ME WHILE WRITING THE intro_var
NOTEBOOK, WHICH SHOULD BE CONSIDERED THE REFERENCE.
'''

#######################################################################


'''
Simple idea for computing the empirical quantile for a random set
of samples by using sorting.
This version eliminates the use of torch and only relies on numpy.
'''

import numpy as np
import matplotlib.pyplot as plt

# Search abput pydentic base model (class); can vielleicht : float
def get_quantile (given_samples : np.array, p : float,
                            verbose=True) -> (float, float):
    '''
    given_samples is an array of real numbers containing the samples
    p is the target probability between 0 and 1
    Returns an estimate CLOSED INTERVAL for the quantile.
    '''
    assert(p > 0. and p < 1.)
    # Sort samples in ascending order
    samples = np.sort(given_samples.reshape(-1,))
    n_samples = samples.shape[0]
    assert(n_samples >= 2)
    # Convert the probability p into "how many elements" 
    # from the sorted samples has to be taken until reaching p.
    # Example: if the samples are 100 sorted number, 
    # take the first 80 to realize the 80% quantile.
    # Round up to one index less / more to build an interval 
    idx_low = int(np.floor(n_samples * p) - 1)
    idx_up = int(idx_low + 1)
    # q is the approximation of the quantile
    q_low = samples[idx_low]
    q_up = samples[idx_up]
    # Double check that the two elements approximate the desired quantile
    p_low = np.sum(samples <= q_low) / n_samples
    p_up = np.sum(samples <= q_up) / n_samples
    
    if verbose:
        print(f"From get_quantile:")
        print(f"\t{p*100:.3f}%-quantile in [{q_low:.3e}, {q_up:.3e}]")
        print(f"\t{p*100:.3f}%-quantile is {q_low:.3e}") 
        print(f"\tP [X <= {q_low :.3e}] = {p_low : .3e}")
        print(f"\tP [X <= {q_up:.3e}] = {p_up : .3e}")
        print("-")
    return (q_low, q_up)
#---


def test_quantile ():
    '''
    This is a simple test based on the true values of a normal distribution.
    '''
    s = np.random.normal(0., 1., (100_000,))
    # True values
    q_999 = 3.0902
    q_995 = 2.5758
    q_950 = 1.6449
    q_50 = 0.
    q_20 = -0.8416
    q_025 = -1.96
    # Estimated
    r_999 = get_quantile(s, 0.999)[0]
    r_995 = get_quantile(s, 0.995)[0]
    r_950 = get_quantile(s, 0.950)[0]
    r_50 = get_quantile(s, 0.50)[0]
    r_20 = get_quantile(s, 0.20)[0]
    r_025 = get_quantile(s, 0.025)[0]
    # User output
    print(f"99.9%: {q_999 : .3e} VS {r_999 : .3e}")
    print(f"99.5%: {q_995 : .3e} VS {r_995 : .3e}")
    print(f"95%: {q_950 : .3e} VS {r_950 : .3e}")
    print(f"50%: {q_50 : .3e} VS {r_50 : .3e}")
    print(f"20%: {q_20 : .3e} VS {r_20 : .3e}")
    print(f"0.25%: {q_025 : .3e} VS {r_025 : .3e}")
    # Actual error check
    errs = []
    errs.append(np.abs(q_999 - r_999))
    errs.append(np.abs(q_995 - r_995))
    errs.append(np.abs(q_950 - r_950))
    errs.append(np.abs(q_50 - r_50))
    errs.append(np.abs(q_20 - r_20))
    errs.append(np.abs(q_025 - r_025))
    # Return 1 if all the errors are small enough
    if False in errs < 0.1:
        return 0
    return 1
#---


def get_var(samples : np.array, confidence : float,
                        verbose : bool = True) -> (float, float):
    '''
    Compute the Value At Risk for a set of samples, by using its
    interpretation as quantile: VaR(c) = - QUANTILE(1 - c)
    Returns an interval where the true val is estimated to lie.
    '''
    p = 1. - confidence
    q_low, q_up = get_quantile(samples, p, verbose = False)
    var_low = - q_up
    var_up = - q_low
    if verbose:
        print(f"{confidence * 100}%-V@R in [{var_low:.3e}, {var_up:.3e}]")
    return (var_low, var_up) 
#---

def test_var ():
    '''
    Testing the previous VaR computation by using a theoretical example
    with known result.
    '''
    n_samples = 10_000
    # Building the two independent random variables
    random_integers_1 = np.random.randint(1000, size = n_samples)
    random_integers_2 = np.random.randint(1000, size = n_samples)

    samples_1 = np.ones(n_samples) * 1000.
    samples_2 = np.ones(n_samples) * 1000.
    for nth in range(n_samples):
        if random_integers_1[nth] > 991:
            samples_1[nth] = -10_000.
        if random_integers_2[nth] > 991:
            samples_2[nth] = -10_000
    # Now, the historical samples are built
    confidence = 0.99 
    var_low, var_high = get_var(samples_1, confidence)
    print("True value is: -1000")
    true_var = -1000.
    err_var = np.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)

    var_low, var_high = get_var(samples_2, confidence)
    print("True value is: -1000")
    true_var = -1000.
    err_var = np.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)

    samples_3 = (samples_1 + samples_2) / 2.
    var_low, var_high = get_var(samples_3, confidence)
    print("True value is: 4500")
    true_var = 4500
    err_var = np.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)
    return True


def err_gaussian (p, true_q):
    max_power = 6
    errors = np.ones(max_power)
    for nth in range(1, max_power + 1):
        n_samples = int(10 ** nth)
        s = np.random.normal(0., 1., (n_samples,))
        q_low, q_up = get_quantile(s, p)
        # Estimate the quantile as the mean point in the interval
#        q = (q_low + q_up) / 2.
        q = q_low
#        q = q_up
        errors[nth - 1] = np.abs(q - true_q)
    
    plt.grid()    
    plt.plot(range(1, max_power+1), errors)
    plt.title(f"Gaussian: {p*100:.3f}%-quantile estimation")
    plt.xlabel(f"#samples (10 power)")
    plt.ylabel(f"Abs error")
    plt.show()
    return 1
#---

err_gaussian(0.005, -2.5758)
err_gaussian(0.010, -2.3263)
err_gaussian(0.10, -1.2816)
err_gaussian(0.20, -0.8416)
err_gaussian(0.50, 0.)
err_gaussian(0.70, 0.5244)
err_gaussian(0.95, 1.6449)
err_gaussian(0.990, 2.3263)

