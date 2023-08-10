import sys
sys.path.remove('/n/app/python/3.7.4-ext/lib/python3.7/site-packages')
import numpy as np 
import os
import sys
import pdb
import statsmodels.api as sm
import scipy.stats


def normalize_covariance_matrix(unnormalized_cov):
	cov = np.copy(unnormalized_cov)
	varz = np.diag(unnormalized_cov)
	for ii in range(cov.shape[0]):
		for jj in range(cov.shape[1]):
			cov[ii,jj] = unnormalized_cov[ii,jj]/(np.sqrt(varz[ii])*np.sqrt(varz[jj]))
	return cov


def simulate_data(n_samples, n_snps, n_effects_per_run, effect_size_variance):
	# Generate random covariance matrix
	A = np.random.normal(size=(n_snps, n_snps))
	unnormalized_cov = np.dot(A, A.transpose())
	cov = normalize_covariance_matrix(unnormalized_cov)

	# Generate genotype data in target pop
	X_target = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov,size=n_samples)
	# Generate genotype datea in reference population
	X_ref = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov,size=n_samples)

	# Generate causal effect sizes
	betas = np.zeros(n_snps)
	for indexer in range(n_effects_per_run):
		betas[indexer] = np.random.normal(loc=0, scale=np.sqrt(effect_size_variance))


	# Simulate trait data
	predicted_means = np.dot(X_target, betas)
	Y = np.random.normal(loc=predicted_means, scale=1.0)


	return Y, X_target, X_ref, betas, cov



def run_standard_linear_regression(Y, X_target):
	#X2 = sm.add_constant(X_target)
	est = sm.OLS(Y, X_target)
	est2 = est.fit()
	#pvalues = est2.pvalues[1:]
	pvalues = est2.pvalues
	coefs = est2.params
	return pvalues, coefs

def update_metrics(metrics, causal_betas, pvalues):
	for ii, causal_beta in enumerate(causal_betas):
		if causal_beta == 0.0:
			metrics['false'] = metrics['false'] + 1
			if pvalues[ii] < .05:
				metrics['false_positives'] = metrics['false_positives'] + 1
	return metrics


def full_multivariate_beta_update(X, y, Xt_X):
	S = np.linalg.inv(Xt_X)
	#mu = np.dot(np.dot(S, np.transpose(X)), y) # # Equivalent to above
	mu = np.dot(S,np.dot(np.transpose(X),y))  # Just more accessable with sum stats
	return mu, S

def run_multivariate_variational_inference_with_full_data(Y, X):
	Xt_X = np.dot(np.transpose(X), X)
	mu, S = full_multivariate_beta_update(X, Y, Xt_X)
	t_stats = mu/np.sqrt(np.diag(S))

	pvalues = 2.0*(1.0-scipy.stats.t.cdf(np.abs(t_stats), X.shape[0] - X.shape[1]-1))

	return pvalues, mu

def run_multivariate_variational_inference_with_out_of_sample_ld(Y, X_target, X_ref):
	Xt_X = np.dot(np.transpose(X_ref), X_ref)



	mu, S = full_multivariate_beta_update(X_target, Y, Xt_X)
	t_stats = mu/np.sqrt(np.diag(S))

	pvalues = 2.0*(1.0-scipy.stats.t.cdf(np.abs(t_stats), X_target.shape[0] - X_target.shape[1]-1))

	return pvalues, mu

def run_multivariate_variational_inference_with_expected_ld(Y, X_target, simulated_cov):
	Xt_X = simulated_cov*X_target.shape[0]
	mu, S = full_multivariate_beta_update(X_target, Y, Xt_X)
	t_stats = mu/np.sqrt(np.diag(S))

	pvalues = 2.0*(1.0-scipy.stats.t.cdf(np.abs(t_stats), X_target.shape[0] - X_target.shape[1]-1))

	return pvalues, mu

def wishart_covariance_update(simulated_inv_cov, mu, S):
	V_inv = simulated_inv_cov + S + np.dot(np.reshape(mu, (len(mu),1)), np.reshape(mu, (1,len(mu))))
	V = np.linalg.inv(V_inv)
	return V


def run_multivariate_variational_inference_with_iterative_ld(Y, X_target, simulated_cov, max_iter=100):
	simulated_inv_cov = np.linalg.inv(simulated_cov)
	target_sample_size = X_target.shape[0]
	Xt_X = simulated_cov*target_sample_size
	for itera in range(max_iter):
		mu, S = full_multivariate_beta_update(X_target, Y, Xt_X)
		updated_cov = wishart_covariance_update(simulated_inv_cov, mu, S)
		Xt_X = updated_cov*target_sample_size
	t_stats = mu/np.sqrt(np.diag(S))
	pvalues = 2.0*(1.0-scipy.stats.t.cdf(np.abs(t_stats), X_target.shape[0] - X_target.shape[1]-1))
	return pvalues, mu

#####################
# Command line args
#####################
temporary_results_dir = sys.argv[1]
results_dir = sys.argv[2]
n_samples = int(sys.argv[3])
n_snps = int(sys.argv[4])

#########################
# Additional parameters
#########################
# We assume a residual variance of 1.0
n_simulations=1000
n_effects_per_run=10
effect_size_variance=.01

# Keep track of metrics for various methods
standard_linear_regression_metrics = {'false_positives': 0, 'false':0}
vi_linear_regression_full_data_metrics = {'false_positives': 0, 'false':0}
vi_linear_regression_out_of_sample_ld_metrics = {'false_positives': 0, 'false':0}
vi_linear_regression_expected_ld_metrics = {'false_positives': 0, 'false':0}
vi_linear_regression_iterative_ld_metrics = {'false_positives': 0, 'false':0}


for simulation_iter in range(n_simulations):
	print(simulation_iter)
	# First simulate data
	Y, X_target, X_ref, causal_beta, simulated_cov = simulate_data(n_samples, n_snps, n_effects_per_run, effect_size_variance)

	# Run standard linear regression
	standard_linear_regression_pvalues, standard_linear_regression_coef = run_standard_linear_regression(Y, X_target)
	standard_linear_regression_metrics = update_metrics(standard_linear_regression_metrics, causal_beta, standard_linear_regression_pvalues)

	# Run variational inference linear regression with full data
	vi_linear_regression_full_data_pvalues, vi_linear_regression_full_data_coef = run_multivariate_variational_inference_with_full_data(Y, X_target)
	vi_linear_regression_full_data_metrics = update_metrics(vi_linear_regression_full_data_metrics, causal_beta, vi_linear_regression_full_data_pvalues)

	# Run variational inference linear regression with out of sample ld
	vi_linear_regression_out_of_sample_ld_pvalues, vi_linear_regression_out_of_sample_ld_coef = run_multivariate_variational_inference_with_out_of_sample_ld(Y, X_target, X_ref)
	vi_linear_regression_out_of_sample_ld_metrics = update_metrics(vi_linear_regression_out_of_sample_ld_metrics, causal_beta, vi_linear_regression_out_of_sample_ld_pvalues)

	# Run variational inference linear regression with expected ld (this is not in sample ld)
	vi_linear_regression_expected_ld_pvalues, vi_linear_regression_expected_ld_coef = run_multivariate_variational_inference_with_expected_ld(Y, X_target, simulated_cov)
	vi_linear_regression_expected_ld_metrics = update_metrics(vi_linear_regression_expected_ld_metrics, causal_beta, vi_linear_regression_expected_ld_pvalues)

	# Run variational inference linear regression with iterative ld
	try:
		vi_linear_regression_iterative_ld_pvalues, vi_linear_regression_iterative_ld_coef = run_multivariate_variational_inference_with_iterative_ld(Y, X_target, simulated_cov, max_iter=100)
		vi_linear_regression_iterative_ld_pvalues2, vi_linear_regression_iterative_ld_coef2 = run_multivariate_variational_inference_with_iterative_ld(Y, X_target, simulated_cov,max_iter=5000)

		vi_linear_regression_iterative_ld_metrics = update_metrics(vi_linear_regression_iterative_ld_metrics, causal_beta, vi_linear_regression_iterative_ld_pvalues2)
	except:
		print('skipped one')


pdb.set_trace()

