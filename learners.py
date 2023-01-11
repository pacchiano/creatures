import random
import numpy as np
import IPython


def vectorized_multinomial_sample(prob_matrix):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    r = np.outer(r, np.ones(prob_matrix.shape[1]))
    u = (s > r)
    mask_u = u[:, 1:]^u[:, 0:-1]
    u[:, 1:] = mask_u
    return u



class EFESLearner:
	def __init__(self, initial_lambda_multiplier, string_length, symbols = []):
		self.symbols = symbols
		self.num_symbols = len(self.symbols)
		self.lambda_matrix = np.ones((string_length, self.num_symbols))*initial_lambda_multiplier*1.0
		self.string_length = string_length
		self.update_probability_matrix()

	def update_probability_matrix(self):
		normalization_factors = 1.0/np.sum(np.exp(self.lambda_matrix), axis = 1)
		normalization_factors = np.outer(normalization_factors, np.ones(self.num_symbols))
		self.probabilities_matrix = np.exp(self.lambda_matrix)*normalization_factors


	def set_lambda_matrix(self, lambda_matrix):
		self.lambda_matrix = lambda_matrix

	def get_lambda_matrix(self):
		return self.lambda_matrix

	def update_statistics1(self, sample_string1, sample_string2, reward1, step_size):
		sample_indicator_matrix1 = sample_string1
		sample_indicator_matrix2 = sample_string2

		grad = reward1*(sample_indicator_matrix1*1.0-sample_indicator_matrix2*1.0)
		self.lambda_matrix += step_size*grad
		self.update_probability_matrix()

	def update_statistics2(self, sample_string1, reward1, reward2, step_size):
		sample_indicator_matrix1 = sample_string1
		grad = (float(reward1)-float(reward2))*(sample_indicator_matrix1*1.0)
		self.lambda_vector += step_size*grad
		self.update_probability_matrix()

	def sample_string(self):
		return vectorized_multinomial_sample(self.probabilities_matrix)
