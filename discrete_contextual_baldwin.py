import numpy as np
import IPython
import copy
import sys
from random import randrange
import matplotlib.pyplot as plt
from learners import EFESLearner



class DiscreteContextual:
	def __init__(self, rademacher_parameters):
		self.num_contexts = len(rademacher_parameters)
		self.rademacher_parameters = rademacher_parameters ### these are actually bernoulli parameters
		self.means = [2*x -1 for x in rademacher_parameters]
		#IPython.embed()
		self.optimal_mean_reward = 1.0/self.num_contexts*(np.sum( [ max(0,x) for x in self.means] ))


	### Returns a context context_index
	def sample_context(self):
		return randrange(self.num_contexts)

	def sample_reward(self, context_index):
		rademacher_parameter = self.rademacher_parameters[context_index]
		sample  = np.random.random() <= rademacher_parameter 
		return 2*sample - 1

	def sample_context_reward(self):
		context_index = self.sample_context()
		reward = self.sample_reward(context_index)
		return (context_index, reward)


class CreatureLearner:
	def __init__(self, num_contexts, evolver_vector = [], delta  = .01):
		### these will keep track of the statistics of the "1" action for each context type.
		self.rew_cumulative_statistics = np.zeros(num_contexts)
		self.num_pulls = np.zeros(num_contexts)
		self.rew_ucb_statistics =  np.ones(num_contexts)*float("inf")
		self.delta = delta
		self.rew_mean_statistics = np.zeros(num_contexts)
		self.evolver_vector = evolver_vector

	def get_action(self, context_index):
		if context_index < len(self.evolver_vector):
			return self.evolver_vector[context_index]

		#raise ValueError("Asdflakmsdflaksmdf got here!")

		if self.rew_ucb_statistics[context_index] >= 0:
			return 1
		else:
			return 0

	def update_stats(self, context_index, reward):
		self.rew_cumulative_statistics[context_index] += reward
		self.num_pulls[context_index] += 1

		self.rew_mean_statistics = self.rew_cumulative_statistics*1.0/self.num_pulls
		
		self.rew_ucb_statistics[context_index] = 1.0*self.rew_cumulative_statistics[context_index]/self.num_pulls[context_index] 
		self.rew_ucb_statistics[context_index] += np.sqrt(1.0*np.log(1/self.delta)/self.num_pulls[context_index])



def run_experiment(num_contexts, parameters, evolver_dimensions,
	evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
	step_size = .1, tag = ""):
	
	target_vector = np.full(num_contexts, True)
	ultimate_reward_list = np.zeros(evolver_timesteps)
	
	if num_contexts != len(parameters):
		raise ValueError("num contexts != len(parameters)")

	discrete_contextual_problem = DiscreteContextual(parameters)

	learner = EFESLearner(initial_lambda_multiplier, evolver_dimensions, symbols = [0,1])
	

	if creature_horizon == 0:
		raise ValueError("Creature horizon is zero")


	for t in range(evolver_timesteps):
		
		evolver_vector = learner.sample_string() 
		sample_vector1 = copy.deepcopy(evolver_vector)

		evolver_vector = evolver_vector[:,1]

		print("Evolver Step t ", t+1, " - creature horizon ", creature_horizon, " - ", tag)
		#IPython.embed()
		print("Evolver vector ", evolver_vector)
		#raise ValueError("Asflk")



		creature_learner = CreatureLearner(num_contexts, evolver_vector = evolver_vector)
		learners_rewards = np.zeros(creature_horizon)
		for h in range(creature_horizon):
			
			context_index, reward = discrete_contextual_problem.sample_context_reward()

			action = creature_learner.get_action(context_index)	
			
			#contexts_actions.append((context_index, action))

			if action == 1:
				learners_rewards[h] = reward
				creature_learner.update_stats(context_index, reward)

			else:
				learners_rewards[h] = 0



		
		ultimate_reward = learners_rewards[-1]
		ultimate_reward_list[t] = ultimate_reward

		#### Update the EFES learner
		sample_vector2 = learner.sample_string()

		print("ultimate reward ", ultimate_reward)

		learner.update_statistics1( sample_vector1, sample_vector2, ultimate_reward, step_size)

	return ultimate_reward_list





if __name__ == '__main__':
	parameters = [.6, .4, .3, .7]
	#IPython.embed()	




	evolver_timesteps = int(sys.argv[1])
	creature_horizon = int(sys.argv[2])
	evolver_dimensions_list = sys.argv[3].split( ",")	
	#IPython.embed()
	evolver_dimensions_list = [int(x) for x in evolver_dimensions_list]

	num_experiments = 10
	step_size = .5

	colors = ["blue", "red", "black"]

	for evolver_dimensions, i in zip(evolver_dimensions_list, range(len(evolver_dimensions_list))):



		# evolver_timesteps = 1000
		# creature_horizon = 1000
		# evolver_dimensions = 2

		Ts = np.arange(evolver_timesteps)+1

		discrete_contextual_problem = DiscreteContextual(parameters)
		optimal_mean_reward = discrete_contextual_problem.optimal_mean_reward

		if evolver_dimensions > len(parameters):
			raise ValueError("Evolver dimensions larger than num parameters")

		ultimate_reward_lists = []
		for _ in range(num_experiments):

			ultimate_reward_list = run_experiment(len(parameters), parameters, evolver_dimensions,
				evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
			step_size = step_size, tag = "")

			ultimate_reward_lists.append(ultimate_reward_list)

		#IPython.embed()

		ultimate_rewards_mean = np.mean(np.cumsum(ultimate_reward_lists,1), 0)
		ultimate_rewards_std = np.std(np.cumsum(ultimate_reward_lists,1), 0)

		cumulative_regrets = np.cumsum(np.ones(evolver_timesteps)*optimal_mean_reward) - ultimate_rewards_mean

		plt.plot(Ts, cumulative_regrets, label = 'Evolver Dim {}'.format(evolver_dimensions), color = colors[i] )
		plt.fill_between(Ts, cumulative_regrets - .5*ultimate_rewards_std, cumulative_regrets + .5*ultimate_rewards_std, color = colors[i], alpha = .2 )
	
	plt.title("Regret T{} H{}".format(evolver_timesteps, creature_horizon))
	plt.xlabel("Timesteps")
	plt.ylabel("Regret")
	plt.legend(loc = "upper right")	
	plt.savefig("./plots/discrete_context_regrets_T{}_H{}.png".format(evolver_timesteps, creature_horizon))

	# discrete_contextual_problem = DiscreteContextual(parameters)
	# creature_learner = CreatureLearner(len(parameters))

	# num_steps = 1000	
	# Ts = np.arange(num_steps) + 1

	# optimal_mean_reward = discrete_contextual_problem.optimal_mean_reward	

	# contexts_actions = []

	# learners_rewards = np.zeros(num_steps)
	# for i in range(num_steps):
	# 	context_index, reward = discrete_contextual_problem.sample_context_reward()

	# 	action = creature_learner.get_action(context_index)	
	# 	print("-------")
	# 	print("Action ", action)
	# 	print("Context ", context_index)

	# 	contexts_actions.append((context_index, action))

	# 	if action == 1:
	# 		learners_rewards[i] = reward
	# 		creature_learner.update_stats(context_index, reward)

	# 	else:
	# 		learners_rewards[i] = 0

	#IPython.embed()


