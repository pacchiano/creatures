import numpy as np
import IPython
import matplotlib.pyplot as plt
import copy
import ray

from learners import EFESLearner


def perturb_binary_vector(vector, probability):
	dimension = np.size(vector)
	what_to_perturb = np.random.random(dimension) <= probability
	# IPython.embed()
	# raise ValueError("ASdflkm")
	delta = what_to_perturb*1.0 - what_to_perturb*vector*1.0 
	
	result = (1-what_to_perturb)*vector + delta
	return result


#### Decreasing learning rate schedule.... Heuristic because the theoretical one is very pessimistic.
def run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
	step_size = .01, tag = "", environment_drift = False, drift_probability = .01):
	target_vector = np.full(dimension, True)
	
	ultimate_reward_list = []

	learner = EFESLearner(initial_lambda_multiplier, dimension, symbols = [0,1])
	



	if creature_horizon == 0:
		raise ValueError("Creature horizon is zero")


	for t in range(evolver_timesteps):
		ultimate_reward = 0
		
		evolver_vector = learner.sample_string() 
		sample_vector1 = copy.deepcopy(evolver_vector)

		evolver_vector = evolver_vector[:,1]
		#IPython.embed()

		print("Evolver Step t ", t+1, " - creature horizon ", creature_horizon, " - ", tag)

		if environment_drift:
			target_vector = perturb_binary_vector(target_vector, drift_probability)
			print("Env Drifting!.Current target vector {}".format(target_vector))

		for h in range(creature_horizon):
			if np.min(evolver_vector == target_vector) >= True:		
							ultimate_reward = 1
							break

			index_to_perturb = np.random.choice(dimension)			
			evolver_vector[index_to_perturb] = not evolver_vector[index_to_perturb]

		
		if np.min(evolver_vector == target_vector) >= True:		
			ultimate_reward = 1
			#break

		ultimate_reward_list.append(ultimate_reward)


		#### Update the EFES learner
		sample_vector2 = learner.sample_string()

		learner.update_statistics1( sample_vector1, sample_vector2, ultimate_reward, step_size)

	return ultimate_reward_list


@ray.remote
def run_search_remote(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
	step_size = .01, tag = "", environment_drift = False, drift_probability = .01):
	return run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
		step_size = step_size, tag = tag, environment_drift = environment_drift, drift_probability = drift_probability)



if __name__ == '__main__':

	num_experiments = 10

	environment_drift = False
	drift_probability = .0001

	#evolver_timesteps = 1000
	dimension = 5
	creature_horizons = [1, 50, 100, 200]
	colors = ["blue", "red", "green", "black"]

	step_size = .101

	evolver_timesteps = 100000

	#Ts = np.arange(evolver_timesteps) + 1
	averaging_window = 100

	Ts = (np.arange(int(evolver_timesteps/averaging_window)) + 1)*averaging_window
	
	#summary = np.zeros(( num_experiments, evolver_timesteps))
	### Average over the last 100 
	USE_RAY = True

	#creature_horizon_results = []

	for creature_horizon, i in zip(creature_horizons, range(len(creature_horizons))):

		if USE_RAY:
			experiments_results = [run_search_remote.remote(dimension, evolver_timesteps, 
				creature_horizon, step_size = step_size, environment_drift = environment_drift, drift_probability = drift_probability, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
			experiments_results = ray.get(experiments_results)
		else:
			experiments_results = [run_search(dimension, evolver_timesteps, 
				creature_horizon, step_size = step_size, environment_drift = environment_drift, drift_probability = drift_probability, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
		
		means = np.mean(experiments_results, 0)
		stds = np.std(experiments_results, 0)

		means = np.mean(means.reshape(int(evolver_timesteps/averaging_window), averaging_window), 1)

		stds = np.mean(stds.reshape(int(evolver_timesteps/averaging_window), averaging_window), 1)
		
		#creature_horizon_results.append((means, stds))



		#IPython.embed()


		plt.plot(Ts, means, label = "H-{}".format(creature_horizon), color = colors[i])
		plt.fill_between(Ts, means- .2*stds , means + .2*stds, alpha = .2, color = colors[i] )
	


	plt.xscale("log")
	plt.yscale("log")

	plt.title("Foraging Search Average Rewards")
	plt.xlabel("Evolver Timesteps")
	plt.ylabel("Average Ultimate Rewards")
	plt.legend(loc = "lower right")
	if environment_drift:
		plt.savefig("./plots/randomwalk_baldwin_{}_drift{}_prob{}.png".format(step_size, environment_drift, drift_probability))
	else:
		plt.savefig("./plots/randomwalk_baldwin_{}.png".format(step_size, environment_drift))


	plt.close("all")


	### Save the data and add commmand line arguments


	#IPython.embed()



