import numpy as np
import IPython
import matplotlib.pyplot as plt
import copy
import ray

from learners import EFESLearner





#### Decreasing learning rate schedule.... Heuristic because the theoretical one is very pessimistic.
def run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
	step_size = .01, tag = ""):
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

		learner.update_statistics1( sample_vector1, sample_vector2, ultimate_reward, step_size*1.0/np.sqrt(t+1))

	return ultimate_reward_list


@ray.remote
def run_search_remote(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, 
	step_size = .01, tag = ""):
	return run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, step_size = step_size, tag = tag)



if __name__ == '__main__':

	num_experiments = 10

	#evolver_timesteps = 1000
	dimension = 5
	creature_horizons = [1, 10]
	colors = ["blue", "red"]

	step_size = 1

	evolver_timesteps = 1000000

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
				creature_horizon, step_size = step_size, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
			experiments_results = ray.get(experiments_results)
		else:
			experiments_results = [run_search(dimension, evolver_timesteps, 
				creature_horizon, step_size = step_size, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
		
		means = np.mean(experiments_results, 0)
		stds = np.std(experiments_results, 0)

		means = np.mean(means.reshape(int(evolver_timesteps/averaging_window), averaging_window), 1)

		stds = np.mean(stds.reshape(int(evolver_timesteps/averaging_window), averaging_window), 1)
		
		#creature_horizon_results.append((means, stds))



		#IPython.embed()


		plt.plot(Ts, means, label = "H-{}".format(creature_horizon), color = colors[i])
		plt.fill_between(Ts, means- .2*stds , means + .2*stds, alpha = .2, color = colors[i] )
	


	plt.xscale("log")
	plt.title("Foraging Search Average Rewards")
	plt.xlabel("Evolver Timesteps")
	plt.ylabel("Average Ultimate Rewards")
	plt.legend(loc = "lower right")
	plt.savefig("./plots/randomwalk_baldwin.png")


	plt.close("all")

	#IPython.embed()



