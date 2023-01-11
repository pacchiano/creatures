import numpy as np
import IPython
import matplotlib.pyplot as plt
import copy
import ray

from learners import EFESLearner






def run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, step_size = .01, tag = ""):
	target_vector = np.full(dimension, True)
	
	ultimate_reward_list = []


	learner = EFESLearner(initial_lambda_multiplier, dimension, symbols = [0,1])


	for t in range(evolver_timesteps):
		ultimate_reward = 0
		
		evolver_vector = learner.sample_string() 
		sample_vector1 = copy.deepcopy(evolver_vector)

		evolver_vector = evolver_vector[:,1]
		#IPython.embed()

		print("Evolver Step t ", t+1, " - creature horizon ", creature_horizon, " - ", tag)

		
		for h in range(creature_horizon):
			index_to_perturb = np.random.choice(dimension)


			
			evolver_vector[index_to_perturb] = not evolver_vector[index_to_perturb]

			if np.min(evolver_vector == target_vector) >= True:		
				ultimate_reward = 1
				break

		ultimate_reward_list.append(ultimate_reward)


		#### Update the EFES learner
		sample_vector2 = learner.sample_string()

		learner.update_statistics1( sample_vector1, sample_vector2, ultimate_reward, step_size)

	return np.mean(ultimate_reward_list)


@ray.remote
def run_search_remote(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, step_size = .01, tag = ""):
	return run_search(dimension, evolver_timesteps, creature_horizon, initial_lambda_multiplier = 0, step_size = step_size, tag = tag)



if __name__ == '__main__':

	num_experiments = 10

	#evolver_timesteps = 1000
	dimension = 10
	creature_horizon = 100
	
	evolver_timesteps_list = [100, 500, 1000, 10000, 30000, 100000, 1000000]

	summary = np.zeros((len(evolver_timesteps_list), num_experiments))

	USE_RAY = True


	for evolver_timesteps,i in zip(evolver_timesteps_list, range(len(evolver_timesteps_list))):
		if USE_RAY:
			experiments_results = [run_search_remote.remote(dimension, evolver_timesteps, creature_horizon, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
			experiments_results = ray.get(experiments_results)
		else:
			experiments_results = [run_search(dimension, evolver_timesteps, creature_horizon, tag = "{} exp{}".format( evolver_timesteps, j+1)) for j in range(num_experiments)]
	
		summary[i, :] = experiments_results

	IPython.embed()

	means = np.mean(summary, 1)
	stds = np.std(summary, 1)

	plt.plot(evolver_timesteps_list, means, label = "Average Rewards", color = "blue")
	plt.fill_between(evolver_timesteps_list, means- .2*stds , means + .2*stds, alpha = .2, color = "blue" )
	plt.title("Random Search Average Rewards - H{}".format(creature_horizon))
	plt.xlabel("Evolver Timesteps")
	plt.ylabel("Average Ultimate Rewards")
	plt.savefig("./plots/randomwalk_baldwin.png")


	plt.close("all")


	


	IPython.embed()



