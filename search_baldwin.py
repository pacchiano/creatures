import numpy as np
import IPython
import matplotlib.pyplot as plt

def generate_random_binary_vector(dimension):
	return np.random.random(dimension) >= .5



def run_search(evolver_dimension, evolver_timesteps, creature_dimension, creature_horizon):
	target_evolver_vector = np.full(evolver_dimension, True)
	target_creature_vector = np.full(creature_dimension, True)

	ultimate_reward_list = []

	found_evolver_match = False

	for t in range(evolver_timesteps):
		ultimate_reward = 0
		if not found_evolver_match:
			evolver_vector = generate_random_binary_vector(evolver_dimension)
		print("Evolver Step t ", t+1, " - creature horizon ", creature_horizon)

		if np.min(evolver_vector == target_evolver_vector) >= True:
			#print("Found evolver match")
			found_evolver_match = True

			for h in range(creature_horizon):
				
				creature_vector = generate_random_binary_vector(creature_dimension)
				if np.min(creature_vector == target_creature_vector) >= True:	
					#print("Found creature match!")
					#IPython.embed()
					ultimate_reward = 1
					break

		ultimate_reward_list.append(ultimate_reward)

	return ultimate_reward_list



def find_first_evolver_step_match(evolver_dimension, creature_dimension, creature_horizon):
	target_evolver_vector = np.full(evolver_dimension, True)
	target_creature_vector = np.full(creature_dimension, True)


	found_evolver_match = False

	t = 1

	while True:

		if not found_evolver_match:
			evolver_vector = generate_random_binary_vector(evolver_dimension)
		print("Evolver Step t ", t+1, " - creature horizon ", creature_horizon)

		if np.min(evolver_vector == target_evolver_vector) >= True:
			#print("Found evolver match")
			found_evolver_match = True

			for h in range(creature_horizon):
				
				creature_vector = generate_random_binary_vector(creature_dimension)
				if np.min(creature_vector == target_creature_vector) >= True:	
					#print("Found creature match!")
					## Found full match

					#IPython.embed()
					return t	
			t += 1


if __name__ == '__main__':

	evolver_dimension = 10
	evolver_timesteps = 10000

	num_experiments = 10

	creature_dimension = 5
	average_rewards = []
	evolver_step_matches = []

	creature_horizons = [1, 5, 10, 50, 100, 200]

	averaging_window = 10

	horizons_results = []

	for creature_horizon in creature_horizons:
 
		all_ultimate_reward_list = []

		for _ in range(num_experiments):
			ultimate_reward_list = run_search(evolver_dimension, evolver_timesteps, creature_dimension, creature_horizon)

			all_ultimate_reward_list.append(ultimate_reward_list)

		ultimate_reward_list = np.mean(all_ultimate_reward_list, 0)

		



		horizons_results.append(ultimate_reward_list)

		average_reward = np.mean(ultimate_reward_list)

		average_rewards.append(average_reward)

		evolver_step_match = find_first_evolver_step_match(evolver_dimension, creature_dimension, creature_horizon)

		evolver_step_matches.append(evolver_step_match)




	#hyperparam_rewards_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)


	plt.plot(creature_horizons, average_rewards, label = "Average Rewards")
	plt.title("Random Search Average Rewards - T{}".format(evolver_timesteps))
	plt.xlabel("Actor Horizons")
	plt.ylabel("Average Ultimate Rewards")
	plt.savefig("./plots/search_baldwin_fixed_evolver_T{}.png".format(evolver_timesteps))


	plt.close("all")


	plt.plot(creature_horizons, evolver_step_matches, label = "Num Evolver Timesteps")
	plt.title("Random Search Num Evolver Steps Needed")
	plt.xlabel("Actor Horizons")
	plt.ylabel("Num Evolver Timesteps")
	plt.savefig("./plots/search_baldwin_num_evolver_steps.png")

	
	plt.close("all")

	Ts = (np.arange(int(evolver_timesteps/averaging_window)) + 1)*averaging_window

	for i,creature_horizon in zip(range(len(creature_horizons)), creature_horizons):

		compressed_horizons_results = np.mean(np.array(horizons_results[i]).reshape(int(evolver_timesteps/averaging_window), averaging_window), 1)

		plt.plot(Ts, compressed_horizons_results, label = "H-{}".format(creature_horizon))

	plt.title("Actor Ultimate Performance Varying Horizon")
	plt.xlabel("Evolver Timesteps")
	plt.ylabel("Ultimate Reward")
	plt.legend(loc = "upper left")
	plt.savefig("./plots/horizon_comparison_baldwin.png")

	plt.close("all")

	IPython.embed()



