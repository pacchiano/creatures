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
	evolver_timesteps = 100000

	creature_dimension = 5
	average_rewards = []
	evolver_step_matches = []

	creature_horizons = [1, 5, 10, 50, 100, 200]

	for creature_horizon in creature_horizons:

		ultimate_reward_list = run_search(evolver_dimension, evolver_timesteps, creature_dimension, creature_horizon)

		average_reward = np.mean(ultimate_reward_list)

		average_rewards.append(average_reward)

		evolver_step_match = find_first_evolver_step_match(evolver_dimension, creature_dimension, creature_horizon)

		evolver_step_matches.append(evolver_step_match)




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

	





	IPython.embed()



