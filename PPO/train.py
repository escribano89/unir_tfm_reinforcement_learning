
import csv
import random

import assistive_gym
import gym
import numpy as np

from config import ENV_NAME, MAX_STEPS, STEPS_FOR_LEARNING  # TODO BORRAR?
from ppo import PPO


# Obtenido de https://github.com/XinJingHao/PPO-Continuous-Pytorch
def beta_dist_action(a, max_action):
    return 2 * (a - 0.5) * max_action

if __name__ == "__main__":
    # Carga del entorno sobre el que el agente interactuará
    _env = gym.make(ENV_NAME)
    # Configuración de la semilla aleatoria
    _env.set_seed(random.randint(1, 200))

    # Información del csv en el que se guardará el proceso de aprendizaje
    _csv_info = ["step", "reward", "episode", "success"]
    _csv_dict_list = []

    # Obtención de la máxima acción permitida por el entorno
    _max_action = _env.action_space.high[0]

    # Inicialización del agente PPO
    _ppo = PPO(number_of_actions=_env.action_space.shape[0], input_dimensions=_env.observation_space.shape)
    
    # Variables para el seguimiento del aprendizaje
    _reward_history = []

    _total_steps = 0
    _trajectory_length = 0
    _episode = 1
    _steps = 0

    while _total_steps < MAX_STEPS:
        _observation = _env.reset()
        _done = False
        _score = 0

        # Se itera mientras no haya finalizado el episodio
        while not _done:
            _action, _probability = _ppo.select_action(_observation)
            # Ajuste de la acción debido a la distribución Beta empleada
            _adapted_action = beta_dist_action(_action, _max_action)
            _next_observation, _reward, _done, _info = _env.step(_adapted_action)

            _score += _reward
            _total_steps += 1
            _trajectory_length += 1

            # Añadimos la experiencia obtenida en la memoria
            _ppo.add_to_memory(_observation, _next_observation, _action, _probability, _reward, _done)

            # Solo se aprende tras transcurrir X pasos
            if _trajectory_length % STEPS_FOR_LEARNING == 0:
                _trajectory_length = 0
                _ppo.learn()

            # Actualizamos la observación anterior
            _observation = _next_observation

            # Se incluye la información en una lista para posteriormente
            # guardarlo en un archivo delimitado por comas
            _csv_dict_list.append({
				"step": _steps,
				"reward": round(_reward, 2),
				"episode": _episode,
				"success": _info.get("task_success") # Esto corresponde con la columna "success" del estudio
			})
            _steps = _steps + 1

        # Actualizamos el histórico de resultados
        _reward_history.append(_score)
        _avg_reward = np.mean(_reward_history[-100:])

        print("Entorno: {}, Episodio: {}, Pasos {}, Recompensa Promedio: {:.1f}".format(
            ENV_NAME, _episode, _total_steps, _avg_reward)
        )
        _episode += 1

    # Desconexión del entorno de Assistive Gym
    _env.disconnect()

    # Una vez ha finalizado el proceso de entrenamiento, se guardan
    # los modelos y los resultados en un csv
    _ppo.save_actor_critic()

    with open("training_steps.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = _csv_info)
        writer.writeheader()
        writer.writerows(_csv_dict_list)
