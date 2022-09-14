
import csv
import random

import assistive_gym
import gym
import numpy as np

from config import ENV_NAME, LEARNING_STARTS, MAX_STEPS, POLICY_NOISE
from td3 import TD3

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

    # Inicialización del agente TD3
    _td3 = TD3(
        states_dimension=_env.observation_space.shape[0],
        actions_dimension=_env.action_space.shape[0],
        max_action=_max_action
    )
    
    # Variables para el seguimiento del aprendizaje
    _reward_history = []

    _total_steps = 0
    _episode = 1
    _steps = 0

    while _total_steps < MAX_STEPS:
        _observation = _env.reset()
        _done = False
        _score = 0

        # Se itera mientras no haya finalizado el episodio
        while not _done:
            # En TD3 hay un período donde se ejecutan acciones aleatorias
            if _total_steps < LEARNING_STARTS:
                _action = _env.action_space.sample()
            else:
                _action = _td3.select_action(np.array(_observation))

                _action = (_action + np.random.normal(0, POLICY_NOISE, size=_env.action_space.shape[0])).clip(
                    _env.action_space.low, _env.action_space.high
                )

            _next_observation, _reward, _done, _info = _env.step(_action)

            _score += _reward
            _total_steps += 1
            _steps += 1

            # Añadimos la experiencia obtenida en la memoria
            _td3.add_to_memory(_observation, _next_observation, _action, _reward, _done)

            # Si el episodio ha terminado, se aprende para cada uno de los timesteps llevados a cabo
            if _done:
                _td3.learn(_steps)

            # Actualizamos la observación anterior
            _observation = _next_observation

            # Se incluye la información en una lista para posteriormente
            # guardarlo en un archivo delimitado por comas
            _csv_dict_list.append({
				"step": _total_steps,
				"reward": round(_reward, 2),
				"episode": _episode,
				"success": _info.get("task_success") # Esto corresponde con la columna "success" del estudio
			})

        _steps = 0

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
    _td3.save_actor_critic()

    with open("training_steps.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = _csv_info)
        writer.writeheader()
        writer.writerows(_csv_dict_list)
