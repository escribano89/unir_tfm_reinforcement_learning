
import csv
import random

import assistive_gym
import gym
import numpy as np
from numpngw import write_apng
from config import ENV_NAME, MAX_STEPS
from ppo import PPO


# Obtenido de https://github.com/XinJingHao/PPO-Continuous-Pytorch
def beta_dist_action(a, max_action):
    return 2 * (a - 0.5) * max_action

if __name__ == "__main__":
    # Carga del entorno sobre el que el agente interactuará
    _env = gym.make(ENV_NAME)
    # Configuración de la semilla aleatoria
    _env.set_seed(random.randint(1, 200))
    
    # Configuración de la cámara
    _env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75],
                     fov=60, camera_width=1920//4, camera_height=1080//4)

    # Información del csv en el que se guardará el proceso de aprendizaje
    _csv_info = ["step", "reward", "episode", "success"]
    _csv_dict_list = []

    # Obtención de la máxima acción permitida por el entorno
    _max_action = _env.action_space.high[0]

    # Inicialización del agente PPO
    _ppo = PPO(number_of_actions=_env.action_space.shape[0], input_dimensions=_env.observation_space.shape)
    _ppo.load_actor_critic()

    # Variables para el seguimiento del aprendizaje
    _reward_history = []
    _total_steps = 0
    _trajectory_length = 0
    _all_success = 0
    _episode = 1
    _steps = 0
    best_reward = 0
    worst_reward = 0

    while _total_steps < MAX_STEPS:
        _observation = _env.reset()
        _frames = []
        _done = False
        _score = 0
        _rollout_rewards = []
        _rollout_success = False

        # Se itera mientras no haya finalizado el episodio
        while not _done:
            _action, _ = _ppo.select_action(_observation)
            # Ajuste de la acción debido a la distribución Beta empleada
            _adapted_action = beta_dist_action(_action, _max_action)
            _next_observation, _reward, _done, _info = _env.step(_adapted_action)

            # Obtención del frame actual
            _img, _ = _env.get_camera_image_depth()
            _frames.append(_img)

            _score += _reward
            _total_steps += 1
            _trajectory_length += 1

            # Guardamos la recompensa obtenida
            _rollout_rewards.append(_reward)

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
            if _done:
                _rollout_success = _info.get("task_success")

            _steps = _steps + 1

        # Actualizamos el histórico de resultados
        _reward_history.append(_score)
        _avg_reward = np.mean(_reward_history[-100:])

        print("Entorno: {}, Episodio: {}, Pasos {}, Recompensa Promedio: {:.1f}".format(
            ENV_NAME, _episode, _total_steps, _avg_reward)
        )
        _episode += 1

        if _rollout_success:
            _all_success += 1

        # Guardamos un png animado para las simulaciones más destacadas
        if np.sum(_rollout_rewards) < _worst_reward:
            _worst_reward = np.sum(_rollout_rewards)
            write_apng(f'rollout/worst_output_{_episode}.png', _frames, delay=100)

        if np.sum(_rollout_rewards) > _best_reward:
            _best_reward = np.sum(_rollout_rewards)
            write_apng(f'rollout/best_output_{_episode}.png', _frames, delay=100)

    # Desconexión del entorno de Assistive Gym
    _env.disconnect()

    # Generación del archivo csv con los resultados obtenidos
    with open(f'rollout/results.csv', 'w') as csvfile:                 
        writer = csv.DictWriter(csvfile, fieldnames = _csv_info)
        writer.writeheader()        
        writer.writerows(_csv_dict_list)

    print("AVG REWARDS: ", np.mean(_reward_history))
    print("AVG STD: ", np.std(_reward_history))
    print("SUCCESS: ", f"{_all_success}%")

