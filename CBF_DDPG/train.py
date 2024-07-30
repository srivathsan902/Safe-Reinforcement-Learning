from tqdm import tqdm
import numpy as np
import os
import cv2
import wandb
import safety_gymnasium
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def log_canvas(positions, velocities, actions):
    plot_frames = []

    for i, pos in enumerate(positions):
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.5, color='green', fill=False)
        ax.add_patch(circle)
        plt.axvline(x=1.125, color='r')
        plt.axvline(x=-1.125, color='r')
        plt.plot([p[0] for p in positions[:i+1]], [p[1] for p in positions[:i+1]], 'bo-', linewidth=0.5, markersize=2)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.title(f'Step {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')

        ax.set_aspect('equal', 'box')

        position_text = f"x: {pos[0]:.2f}, y: {pos[1]:.2f}"
        velocity_text = f"v_x: {velocities[i][0]:.2f}, v_y: {velocities[i][1]:.2f}"
        action_text = f"Force: {actions[i][0]:.2f}, Omega: {actions[i][1]:.2f}"
        text_str = f"{position_text}\n{velocity_text}\n{action_text}"
        plt.text(2.2, 2.5, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Create a canvas and render the figure
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert the canvas to a numpy array
        buf = canvas.tostring_rgb()
        width, height = canvas.get_width_height()
        plot_image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

        plot_frames.append(plot_image)

        plt.close(fig)

    # Convert plot frames list to a numpy array with shape (time, height, width, channels)
    plot_video_data = np.stack(plot_frames, axis=0)

    # Transpose to (time, channels, height, width) as required by wandb
    plot_video_data = np.transpose(plot_video_data, (0, 3, 1, 2))
    return plot_video_data
    
def log_video(env, params, episode, agent):
    positions = []
    velocities = []
    actions = []
    # env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
    # render_mode = 'rgb_array'
    # video_env = safety_gymnasium.make(env_id, render_mode = render_mode)
    # frames = []
    state, info = env.reset()
    for step in range(250):
        pos = env.task.agent.pos
        positions.append(pos)
        vel = env.task.agent.vel
        velocities.append(vel)

        
        agent.set_pos(pos)
        action, _ = agent.select_action(np.array(state), debug_cbf=False)
        actions.append(action)
        next_state, reward, cost, done, truncated, _ = env.step(action)

        if abs(pos[1]) > 5:
            truncated = True
        
        state = next_state
        if done or truncated:
            break

        # frame = video_env.render()
        # frames.append(frame)

    # video_data = np.stack(frames, axis=0)
    # video_data = np.transpose(video_data, (0, 3, 1, 2))

    plot_video_data = log_canvas(positions, velocities, actions)
    fps = len(plot_video_data) / 10

    # Log video to wandb
    wandb_enabled = params['base']['wandb_enabled']
    if wandb_enabled:
        wandb.log({
            # f"Episode {episode + 1} Ego centric": wandb.Video(video_data, fps=fps, format="mp4"),
                   f"Episode {episode + 1}": wandb.Video(plot_video_data, fps=fps, format="mp4")
                   })


def train(env, agent, dir_name, params, start_episode = 0):
    
    env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')

    num_episodes = params['train'].get('num_episodes', 1000)
    batch_size = params['train'].get('batch_size', 64)
    max_steps_per_episode = params['train'].get('max_steps_per_episode', 250)

    SAVE_EVERY = params['train'].get('save_every', 100)
    RECORD_EVERY = params['train'].get('record_every', 50)

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):

        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        num_safe_actions = 0
        episode_safety_calls = 0


        for t in range(max_steps_per_episode):
            # print('Episode:', episode, 'Step:', t, end= "\r")
            pos = env.task.agent.pos
        
            agent.set_pos(pos)
            action, cbf_optimizer_used = agent.select_action(np.array(state), debug_cbf=False)
            
            next_state, reward, cost, done, truncated, _ = env.step(action)

            if abs(pos[1]) > 5:
                truncated = True

            if cost == 0:
                num_safe_actions += 1

            if cbf_optimizer_used:
                episode_safety_calls += 1

            agent.replay_buffer.add(state, action, reward, next_state, cost, done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            if done or truncated:
                break

        # os.system('cls' if os.name == 'nt' else 'clear')
        if (episode + 1) % RECORD_EVERY == 0:
            for i in range(4):
                log_video(env, params, episode, agent)

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(os.path.join(dir_name , f'{episode + 1}'))

            wandb_enabled = params['base']['wandb_enabled']

            if wandb_enabled:

                local_path = os.path.join(dir_name, f'{episode + 1}')
                run_name = dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")

                for file in os.listdir(os.path.join(dir_name, f'{episode + 1}')):
                    model_path = os.path.join(local_path, file)
                    file_name = os.path.splitext(file)[0]

                    artifact = wandb.Artifact(file_name, type="model")
                    artifact.add_file(model_path)
                    artifact.metadata = {
                        "root": local_path
                    }
                    wandb.log_artifact(artifact, aliases=[run_name + f"-{episode + 1}"])
                

        percent_safe_actions = num_safe_actions / (t+1) * 100
        yield episode, episode_reward, episode_cost, percent_safe_actions, episode_safety_calls
    
    env.close()
    
