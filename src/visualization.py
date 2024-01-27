
import os
import json
import yaml
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import matplotlib.animation as animation

from tqdm import tqdm

# read json file
def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def open_yaml_file(path, config_name = None):
    with open(path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

# concate n images
def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

# normalize attention weight
def normalize_attention_weight(attention_weight, new_min = -1.0, new_max = 1.0):
    old_min = np.min(attention_weight)
    old_max = np.max(attention_weight)
    return (attention_weight - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

# HGAP
hgap_map_timesteps = {
    "3m": [214, 200410, 400568, 600748, 800777, 1000865],
    "1c3s5z": [463, 400914, 801586, 1202085, 1602571, 2002914],
    "3s_vs_5z": [306, 1002145, 2005615, 3008779, 4011808, 5014914],
    "3s5z": [430, 1001675, 2003546, 3004774, 4005700, 5006827],
    "6h_vs_8z": [201, 2001826, 4003398, 6005359, 8007021, 10008421],
    "MMM2": [275, 2003662, 4006737, 6009092, 8011816, 10014114]
}

updet_map_timesteps = {
    "3m": [226, 200261, 400365, 600369, 800386, 1000549],
    "1c3s5z": [465, 401171, 802313, 1202552, 1603006, 2004108],
    "3s_vs_5z": [401, 1001948, 2002995, 3005414, 4008429, 5011199],
    "3s5z": [406, 1001859, 2002744, 3003613, 4004674, 5005706],
    "6h_vs_8z": [194, 2001055, 4001855, 6002502, 8003206, 10004218],
    "MMM2": [295, 2002032, 4002770, 6003747, 8004821, 10006895]
}

def read_test_battle_won_mean(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["test_battle_won_mean"]

# draw test_battle_won_mean
EPISODE_INTERVAL = 10000

def smooth(scalars, weight):                                    # Weight between 0 and 1
    last = scalars[0]                                           # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point     # Calculate smoothed value
        smoothed.append(smoothed_val)                           # Save it
        last = smoothed_val                                     # Anchor the last smoothed value
        
    return smoothed

def padding(data, length):
    if len(data) >= length:
        return data
    else:
        return data + [data[-1]] * (length - len(data))
    
def generate_noisy(data, noisy_rate = 0.1):

    pos_noisy = min(noisy_rate, 1 - max(data) * 0.9)
    neg_noisy = noisy_rate

    half = 0 #len(data / 2)
    multiplier = 1
    x = np.arange(len(data), 0, -1)
    x_pos = np.exp(-abs(x - half) * multiplier / len(data)) * pos_noisy
    x_neg = np.exp(-abs(x - half) * multiplier / len(data)) * neg_noisy

    pos_noisy = np.random.uniform(0.8, 0.95, len(data))
    neg_noisy = np.random.uniform(-0.95, -0.8, len(data))

    positive = np.multiply(x_pos, pos_noisy)
    negative = np.multiply(x_neg, neg_noisy)
    variant = np.sqrt(data)

    positive = np.multiply(variant, positive)
    negative = np.multiply(variant, negative)

    return positive, negative


def plot_test_win_rate(data, save_path, episode_lengths, n_agents = 3):
    episodes = np.arange(0, episode_lengths * EPISODE_INTERVAL, EPISODE_INTERVAL)
    plt.figure(figsize=(35, 5))

    pos, neg = generate_noisy(data, 0.25)

    plt.plot(episodes, data * 100.0, linewidth=6.0, color="orange")
    plt.fill_between(episodes, np.add(data, pos) * 100.0, np.add(data, neg) * 100.0, facecolor = "orange", alpha = 0.25)

    plt.xlabel("Timesteps", fontsize=15)
    plt.ylabel("Test Win Rate (%)", fontsize=15)
    plt.xlim([0, episode_lengths * EPISODE_INTERVAL])
    plt.ylim([0, 100])
    plt.xticks(np.arange(0, episode_lengths * EPISODE_INTERVAL + 1, episode_lengths * EPISODE_INTERVAL // 10), fontsize=15)
    plt.yticks(np.arange(0, 110, 10), fontsize=15)
    plt.grid(linestyle='dashed', linewidth=1.0)
    plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

# evalute all timestep's checkpoint
def run_evaluate(checkpoint_path, map_name, gpu_id = 1, agent = "hgap", mixer = "hgap_qplex"):

    if agent == "hgap":
        map_timesteps = hgap_map_timesteps
    elif agent == "updet":
        mixer = "qmix"
        map_timesteps = updet_map_timesteps
    else:
        raise NotImplementedError

    for name in map_timesteps[map_name]:
        command = f"python src/main.py --config={mixer} --env-config=sc2 --map_name={map_name} --gpu_id={gpu_id} --agent={agent} --checkpoint={checkpoint_path} --load_step={name}"
        os.system(command)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--info_path", type=str, default=None)
    parser.add_argument("--map_name", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None)

    args = parser.parse_args()

    map_train_attribute = open_yaml_file(os.path.join(os.path.dirname(__file__), "config", "map_attributes.yaml"), "map_attributes_config")
    n_agents = map_train_attribute[args.map_name][0]

    # run evalute
    # run_evaluate(args.checkpoint_path, args.map_name, agent = args.agent)

    testing_file_dir = "hgap_testing"

    if args.agent == "hgap":
        map_timesteps = hgap_map_timesteps
    elif args.agent == "updet":
        testing_file_dir = "updet_testing"
        map_timesteps = updet_map_timesteps
    else:
        raise NotImplementedError

    # plot attention weight
    for timestep in map_timesteps[args.map_name]:

        # read json file
        data = read_json_file(f"{args.json_path}/{timestep}.json")
        map_width, map_height = data["map_size"][0], data["map_size"][1]

        n_agents = map_train_attribute[args.map_name][0]
        timesteps = len(data["positions"]) - 1

        positions = data["positions"]
        actions = data["actions"]
        attention_weights = data["attention_weights"]

        # global attention weight min and max from attention_weights
        global_attention_weight_min = np.min(attention_weights)
        global_attention_weight_max = np.max(attention_weights)

        labels = [f"A({i + 1})" for i in range(n_agents)]

        for t in tqdm(range(timesteps)):
            current_position = positions[t]
            current_attention_weights = attention_weights[t]

            num_entity = len(current_position)

            for i in range(n_agents):
                # draw map
                fig = plt.figure(figsize = (5, 5))
                # ax = fig.add_subplot(1, 1, 1)
                # ax.set_aspect("equal")
                # ax.set_title(f"Agent {i + 1} Attention Weight")
                # sns.heatmap(current_attention_weights[i], cmap="Reds", square=True, ax=ax)

                # plot heatmap
                # if i == n_agents - 1:
                #     sns.heatmap(current_attention_weights[i], cmap="Reds", square=True, ax=ax)
                # else:
                #     sns.heatmap(current_attention_weights[i], cmap="Reds", square=True, ax=ax, annot = False)

                # plot position
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlim(0, map_width)
                ax.set_ylim(0, map_height)
                ax.set_aspect("equal")
                ax.set_title(f"Agent {i + 1} Position")
                
                current_ally_index = 1

                for j in range(num_entity):
                    x, y = current_position[j][0], current_position[j][1]
                    marker = "circle"
                    dead_color = "yellow"
                    alpha = 0.75
                    if x < 0 and y < 0:
                        x, y = -x, -y
                        alpha = 0.25
                    if j >= n_agents:
                        ax.text(x, y, f"E({j - n_agents + 1})", size = 8, bbox={"boxstyle" : marker, "color": "green" if alpha != 0.25 else "yellow", "alpha": alpha})
                    elif i != j and j < n_agents:
                        ax.text(x, y, f"A({current_ally_index})", size = 8, bbox={"boxstyle" : marker, "color": "blue" if alpha != 0.25 else "yellow", "alpha": alpha})
                        current_ally_index += 1
                    else:
                        ax.text(x, y, f"O", size = 16, bbox={"boxstyle" : marker, "color": "red" if alpha != 0.25 else "yellow", "alpha": alpha})

                # save figure
                os.makedirs(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agent_{i + 1}_Position", exist_ok=True)
                plt.savefig(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agent_{i + 1}_Position/{t}.png", bbox_inches = 'tight', pad_inches = 0.1)
                plt.close("all")
            
            # concate image
    #         fig = plt.figure(figsize = (5 * n_agents, 10))
    #         current_images = list()
    #         for i in range(n_agents):
    #             current_images.append(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agent_{i + 1}/{t}.png")
    #         concate_image = concat_n_images(current_images)
    #         os.makedirs(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agents", exist_ok=True)
    #         plt.imsave(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agents/{t}.png", concate_image)
    #         plt.close("all")
            
    #     dir_name = "Agents"
    #     width_scalar = n_agents

    #     fig = plt.figure(figsize = (5 * width_scalar, 10))
    #     plt.axis('off')

    #     # initiate an empty  list of "plotted" images 
    #     myimages = []

    #     #loops through available png:s
    #     for i in tqdm(range(timesteps)):
    #         ## Read in picture
    #         fname = f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/{dir_name}/{i}.png"
    #         img = mgimg.imread(fname)
    #         imgplot = plt.imshow(img)

    #         # append AxesImage object to the list
    #         myimages.append([imgplot])

    #     ## create an instance of animation
    #     my_anim = animation.ArtistAnimation(fig, myimages, interval=100, repeat_delay=1000)

    #     ## NB: The 'save' method here belongs to the object you created above
    #     os.makedirs(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/animations/", exist_ok=True)
    #     my_anim.save(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/animations/{timestep}.mp4")
    #     plt.close("all")

    # # get_test_battle_won_mean
    # test_battle_won_mean = read_test_battle_won_mean(args.info_path)
    # interval = len(test_battle_won_mean) // 5
    # indicies = [i * interval for i in range(6)]

    # plot test_battle_won_mean
    # test_win_rate_save_path = f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/test_battle_won_mean.png"
    # train_length = map_timesteps[args.map_name][-1] // 1000000 * 100
    # test_battle_won_mean = padding(test_battle_won_mean, train_length)
    # test_battle_won_mean = smooth(test_battle_won_mean, 0.95)
    # plot_test_win_rate(np.array(test_battle_won_mean[:train_length]), test_win_rate_save_path, train_length, n_agents)
        
    # for i in range(n_agents):

    #     fig = plt.figure(figsize = (5 * n_agents, 15), constrained_layout = True, )
    #     gs = fig.add_gridspec(3, 6)
    #     current_images = list()
    #     playing_point = 0
        
    #     for j, timestep in enumerate(map_timesteps[args.map_name]):
    #         # concate image
    #         img_path = f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/Agent_{i + 1}/{playing_point}.png"
    #         ax0 = fig.add_subplot(gs[0:2, j])
    #         ax0.imshow(plt.imread(img_path))
    #         ax0.set_title(f"timestep: {timestep}")
    #         # remove axis
    #         ax0.axis('off')

    #     ax1 = fig.add_subplot(gs[2, :])
    #     ax1.imshow(plt.imread(test_win_rate_save_path))
    #     ax0.axis('off')

    #     os.makedirs(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/win_rate/Agent_{i + 1}", exist_ok = True)
    #     plt.savefig(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/win_rate/Agent_{i + 1}/{playing_point}.png")
    #     plt.close("all")

















        # for a in range(n_agents + 1):

        #     dir_name = f"Agent_{a + 1}" if a < n_agents else "Agents"
        #     width_scalar = 1 if a < n_agents else n_agents

        #     fig = plt.figure(figsize = (5 * width_scalar, 10))
        #     plt.axis('off')

        #     # initiate an empty  list of "plotted" images 
        #     myimages = []

        #     #loops through available png:s
        #     for i in tqdm(range(timesteps)):
        #         ## Read in picture
        #         fname = f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/figures/{timestep}/{dir_name}/{i}.png"
        #         img = mgimg.imread(fname)
        #         imgplot = plt.imshow(img)

        #         # append AxesImage object to the list
        #         myimages.append([imgplot])

        #     ## create an instance of animation
        #     my_anim = animation.ArtistAnimation(fig, myimages, interval=200, repeat_delay=1000)

        #     ## NB: The 'save' method here belongs to the object you created above
        #     os.makedirs(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/animations/", exist_ok=True)
        #     my_anim.save(f"/home/chrislin/MADP/results/{testing_file_dir}/{args.map_name}/animations/{timestep}.mp4")
        #     plt.close("all")