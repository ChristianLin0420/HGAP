
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
map_timesteps = {
    "3m": [268, 200538, 400734, 600996, 801248, 1001502],
    "1c3s5z": [426, 401218, 802976, 1203451, 1603960, 2004441],
    "3s_vs_5z": [489, 1006752, 2014620, 3023956, 4030443, 5037615],
    "3s5z": [401, 1001483, 2002433, 3003678, 4004666, 5005375],
    "6h_vs_8z": [180, 2001332, 4002692, 6004190, 8005583, 10007880],
    "corridor": []
}

# evalute all timestep's checkpoint
def run_evaluate(checkpoint_path, map_name, gpu_id = 1, agent = "hgap", mixer = "hgap_qplex"):

    for name in map_timesteps[map_name]:
        command = f"python src/main.py --config={mixer} --env-config=sc2 --map_name={map_name} --gpu_id={gpu_id} --agent={agent} --checkpoint={checkpoint_path} --load_step={name}"
        print(command)
        os.system(command)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--map_name", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)

    args = parser.parse_args()

    # run evalute
    run_evaluate(args.checkpoint_path, args.map_name)

    for timestep in map_timesteps[args.map_name]:

        # read json file
        data = read_json_file(f"{args.json_path}/{timestep}.json")

        map_width, map_height = data["map_size"][0], data["map_size"][1]

        map_train_attribute = open_yaml_file(os.path.join(os.path.dirname(__file__), "config", "map_attributes.yaml"), "map_attributes_config")

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
                fig = plt.figure(figsize = (5, 10))
                ax = fig.add_subplot(2, 1, 1)
                ax.set_aspect("equal")
                ax.set_title(f"Agent {i + 1} Attention Weight")

                # plot heatmap
                if i == n_agents - 1:
                    sns.heatmap(normalize_attention_weight(current_attention_weights[i]), cmap="Reds", square=True, ax=ax)
                else:
                    sns.heatmap(normalize_attention_weight(current_attention_weights[i]), cmap="Reds", square=True, ax=ax, annot = False)

                # plot position
                ax = fig.add_subplot(2, 1, 2)
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
                        # marker = "rectangle"
                        x, y = -x, -y
                        alpha = 0.25
                    if j >= n_agents:
                        # ax.scatter(x, y, c="black", s=50, marker=marker)
                        ax.text(x, y, f"E({j - n_agents + 1})", size = 8, bbox={"boxstyle" : marker, "color": "green" if alpha != 0.25 else "yellow", "alpha": alpha})
                    elif i != j and j < n_agents:
                        # ax.scatter(x, y, c="blue", s=50, marker=marker)
                        ax.text(x, y, f"A({current_ally_index})", size = 8, bbox={"boxstyle" : marker, "color": "blue" if alpha != 0.25 else "yellow", "alpha": alpha})
                        current_ally_index += 1
                    else:
                        # ax.scatter(x, y, c="red", s=100, marker=marker)
                        ax.text(x, y, f"O", size = 16, bbox={"boxstyle" : marker, "color": "red" if alpha != 0.25 else "yellow", "alpha": alpha})

                # save figure
                os.makedirs(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/Agent_{i + 1}", exist_ok=True)
                plt.savefig(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/Agent_{i + 1}/{t}.png", bbox_inches = 'tight', pad_inches = 0)
                plt.close("all")
            
            # concate image
            fig = plt.figure(figsize = (5 * n_agents, 10))
            current_images = list()
            for i in range(n_agents):
                current_images.append(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/Agent_{i + 1}/{t}.png")
            concate_image = concat_n_images(current_images)
            os.makedirs(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/Agents", exist_ok=True)
            plt.imsave(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/Agents/{t}.png", concate_image)
            plt.close("all")


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
        #         fname = f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/{dir_name}/{i}.png"
        #         img = mgimg.imread(fname)
        #         imgplot = plt.imshow(img)

        #         # append AxesImage object to the list
        #         myimages.append([imgplot])

        #     ## create an instance of animation
        #     my_anim = animation.ArtistAnimation(fig, myimages, interval=200, repeat_delay=1000)

        #     ## NB: The 'save' method here belongs to the object you created above
        #     os.makedirs(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/animations/", exist_ok=True)
        #     my_anim.save(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/animations/{timestep}.mp4")
        #     plt.close("all")
            
        dir_name = "Agents"
        width_scalar = n_agents

        fig = plt.figure(figsize = (5 * width_scalar, 10))
        plt.axis('off')

        # initiate an empty  list of "plotted" images 
        myimages = []

        #loops through available png:s
        for i in tqdm(range(timesteps)):
            ## Read in picture
            fname = f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/figures/{timestep}/{dir_name}/{i}.png"
            img = mgimg.imread(fname)
            imgplot = plt.imshow(img)

            # append AxesImage object to the list
            myimages.append([imgplot])

        ## create an instance of animation
        my_anim = animation.ArtistAnimation(fig, myimages, interval=200, repeat_delay=1000)

        ## NB: The 'save' method here belongs to the object you created above
        os.makedirs(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/animations/", exist_ok=True)
        my_anim.save(f"/home/chrislin/MADP/results/hgap_testing/{args.map_name}/animations/{timestep}.mp4")
        plt.close("all")
        

