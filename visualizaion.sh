
#### HGAP ####

map="3m"
infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/3m/dmaq/1/info.json"
checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/3m/dmaq/hgap_qplex__2024-01-22_10-53-45"
agent="hgap"

python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="1c3s5z"
# infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/1c3s5z/dmaq/1/info.json"
# checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/1c3s5z/dmaq/hgap_qplex__2024-01-22_10-54-11"
# agent="hgap"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="3s_vs_5z"
# infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/3s_vs_5z/dmaq/1/info.json"
# checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/3s_vs_5z/dmaq/hgap_qplex__2024-01-22_10-57-49"
# agent="hgap"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="3s5z"
# infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/3s5z/dmaq/1/info.json"
# checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/3s5z/dmaq/hgap_qplex__2024-01-22_11-27-46"
# agent="hgap"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="6h_vs_8z"
# infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/6h_vs_8z/dmaq/2/info.json"
# checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/6h_vs_8z/dmaq/hgap_qplex__2024-01-22_10-30-51"
# agent="hgap"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="MMM2"
# checkpoint="/home/chrislin/MADP/results/hgap_visualization/models/MMM2/dmaq/hgap_qplex__2024-01-19_11-49-30"
# infopath="/home/chrislin/MADP/results/hgap_visualization/sacred/MMM2/dmaq/2/info.json"
# agent="hgap"

python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/hgap_testing/$map/records/ --info_path=$infopath --agent=$agent




#### UPDeT ####

# map="3m"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/3m/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/3m/qmix/qmix__2024-01-19_11-29-37"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="1c3s5z"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/1c3s5z/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/1c3s5z/qmix/qmix__2024-01-19_11-29-12"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="3s_vs_5z"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/3s_vs_5z/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/3s_vs_5z/qmix/qmix__2024-01-19_16-24-25"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="3s5z"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/3s5z/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/3s5z/qmix/qmix__2024-01-19_20-25-29"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="6h_vs_8z"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/6h_vs_8z/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/6h_vs_8z/qmix/qmix__2024-01-19_11-28-55"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent

# map="MMM2"
# infopath="/home/chrislin/MADP/results/updet_visualization/sacred/MMM2/qmix/1/info.json"
# checkpoint="/home/chrislin/MADP/results/updet_visualization/models/MMM2/qmix/qmix__2024-01-19_11-28-22"
# agent="updet"

# python src/visualization.py --checkpoint=$checkpoint --map_name=$map --json_path=/home/chrislin/MADP/results/updet_testing/$map/records/ --info_path=$infopath --agent=$agent