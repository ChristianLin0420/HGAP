
export CUDA_VISIBLE_DEVICES=0,1
train_count=3
train_index=0
# agent='hpns_attention'
# mixers='hpn_attention_qmix'
agent='hgap'
mixers='hgap_qmix'
maps='3s_vs_5z 5m_vs_6m'
gpu_id=0

### 1c3s5z 2s3z 3m 8m 3s_vs_3z 
### 3s_vs_5z 3s5z 5m_vs_6m 
### 8m_vs_9m 10m_vs_11m 25m 
### 3s5z_vs_3s6z 6h_vs_8z 
### 27m_vs_30m bane_vs_bane 
### corridor MMM2

for i in $(seq 1 $train_count); do 
    for mixer in $mixers; do
        for map in $maps; do
            python src/main.py --config=$mixer --env-config=sc2 --map_name=$map --gpu_id=$gpu_id --agent=$agent
        done
    done
done