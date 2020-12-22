# visual_localization

visual_localization conntains 3 parts:
 * `extract_features.py`
 * `eval.py`
 * `eval_superglue.py`


### `extract_features.py`
This part extracts global descriptor of a image.
```
usage: extract_feature.py [-h] [--floor FLOOR] [--globaldesc GLOBALDESC] [--batchsize BATCHSIZE] 
                          [--dataset_path DATASET_PATH] [--checkpoint_path CHECKPOINT_PATH]
```
Arguments:
* `--floor`: Which floor: 1f or b1?
* `--globaldesc`: Global descriptor to use: `netvlad` or `apgem`
* `--batchsize`: Batch size
* `--dataset_path`: Path to dataset directory (parent directory of 1f and b1)
* `--checkpoint_path`: Path to checkpoint

Example:
```
python extract_feature.py --floor b1 --globaldesc netvlad --batchsize 32 \
--dataset_path ../dataset --checkpoint_path ../checkpoints/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar
```
It saves global descriptor of each db and query image as 2d numpy array. The saved file name is `{globaldesc}_db_{floor}_features.npy` and `{globaldesc}_query_{floor}_features.npy`

### `eval.py`
This part performs pose estimation using rootSIFT. It need result of `extract_features.py`.
```
usage: eval.py [-h] [--floor FLOOR] [--globaldesc GLOBALDESC] [--rank_knn_num RANK_KNN_NUM] 
               [--rerank_knn_num RERANK_KNN_NUM] [--dataset_path DATASET_PATH]
```
Arguments:
* `--floor`: Which floor: 1f or b1?
* `--globaldesc`: Global descriptor to use: `netvlad` or `apgem`
* `--dataset_path`: Path to dataset directory (parent directory of 1f and b1)
* `--rank_knn_num`: Number of nearest neighbor to select at ranking phase
* `--rerank_knn_num`: Number of nearest neighbor to select at reranking phase

Example:
```
python eval.py --floor b1 --globaldesc apgem --rank_knn_num 10 \
--rerank_knn_num 5 --dataset_path ../dataset
```

It saves answer as json file with name `{globaldesc}_rootsift_rank_knn_num_{rank_knn_num}_rerank_knn_num_{rerank_knn_num}_answer_{floor}.json`. It also saves intermediate result on every 100 image evaluations.


### `eval_superglue.py`
This part performs pose estimation using SuperGlue. It need result of `extract_features.py`.
```
usage: eval.py [-h] [--floor FLOOR] [--globaldesc GLOBALDESC] [--rank_knn_num RANK_KNN_NUM] 
               [--rerank_knn_num RERANK_KNN_NUM] [--dataset_path DATASET_PATH]
```
Arguments:
* `--floor`: Which floor: 1f or b1?
* `--globaldesc`: Global descriptor to use: `netvlad` or `apgem`
* `--dataset_path`: Path to dataset directory (parent directory of 1f and b1)
* `--rank_knn_num`: Number of nearest neighbor to select at ranking phase
* `--rerank_knn_num`: Number of nearest neighbor to select at reranking phase

Example:
```
python eval_superglue.py --floor b1 --globaldesc apgem --rank_knn_num 10 \
--rerank_knn_num 5 --dataset_path ../dataset
```

It saves answer as json file with name `{globaldesc}_superglue_rank_knn_num_{rank_knn_num}_rerank_knn_num_{rerank_knn_num}_answer_{floor}.json`. It also saves intermediate result on every 100 image evaluations.
