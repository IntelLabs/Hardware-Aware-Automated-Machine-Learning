# SQFT - Neural Low-Rank Adapter Search (NLS) in OPEA

To facilitate the utilization of SQFT's Neural Low-Rank Adapter Search (NLS) algorithm and quickly benefit from it, 
[Open Platform for Enterprise AI (OPEA)](https://github.com/opea-project) is supporting model fine-tuning with NLS strategies as one of the Generative AI solutions ([link](https://github.com/opea-project/GenAIComps/tree/main/comps/finetuning#322-instruction-tuning-with-sqfts-neural-low-rank-adapter-search-nls)).
The fine-tuning microservice with NLS involves adapting the model to specific tasks or datasets to improve its performance on those tasks, with the following features:
- Make the adapter elastic
- Explore more adapter configurations
- Outperform LoRA (especially in sparse model fine-tuning)
- Alleviate the challenge users face to set the rank when using LoRA

More details about SQFT can be found in this [paper](https://aclanthology.org/2024.findings-emnlp.749/). 

## 🚀 Start the NLS Microservice with Python

Follow [these instructions](https://github.com/opea-project/GenAIComps/tree/main/comps/finetuning#1-start-microservice-with-python-option-1) to start the microservices.
The difference is that when installing the requirements, additional dependencies need to be installed to enable NLS:

```bash
git clone https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning.git haaml
cp -r haaml/SQFT/patches . && rm -rf haaml
PATH_TO_FINETUNE=$PWD
mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.44.2 && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/transformers-v4.44.2.patch && pip install -e . && cd ..

# peft
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.10.0 && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/peft-v0.10.0.patch && pip install -e . && cd ..

# nncf
git clone https://github.com/openvinotoolkit/nncf.git
cd nncf && git checkout f143e1c && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/nncf-f143e1c.patch && pip install -e . && cd ..
```

## 🚀 Consume the NLS Fine-tuning Service

We use LLama-3.2-1B to fine-tune the [Arc-E](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy) dataset as a simple example to demonstrate how to use NLS on OPEA Microservice.

### Prepare and upload a training file

First, we need to process the training dataset into the instruction format, for example:

```json
{
    "instruction": "Which factor will most likely cause a person to develop a fever?",
    "input": "",
    "output": "a bacterial population in the bloodstream"
}
```
Here, we use the Arc-E dataset as an example. The processing of the Arc-E training set is performed via the script [dataset/preprocess_arc.py](./dataset/preprocess_arc.py). 
After obtaining the processed dataset file [arce_train_instruct.json](./dataset/arce_train_instruct.json), we can upload it to the server with this command:
```bash
# upload a training file
curl http://<your ip>:8015/v1/files -X POST -H "Content-Type: multipart/form-data" -F "file=@dataset/arce_train_instruct.json" -F purpose="fine-tune"
```

### Create an NLS fine-tuning job

For model selection, we offer several sparse and quantized base models for users to choose from (the complete SQFT scheme). 
Please refer to [this Table](../README.md#released-foundation-models-) or the [HuggingFace SQFT Model Collection](https://huggingface.co/collections/IntelLabs/sqft-66cd56f90b240963f9cf1a67).
For the service this demonstration, we use model [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).
After uploading a training file, use the following command to launch a fine-tuning job using `meta-llama/Llama-3.2-1B` as the base model.

```bash
# create a fine-tuning job (Neural Low-rank adapter Search)
# Max LoRA rank: 16
#   LoRA target modules            -> Low-rank search space
#   ["q_proj", "k_proj", "v_proj"] -> [16,12,8]
curl http://<your ip>:8015/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "arce_train_instruct.json",
    "model": "meta-llama/Llama-3.2-1B",
    "General": {
      "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "neural_lora_search": true,
        "target_module_groups": [["q_proj", "k_proj", "v_proj"]],
        "search_space": ["16,12,8"]
      }
    },
    "Training": {
      "learning_rate": 1e-04, "epochs": 5, "batch_size": 16
    },
    "Dataset": {
      "max_length": 256
    }
  }'
```

Below are some explanations for the parameters related to the Neural Low-rank adapter Search algorithm:

- `neural_lora_search` indicates whether the Neural LoRA Search (NLS) algorithm is enabled.
- `target_module_groups` specifies the target module groups, which means that the adapters within the same group will share the same activated low-rank value.
- `search_space` specifies the search space for each target module (adapter) group. Here, we use `["16,12,8"]`, meaning that the possible rank for each group is [16, 12, 8].


<details>
<summary>A more advanced example</summary>

```bash
# create a fine-tuning job with NLS
# Max LoRA rank: 16
#   LoRA target modules            -> Low-rank search space
#   ["q_proj", "k_proj", "v_proj"] -> [16,12,8]
#   ["up_proj"]                    -> [16,12,8]
#   ["down_proj"]                  -> [16,12,8]
curl http://<your ip>:8015/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "arce_train_instruct.json",
    "model": "meta-llama/Llama-3.2-1B",
    "General": {
      "lora_config": {
        "r": 16,
        "neural_lora_search": true,
        "target_module_groups": [["q_proj", "k_proj", "v_proj"], ["up_proj"], ["down_proj"]],
        "search_space": ["16,12,8", "16,12,8", "16,12,8"]
      }
    }
  }'
```

This example sets up three dependency adapter groups:
1. `q_proj`, `k_proj`, `v_proj`
2. `up_proj`
3. `down_proj`

`search_space` is `["16,12,8", "16,12,8", "16,12,8"]`, meaning that the search space for each adapter group is [16, 12, 8].

</details>

Note that the number of groups should be equal to the number of search spaces (one-to-one correspondence).
Feel free to try your favorite group design and search spaces.

### Leverage the Fine-tuned Super-Adapter

#### Extract a Sub-Adapter

After completing the fine-tuning stage and obtaining an NLS super-adapter, the next step is extract a desired sub-adapter. The following command demonstrates how to extract the heuristic sub-adapter.
Additionally, more powerful sub-adapters can be obtained through other advanced search algorithms (more details can be found in [here](#search)).

```bash
curl http://<your ip>:8015/v1/finetune/extract_sub_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "heuristic"
  }'
```

`sub_adapter_version` can be heuristic, minimal, or a custom name.
When `sub_adapter_version` is set to a custom name, we need to provide a specific configuration in `custom_config`.
The extracted adapter will be saved in `<path to the output directory for this job> / <sub_adapter_version>`.

<details>
<summary>An example of a custom configuration</summary>

```bash
curl http://<your ip>:8015/v1/finetune/extract_sub_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "optimal",
    "custom_config": [16, 12, 8, 8, 16, 12, 16, 16, 16, 16, 16, 8, 16, 12, 8, 12],
  }'
```

In the fine-tuning job with the Neural Low-rank adapter Search algorithm,  the `nncf_config.json` file (which includes the elastic adapter information) will be saved in the job's output directory.
The `custom_config` must correspond with the `overwrite_groups` (adapter modules) or `overwrite_groups_widths`
(search space for the rank of adapter modules) in `nncf_config.json`. 
The above command corresponds to the example in [example_nncf_config/nncf_config.json](./example_nncf_config/nncf_config.json), where the custom config `[16, 12, 8, 8, 16, 12, 16, 16, 16, 16, 16, 8, 16, 12, 8, 12]` represents the LoRA rank size of the adapters for `q_proj`, `k_proj`, and `v_proj` in each layer.
It will save the sub-adapter to `<path to the output directory for this job> / optimal`.

</details>

#### Merge Adapter to Base Model

The following command demonstrates how to merge a sub-adapter (using the Heuristic sub-adapter as an example) into the base pre-trained model to obtain the final fine-tuned model:

```bash
curl http://<your ip>:8015/v1/finetune/merge_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "heuristic"
  }'
```

The merged model will be saved in `<path to the output directory for this job> / <sub_adapter_version> / merged_model`.

#### Search

To further explore for high-performing sub-adapter configurations within the super-adapter, we can utilize more advanced search algorithms to search the super-adapter.
Due to the flexibility and wide range of choices in the search settings, the service does not support the search process (but it supports providing a specific sub-adapter configuration to extract the sub-adapter; refer to [here](#extract-sub-adapter)).
The search needs to be conducted service-externally according to user preferences.

In our example, we provide a simple script ([search.py](./search.py)) for the search with Arc-E validation set to obtain some optimal sub-adapters.
The command is as follows:

```bash
python search.py \
  --base_model_path meta-llama/Llama-3.2-1B \
  --adapter_model_path <path to super adapter> \
  --output_dir <path to search results>
```
