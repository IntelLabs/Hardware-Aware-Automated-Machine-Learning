import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizer
)
import faiss
from peft import LoraConfig, get_peft_model
from PIL import Image

from activeft.sift import Retriever
from ct3_config import CT3Config
from data.data_qwen import DataCollatorForSupervisedDataset, LazySupervisedDataset
from kb.utils import load_knowledge_base
from models.embedder import CLIPViTEmbedder, load_clip_vit_embedder, get_multimodal_embedding
from models.local_adapter_manager import LocalAdapterManager


class TTTModel(torch.nn.Module):
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        processor: AutoProcessor,
        disable_ttt: bool = False,
        embedder: CLIPViTEmbedder = None, 
        knowledge_base_samples: List[str] = None, 
        retriever: Retriever = None, 
        num_ttt_samples: int = 32, 
        num_epochs: int = 2, 
        learning_rate: float = 5e-5, 
        batch_size: int = 1,
        shuffle_ttt_samples: bool = False, 
        data_reduction: bool = False, 
        weighted_sample_training: bool = False,
        local_adapter_managing: bool = False,
        local_adapter_manager_size: int = 4,
        local_adapter_manager_threshold: float = 0.4,
        local_adapter_manager_strategy: str = "lfu",
        cache_search_info: Optional[Dict[str, Any]] = None
    ) -> None:
        super(TTTModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.disable_ttt = disable_ttt

        self.embedder = embedder
        self.knowledge_base_samples = knowledge_base_samples
        self.retriever = retriever

        self.num_ttt_samples = num_ttt_samples
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.shuffle_ttt_samples = shuffle_ttt_samples
        self.data_reduction = data_reduction
        self.weighted_sample_training = weighted_sample_training

        self.local_adapter_managing = local_adapter_managing
        self.local_adapter_manager = None
        if local_adapter_managing:
            self.local_adapter_manager = LocalAdapterManager(
                local_adapter_manager_size,
                local_adapter_manager_threshold,
                local_adapter_manager_strategy
            )
        
        self.cache_search_info = cache_search_info

        self.trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.trainable_param_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        self.original_state_dict = {
            name: param.clone() for name, param in self.model.named_parameters() if name in self.trainable_param_names
        }

    def forward(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs, generate=True)

    def _forward_or_generate(self, *args, generate=False, **kwargs):
        visual_list = kwargs.pop("visual_list")
        if self.disable_ttt:
            # Ensure the original state
            self.model.load_state_dict(self.original_state_dict, strict=False)
        else:

            if 'input_ids' in kwargs:
                input_ids = kwargs['input_ids']
            elif args and len(args) > 0:
                input_ids = args[0]
            else:
                raise ValueError("Neither 'input_ids' in kwargs nor a non-empty args provided")

            assert input_ids.size(0) == 1, "Only supports batch size 1."

            query = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            text_embedding = self.embedder([query]).cpu()
            assert isinstance(visual_list, list) and all(isinstance(v, Image.Image) for v in visual_list), \
                "visual_list must be a list of PIL Image objects."
            query_embeddings = []
            for visual in visual_list:
                image_embedding = self.embedder(visual).cpu()
                query_embedding = get_multimodal_embedding(text_embedding, image_embedding)
                query_embeddings.append(query_embedding)

            # Check if we can reuse weights from the cache
            best_cache_entry = None
            if self.local_adapter_managing:
                # best_cache_entry = self.local_adapter_manager.retrieve_best_cache_entry(query_embedding)
                raise NotImplementedError("Local adapter managing is not implemented yet.")
            if best_cache_entry is not None:
                # print("Reusing weights from the cache.")
                adapter_params = best_cache_entry[2]
                self.model.load_state_dict(adapter_params, strict=False)
            else:
                # Ensure the original state before TTT
                self.model.load_state_dict(self.original_state_dict, strict=False)
                retrieved_samples, _ = self.retrieval(query_embeddings)
                
                if self.data_reduction:
                    scores = [retrieved_sample[1] for retrieved_sample in retrieved_samples]
                    mean_score = np.mean(scores)
                    median_score = np.median(scores)
                    adaptive_threshold = min(mean_score, median_score)
                    retrieved_samples = [
                        retrieved_sample for retrieved_sample in retrieved_samples if retrieved_sample[1] > adaptive_threshold
                    ]

                num_fine_tune_samples = len(retrieved_samples)
                # Fine-tune the model
                if num_fine_tune_samples > 0:
                    fine_tune_samples = [retrieved_sample[0] for retrieved_sample in retrieved_samples]
                    sample_weights = None
                    if self.weighted_sample_training:
                        scores = [retrieved_sample[1] for retrieved_sample in retrieved_samples]
                        sample_weights = torch.tensor(scores, dtype=torch.float32)
                        sample_weights = sample_weights / sample_weights.sum()  # Normalize sample weights
                    
                    self.fine_tune(fine_tune_samples, sample_weights=sample_weights)
                
                    if self.local_adapter_managing:
                        # Save the fine-tuned parameters to the cache
                        adapter_params = {
                            name: param.clone() for name, param in self.model.named_parameters() if name in self.trainable_param_names
                        }
                        self.local_adapter_manager.add_to_cache(query, query_embedding, adapter_params)
            
        # Perform the forward pass or generate
        if generate:
            output = self.model.generate(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        
        return output

    def retrieval(self, query_embeddings: List[torch.Tensor]) -> Tuple[List[Tuple[Any, float]], List[int]]:
        """
        Perform retrieval for each query embedding. If only one embedding is provided, 
        return its retrieval result directly. Otherwise, merge all results, re-rank by score, and return the top-N.
        Args:
            query_embeddings (List[torch.Tensor]): List of query embedding tensors.
        Returns:
            Tuple[List[Tuple[Any, float]], List[int]]: Retrieved samples and their indices.
        """
        if len(query_embeddings) == 1:
            # Fast path for a single query embedding
            values, indices, _, _ = self.retriever.search(query_embeddings[0], N=self.num_ttt_samples, K=1000)
            scores = [abs(float(val)) for val in values]
            indices = [int(idx) for idx in indices]
            retrieved_samples = [(self.knowledge_base_samples[idx], scores[i]) for i, idx in enumerate(indices)]
            return retrieved_samples, indices

        # Multi-query path: merge and re-rank
        all_results = []
        for query_embedding in query_embeddings:
            values, indices, _, _ = self.retriever.search(query_embedding, N=self.num_ttt_samples, K=1000)
            scores = [abs(float(val)) for val in values]
            indices = [int(idx) for idx in indices]
            for i, idx in enumerate(indices):
                all_results.append((self.knowledge_base_samples[idx], scores[i], idx))

        # Sort all results by score in descending order
        all_results.sort(key=lambda x: x[1], reverse=True)
        # Select the top num_ttt_samples results
        top_results = all_results[:self.num_ttt_samples]
        retrieved_samples = [(item[0], item[1]) for item in top_results]
        indices = [item[2] for item in top_results]
        return retrieved_samples, indices

    def fine_tune(self, training_samples: List[Dict[str, Any]], sample_weights: Optional[torch.Tensor] = None) -> None:        
        self.model.train()
        optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)
        num_fine_tune_samples = len(training_samples)
        dataset = LazySupervisedDataset(
            samples=training_samples, 
            model_type=self.model.config.model_type,
            tokenizer=self.tokenizer,
            processor=self.processor
        )
        with torch.enable_grad():  # Ensure gradients are enabled
            for epoch in range(self.num_epochs):
                for i, batch in enumerate(
                        torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=self.batch_size, 
                            collate_fn=self.data_collator, 
                            shuffle=self.shuffle_ttt_samples
                        )
                    ):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    print(f"training loss: {loss}")

                    if sample_weights is not None:
                        # Apply sample weights
                        assert self.batch_size == 1
                        weight = sample_weights[i].to(self.model.device)
                        loss = loss * weight * num_fine_tune_samples

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    optimizer.step()
        self.model.eval()
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class TTTModelAnalyzer(TTTModel):
    def __init__(self, *args, log_file: str = "analysis.json", **kwargs) -> None:
        super(TTTModelAnalyzer, self).__init__(*args, **kwargs)
        assert not self.disable_ttt, "TTTModelAnalyzer should not be used with disable_ttt=True."
        self.analysis_data = []
        self.log_file = log_file

    def _forward_or_generate(self, *args, generate=False, **kwargs):
        visual_list = kwargs.pop("visual_list")
        start_time = time.time()
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif args and len(args) > 0:
            input_ids = args[0]
        else:
            raise ValueError("Neither 'input_ids' in kwargs nor a non-empty args provided")

        assert input_ids.size(0) == 1, "Only supports batch size 1."

        embedding_start_time = time.time()
        query = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        text_embedding = self.embedder([query]).cpu()
        assert isinstance(visual_list, list) and all(isinstance(v, Image.Image) for v in visual_list), \
            "visual_list must be a list of PIL Image objects."
        query_embeddings = []
        for visual in visual_list:
            image_embedding = self.embedder(visual).cpu()
            query_embedding = get_multimodal_embedding(text_embedding, image_embedding)
            query_embeddings.append(query_embedding)
        embedding_end_time = time.time()

        query_data = {
            "query": query,
            "used_cache": False,
            "cache_query": None,
            "retrieved_indices": None,
            "fine_tune_samples": None,
            "retrieved_scores": None,
            "losses": [],
            "timing": {
                "embedding": embedding_end_time - embedding_start_time
            },
            "is_multiple_visuals": len(visual_list) > 1,
        }

        # Check if we can reuse weights from the cache
        best_cache_entry = None
        if self.local_adapter_managing:
            # cache_start_time = time.time()
            # best_cache_entry = self.local_adapter_manager.retrieve_best_cache_entry(query_embedding)
            # cache_end_time = time.time()
            # query_data["timing"]["cache_retrieval"] = cache_end_time - cache_start_time
            raise NotImplementedError("Local adapter managing is not implemented yet.")
        if best_cache_entry is not None:
            # print("Reusing weights from the cache.")
            adapter_params = best_cache_entry[2]
            self.model.load_state_dict(adapter_params, strict=False)
            query_data["used_cache"] = True
            cache_query = best_cache_entry[0]
            query_data["cache_query"] = cache_query
        else:
            # Ensure the original state before TTT
            self.model.load_state_dict(self.original_state_dict, strict=False)

            retrieval_start_time = time.time()
            retrieved_samples, indices = self.retrieval(query_embeddings)
            retrieval_end_time = time.time()
            query_data["timing"]["kb_retrieval"] = retrieval_end_time - retrieval_start_time

            if self.data_reduction:
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                adaptive_threshold = min(mean_score, median_score)
                retrieved_samples = [
                    retrieved_sample for retrieved_sample in retrieved_samples if retrieved_sample[1] > adaptive_threshold
                ]
                indices = [
                    idx for i, idx in enumerate(indices) if retrieved_samples[i][1] > adaptive_threshold
                ]
    
            scores = [retrieved_sample[1] for retrieved_sample in retrieved_samples]
            query_data["retrieved_indices"] = indices
            query_data["retrieved_scores"] = scores
            
            num_fine_tune_samples = len(retrieved_samples)
            # Fine-tune the model
            if num_fine_tune_samples > 0:
                fine_tune_samples = [retrieved_sample[0] for retrieved_sample in retrieved_samples]
                query_data["fine_tune_samples"] = fine_tune_samples.copy()
                sample_weights = None
                if self.weighted_sample_training:
                    scores = [retrieved_sample[1] for retrieved_sample in retrieved_samples]
                    sample_weights = torch.tensor(list(scores), dtype=torch.float32)
                    sample_weights = sample_weights / sample_weights.sum()  # Normalize sample weights

                fine_tune_start_time = time.time()
                self.fine_tune(fine_tune_samples, sample_weights=sample_weights, query_data=query_data)
                fine_tune_end_time = time.time()
                
                query_data["timing"]["fine_tuning"] = fine_tune_end_time - fine_tune_start_time
            
                if self.local_adapter_managing:
                    # Save the fine-tuned parameters to the cache
                    adapter_params = {name: param.clone() for name, param in self.model.named_parameters() if name in self.trainable_param_names}
                    self.local_adapter_manager.add_to_cache(query, query_embedding, adapter_params)

        # Perform the forward pass or generate
        forward_start_time = time.time()
        if generate:
            output = self.model.generate(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        forward_end_time = time.time()
        query_data["timing"]["forward_pass"] = forward_end_time - forward_start_time

        self.analysis_data.append(query_data)
        end_time = time.time()
        query_data["timing"]["total"] = end_time - start_time
        return output

    def fine_tune(self, training_samples: List[Dict[str, Any]], sample_weights: Optional[torch.Tensor] = None, query_data: Optional[Dict[str, Any]] = None) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)
        num_fine_tune_samples = len(training_samples)
        dataset = LazySupervisedDataset(
            samples=training_samples, 
            model_type=self.model.config.model_type,
            tokenizer=self.tokenizer,
            processor=self.processor
        )
        with torch.enable_grad():  # Ensure gradients are enabled
            for epoch in range(self.num_epochs):
                for i, batch in enumerate(
                    torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=self.batch_size, 
                        collate_fn=self.data_collator, 
                        shuffle=self.shuffle_ttt_samples
                    )
                ):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    if sample_weights is not None:
                        # Apply sample weights
                        assert self.batch_size == 1
                        weight = sample_weights[i].to(self.model.device)
                        loss = loss * weight * num_fine_tune_samples

                    if query_data is not None:
                        query_data["losses"].append(loss.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    optimizer.step()
        self.model.eval()

    def save_analysis_data(self, output_path: str) -> None:
        with open(os.path.join(output_path, self.log_file), 'w') as f:
            json.dump(self.analysis_data, f, indent=4)


def convert_to_ttt_model(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    processor: AutoProcessor,
    ct3_config: CT3Config, 
    ct3_server_url: str = None
) -> TTTModel:

    if not ct3_config.ttt:
        model = TTTModel(
            model, 
            tokenizer,
            processor,
            disable_ttt=True
        )
        return model

    model.config.use_cache = False
    for name, param in model.named_parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=ct3_config.lora_rank,
        lora_alpha=ct3_config.lora_alpha,
        lora_dropout=0.05,
        target_modules=ct3_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        if 'visual' in name:
            param.requires_grad = False
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    embedder = load_clip_vit_embedder()

    knowledge_base_samples = None
    retriever = None
    if ct3_server_url is None:
        # Initialize FAISS index for similarity search
        if ct3_config.faiss_index == "IndexFlatIP":
            knowledge_base_samples, embs = load_knowledge_base(ct3_config.kb_paths)
            assert len(knowledge_base_samples) == len(embs), \
                "Knowledge base samples and embeddings must have the same length."
            index = faiss.IndexFlatIP(embedder.dimension * 2)
            index.add(embs)
            retriever = Retriever(index, only_faiss=ct3_config.only_faiss)
        else:
            raise ValueError(f"Unsupported Faiss Index: {ct3_config.faiss_index}")
    else:
        raise NotImplementedError("Remote server support is not implemented yet.")
    
    cache_search_info = None
    if ct3_config.cache_search_info_path is not None:
        with open(ct3_config.cache_search_info_path, "r") as f:
            cache_search_info_data = json.load(f)
        cache_search_info = {}
        for item in cache_search_info_data:
            query = item["query"]
            retrieved_indices = item["retrieved_indices"]
            retrieved_scores = item["retrieved_scores"]
            cache_search_info[query] = (retrieved_scores, retrieved_indices)
    
    ttt_model_cls = TTTModelAnalyzer if ct3_config.analysis else TTTModel
    model = ttt_model_cls(
        model, 
        tokenizer,
        processor,
        embedder=embedder, 
        knowledge_base_samples=knowledge_base_samples,
        retriever=retriever,
        num_ttt_samples=ct3_config.num_ttt_samples, 
        num_epochs=ct3_config.num_epochs, 
        learning_rate=ct3_config.learning_rate, 
        batch_size=ct3_config.batch_size,
        shuffle_ttt_samples=ct3_config.shuffle_ttt_samples,
        data_reduction=ct3_config.data_reduction,
        weighted_sample_training=ct3_config.weighted_sample_training,
        local_adapter_managing=ct3_config.local_adapter_managing,
        local_adapter_manager_size=ct3_config.local_adapter_manager_size,
        local_adapter_manager_threshold=ct3_config.local_adapter_manager_threshold,
        cache_search_info=cache_search_info,
    )
    return model
