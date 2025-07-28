import os
import json
import requests
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model

import faiss
from activeft.sift import Retriever

from ct3_config import CT3Config
from models.local_adapter_manager import LocalAdapterManager
from models.client_request_doc import ClientRequestDoc
from models.embedder import Embedder, load_embedder
from kb.utils import load_knowledge_base


def tokenize(tokenizer: PreTrainedTokenizer, prompt: str, add_eos_token: bool = True) -> Dict[str, Any]:
    """
    Tokenize the given prompt.
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding=True,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 1024
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


class TTTModel(torch.nn.Module):
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        disable_ttt: bool = False,
        embedder: Embedder = None, 
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
            self.local_adapter_manager = LocalAdapterManager(local_adapter_manager_size, local_adapter_manager_threshold, local_adapter_manager_strategy)
        
        self.cache_search_info = cache_search_info

        self.trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.trainable_param_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        self.original_state_dict = {name: param.clone() for name, param in self.model.named_parameters() if name in self.trainable_param_names}

    def forward(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs, generate=True)

    def _forward_or_generate(self, *args, generate=False, **kwargs):

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
            query_embedding = self.embedder([query]).cpu()

            # Check if we can reuse weights from the cache
            best_cache_entry = None
            if self.local_adapter_managing:
                best_cache_entry = self.local_adapter_manager.retrieve_best_cache_entry(query_embedding)
            if best_cache_entry is not None:
                # print("Reusing weights from the cache.")
                adapter_params = best_cache_entry[2]
                self.model.load_state_dict(adapter_params, strict=False)
            else:
                # Ensure the original state before TTT
                self.model.load_state_dict(self.original_state_dict, strict=False)

                retrieved_samples, _ = self.retrieval(query_embedding, query=query)
                if self.data_reduction:
                    scores = list(retrieved_samples.values())
                    mean_score = np.mean(scores)
                    median_score = np.median(scores)
                    adaptive_threshold = min(mean_score, median_score)
                    retrieved_samples = {prompt: score for prompt, score in retrieved_samples.items() if score > adaptive_threshold}
    
                num_fine_tune_samples = len(retrieved_samples)
                # Fine-tune the model
                if num_fine_tune_samples > 0:
                    fine_tune_samples = [tokenize(self.tokenizer, prompt) for prompt in list(retrieved_samples.keys())]
                    
                    sample_weights = None
                    if self.weighted_sample_training:
                        sample_weights = torch.tensor(list(retrieved_samples.values()), dtype=torch.float32)
                        sample_weights = sample_weights / sample_weights.sum()  # Normalize sample weights
                    
                    self.fine_tune(fine_tune_samples, sample_weights=sample_weights)
                
                    if self.local_adapter_managing:
                        # Save the fine-tuned parameters to the cache
                        adapter_params = {name: param.clone() for name, param in self.model.named_parameters() if name in self.trainable_param_names}
                        self.local_adapter_manager.add_to_cache(query, query_embedding, adapter_params)
            
        # Perform the forward pass or generate
        if generate:
            output = self.model.generate(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        
        return output

    def retrieval(self, query_embedding: torch.Tensor, query: Optional[str] = None) -> Tuple[Dict[str, float], List[int]]:
        if self.cache_search_info is not None and query is not None:
            values, indices = self.cache_search_info[query]
        else:
            values, indices, _, _ = self.retriever.search(query_embedding, N=self.num_ttt_samples, K=1000)

        scores = [abs(float(val)) for val in values]
        indices = [int(idx) for idx in indices]
        retrieved_samples = {self.knowledge_base_samples[idx]: scores[i] for i, idx in enumerate(indices)}

        return retrieved_samples, indices

    def fine_tune(self, dataset: List[Dict[str, Any]], sample_weights: Optional[torch.Tensor] = None) -> None:        
        self.model.train()
        optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)
        num_fine_tune_samples = len(dataset)
        with torch.enable_grad():  # Ensure gradients are enabled
            for epoch in range(self.num_epochs):
                for i, batch in enumerate(torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=self.shuffle_ttt_samples)):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss

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
        assert not self.disable_ttt
        self.analysis_data = []
        self.log_file = log_file

    def _forward_or_generate(self, *args, generate=False, **kwargs):
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
        query_embedding = self.embedder([query]).cpu()
        embedding_end_time = time.time()

        query_data = {
            "query": query,
            "used_cache": False,
            "cache_query": None,
            "retrieved_indices": None,
            "retrieved_scores": None,
            "losses": [],
            "timing": {
                "embedding": embedding_end_time - embedding_start_time
            }
        }

        # Check if we can reuse weights from the cache
        best_cache_entry = None
        if self.local_adapter_managing:
            cache_start_time = time.time()
            best_cache_entry = self.local_adapter_manager.retrieve_best_cache_entry(query_embedding)
            cache_end_time = time.time()
            query_data["timing"]["cache_retrieval"] = cache_end_time - cache_start_time

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
            retrieved_samples, indices = self.retrieval(query_embedding, query=query)
            retrieval_end_time = time.time()
            query_data["timing"]["kb_retrieval"] = retrieval_end_time - retrieval_start_time

            if self.data_reduction:
                scores = list(retrieved_samples.values())
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                adaptive_threshold = min(mean_score, median_score)
                retrieved_samples = {prompt: score for prompt, score in retrieved_samples.items() if score > adaptive_threshold}
    
            query_data["retrieved_indices"] = indices
            query_data["retrieved_scores"] = list(retrieved_samples.values())
            
            num_fine_tune_samples = len(retrieved_samples)
            # Fine-tune the model
            if num_fine_tune_samples > 0:
                fine_tune_samples = [tokenize(self.tokenizer, prompt) for prompt in list(retrieved_samples.keys())]
                
                sample_weights = None
                if self.weighted_sample_training:
                    sample_weights = torch.tensor(list(retrieved_samples.values()), dtype=torch.float32)
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

    def fine_tune(self, dataset: List[Dict[str, Any]], sample_weights: Optional[torch.Tensor] = None, query_data: Optional[Dict[str, Any]] = None) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)
        num_fine_tune_samples = len(dataset)
        with torch.enable_grad():  # Ensure gradients are enabled
            for epoch in range(self.num_epochs):
                for i, batch in enumerate(torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=self.shuffle_ttt_samples)):
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


class TTTRemoteModel(TTTModel):
    def __init__(self, *args, ct3_server_url: str, **kwargs) -> None:
        super(TTTRemoteModel, self).__init__(*args, **kwargs)
        self.ct3_server_url = ct3_server_url

    def retrieval(self, query_embedding: torch.Tensor, query: Optional[str] = None) -> Tuple[Dict[str, float], Optional[List[int]]]:
        # Send the request to the server
        embeddingdoc = ClientRequestDoc(query_embedding=query_embedding, num_retrieved_samples=self.num_ttt_samples)
        response = requests.post(self.ct3_server_url, data=embeddingdoc.json()).json()
        retrieved_samples = response["retrieved_samples"] if "retrieved_samples" in response else {}

        return retrieved_samples, None


def convert_to_ttt_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, ct3_config: CT3Config, ct3_server_url: str = None) -> TTTModel:

    if not ct3_config.ttt:
        model = TTTModel(
            model, 
            tokenizer,
            disable_ttt=True
        )
        return model

    model.config.use_cache = False
    for name, param in model.named_parameters():
        param.requires_grad = False

    if ct3_config.use_ipex_llm:
        try:
            from ipex_llm.transformers.qlora import get_peft_model as ipex_get_peft_model, prepare_model_for_kbit_training, LoraConfig as IpexLoraConfig
        except ImportError:
            raise ImportError("ipex-llm is not installed. Please install ipex-llm to use this feature.")
        
        model = prepare_model_for_kbit_training(model)
        lora_config = IpexLoraConfig(
            r=ct3_config.lora_rank,
            lora_alpha=ct3_config.lora_alpha,
            lora_dropout=0.1,
            target_modules=ct3_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = ipex_get_peft_model(model, lora_config)
    else:
        lora_config = LoraConfig(
            r=ct3_config.lora_rank,
            lora_alpha=ct3_config.lora_alpha,
            lora_dropout=0.1,
            target_modules=ct3_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    embedder = load_embedder()

    knowledge_base_samples = None
    retriever = None
    if ct3_server_url is None:
        # Initialize FAISS index for similarity search

        if ct3_config.faiss_index == "IndexFlatIP":
            # IndexFlatIP
            index = faiss.IndexFlatIP(embedder.dimension)
            # Load and concatenate knowledge_base
            knowledge_base_samples, embs = load_knowledge_base(ct3_config.knowledge_base_path)
            index.add(embs)
            retriever = Retriever(index, only_faiss=ct3_config.only_faiss)
        elif ct3_config.faiss_index == "IndexIVFPQ":
            # IndexIVFPQ
            nlist = 100
            m = 8   # number of subquantizers
            quantizer = faiss.IndexFlatL2(embedder.dimension)
            # index = faiss.IndexIVFFlat(quantizer, embedder.dimension, nlist, faiss.METRIC_L2)
            index = faiss.IndexIVFPQ(quantizer, embedder.dimension, nlist, m, 8)
            # Load and concatenate knowledge_base
            knowledge_base_samples, embs = load_knowledge_base(ct3_config.knowledge_base_path)
            index.train(embs)
            index.add(embs)
            assert index.is_trained
            index.nprobe = 10
            retriever = Retriever(index, only_faiss=ct3_config.only_faiss, also_query_opposite=False)
        else:
            raise ValueError(f"Unsupported Faiss Index: {ct3_config.faiss_index}")
    
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
    
    if ct3_server_url is not None:
        model = TTTRemoteModel(
            model, 
            tokenizer,
            embedder=embedder, 
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
            ct3_server_url=ct3_server_url
        )
    else:
        ttt_model_cls = TTTModelAnalyzer if ct3_config.analysis else TTTModel
        model = ttt_model_cls(
            model, 
            tokenizer,
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
