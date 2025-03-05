# Generalized pooling formula is taken from the following research paper :
# "Enhancing Sentence Embedding with Generalized Pooling" Qian Chen, Zhen-Hua Ling, Xiaodan Zhu. COLING (2018)
# https://aclanthology.org/C18-1154.pdf

# Module created and code adapted following the Sentence Transformer documentation :
# https://sbert.net/docs/sentence_transformer/usage/custom_models.html


import math
import torch
from torch import nn
import torch.nn.functional as F
import os
import json

class MultiHeadGeneralizedPooling(nn.Module):
    # Pooling type
    ADDITIVE_POOLING = 0
    MAX_POOLING = 1
    MEAN_POOLING = 2

    # Wieght initialization for additive pooling
    MEAN = 0
    NOISED = 1
    RANDOM = 2


    def __init__(self, pooling_type:int, token_dim: int = 768, sentence_dim: int = 768, num_heads: int = 8, initialize: int=2) -> None:
        """
        MultiHeadGeneralizedPooling class implements a multi-head pooling mechanism for performing sentence embedding from token embeddings.
        
        Attributes:
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each head.
            sentence_dim (int): The dimension of the sentence embeddings.
            token_dim (int): The dimension of the token embeddings.
            initialize (int): Sets the initialization method for the weights: MEAN (0), NOISED (1), or RANDOM (2).
            pooling_type (int): The type of pooling to be used: ADDITIVE or DOT_PRODUCT.
        """

        super(MultiHeadGeneralizedPooling, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = int(sentence_dim / self.num_heads)
        self.sentence_dim = sentence_dim
        self.token_dim = token_dim
        self.hidden_dim = 4 * self.head_dim
        self.initialize = initialize
        self.pooling_type = pooling_type

        assert sentence_dim == token_dim
        
        # Initialize pooling
        if pooling_type == self.ADDITIVE_POOLING :
            self.initialize_additive_pooling()        
        elif pooling_type != self.MEAN_POOLING and pooling_type != self.MAX_POOLING :
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
    
    
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict [str, torch.Tensor]:
        """
        Forward pass for the multi-head generalized pooling layer.

        Args:
            features (dict[str, torch.Tensor]): A dictionary containing feature tensors.
            **kwargs: Additional keyword arguments.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the pooled feature tensors.
            The pooling operation is determined by the `pooling_type` attribute. If `pooling_type`
            is `ADDITIVE`, the `forward_additive` method is used. Otherwise, the `forward_dot_product`
            method is used.
        """
        
        if self.pooling_type == self.ADDITIVE_POOLING :
            return self.forward_additive(features, **kwargs)
        elif self.pooling_type == self.MEAN_POOLING :
            H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
            pooled_output = torch.mean(H, dim=1)  # Mean pooling over the sequence length
            assert pooled_output.shape[1] == self.sentence_dim
            features["sentence_embedding"] = pooled_output
            return features  # Return the final multi-head pooled sentence embedding

        elif self.pooling_type == self.MAX_POOLING : 
            H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
            pooled_output, _ = torch.max(H, dim=1)  # Max pooling over the sequence length
            assert pooled_output.shape[1] == self.sentence_dim
            features["sentence_embedding"] = pooled_output
            return features  # Return the final multi-head pooled sentence embedding
        else :
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
         

    def initialize_additive_pooling(self) -> None:
        """
        Initialize weights to simulate mean pooling by making the attention distribution uniform for each head.
        """
        # Define learnable weights and biases for each head
        self.P = nn.ModuleList([nn.Linear(self.token_dim, self.head_dim) for _ in range(self.num_heads)]) # Projection matrices to apply
        self.W1 = nn.ModuleList([nn.Linear(self.head_dim, self.hidden_dim) for _ in range(self.num_heads)])  # W1^i for each head
        self.W2 = nn.ModuleList([nn.Linear(self.hidden_dim, self.head_dim) for _ in range(self.num_heads)])  # W2^i for each head

        if self.initialize == self.MEAN or self.initialize == self.NOISED :
            # Initialize all heads with weights that simulate mean pooling
            for i in range(self.num_heads):
                nn.init.constant_(self.W1[i].weight, 0)  # Set W1 weights to 0
                nn.init.constant_(self.W1[i].bias, 0)    # Set W1 bias to 0
                nn.init.constant_(self.W2[i].weight, 0)  # Set W2 weights to 0
                nn.init.constant_(self.W2[i].bias, 1)    # Set W2 bias to 1, ensuring equal output for each token
                
                nn.init.constant_(self.P[i].weight, 0)   # Initialize weight to identity matrix
                nn.init.eye_(self.P[i].weight[:, self.head_dim * i : self.head_dim * (i + 1)]) # Initialize the projections to successively be a slice of the original embedding matrix
                nn.init.constant_(self.P[i].bias, 0)     # Set bias to 0

                if self.initialize == self.NOISED :       
                    # Add small random perturbations
                    with torch.no_grad():
                        self.W1[i].weight.add_(torch.randn_like(self.W1[i].weight) * 0.01)
                        self.W1[i].bias.add_(torch.randn_like(self.W1[i].bias) * 0.01)
                        self.W2[i].weight.add_(torch.randn_like(self.W2[i].weight) * 0.01)
                        self.W2[i].bias.add_(torch.randn_like(self.W2[i].bias) * 0.01)
                        self.P[i].weight.add_(torch.randn_like(self.P[i].weight) * 0.01)
                        self.P[i].bias.add_(torch.randn_like(self.P[i].bias) * 0.01)

        elif self.initialize == self.RANDOM :
            # Initialize weights randomly
            for i in range(self.num_heads):
                nn.init.kaiming_uniform_(self.W1[i].weight, a=0)
                nn.init.zeros_(self.W1[i].bias)
                nn.init.kaiming_uniform_(self.W2[i].weight, a=0)
                nn.init.zeros_(self.W2[i].bias)
                nn.init.kaiming_uniform_(self.P[i].weight, a=0)
                nn.init.zeros_(self.P[i].bias)
        else :
            raise ValueError(f"Unsupported initialization type: {self.initialize}")

    def forward_additive(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:
        """
        Perform multi-head generalized pooling on the token embeddings using the formula given in the research paper.
        
        Args:
            features (dict[str, torch.Tensor]): A dictionary containing:
            - "token_embeddings" (torch.Tensor): Token-level embeddings (batch_size, seq_len, token_dim).
            - "attention_mask" (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).
            
            dict[str, torch.Tensor]: A dictionary with the pooled sentence embeddings under the key "sentence_embedding" (batch_size, num_heads * embedding_dim).
        Returns:
            torch.Tensor: The pooled sentence embeddings (batch_size, num_heads * embedding_dim).
        """

        attention_mask = features["attention_mask"].unsqueeze(-1)  # (batch_size, 1, seq_len)

        head_outputs = []  # To store output from each head
        
        for i in range(self.num_heads):

            H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
            H_i = self.P[i](H) # Projecting H in a lower dimension
            A_i = self.W1[i](H_i)  # (batch_size, seq_len, hidden_dim) for head i
            A_i = F.relu(A_i)  # Apply ReLU activation

            # Second linear transformation: W2^i * ReLU(W1^i * H^T + b1^i)
            A_i = self.W2[i](A_i)  # (batch_size, seq_len, token_dim) for head i

            # Apply softmax to get attention weights for head i
            attention_mask_expanded = attention_mask.expand(-1, -1, self.head_dim)
            A_i = F.softmax(A_i + attention_mask_expanded.log(), dim=1)  # Softmax along seq_len
            

            # Apply attention weights to get the weighted sum of token embeddings for head i
            v_i = torch.sum(H_i * A_i, dim=1)  # Weighted sum over seq_len (batch_size, token_dim)
            
            head_outputs.append(v_i)  # Store the output of this head
        
        # Concatenate outputs from all heads along the embedding dimension
        pooled_output = torch.cat(head_outputs, dim=-1)  # (batch_size, num_heads * hidden_dim = self.token_dim)
        assert pooled_output.shape[1] == self.sentence_dim

        features["sentence_embedding"] = pooled_output
        return features  # Return the final multi-head pooled sentence embedding

    def get_config_dict(self) -> dict[str, float]:
        """
        Returns a dictionary containing the configuration parameters of the pooling layer.
        Returns:
            dict[str, float]: A dictionary with the following keys:
                - "sentence_dim": Dimension of the sentence embeddings.
                - "token_dim": Dimension of the token embeddings.
                - "num_heads": Number of attention heads.
                - "initialize": Initialization parameter for the pooling layer.
                - "pooling_type": Type of pooling used.
        """
        return {"sentence_dim": self.sentence_dim, "token_dim": self.token_dim, "num_heads": self.num_heads, "initialize": self.initialize, "pooling_type" : self.pooling_type}

    def get_sentence_embedding_dimension(self) -> int:
        """
        Returns the dimension of the sentence embeddings.
        This method provides the dimensionality of the sentence embeddings
        used in the model.

        Returns:
            int: The dimension of the sentence embeddings.
        """

        return self.sentence_dim
    
    def save(self, save_dir: str, **kwargs) -> None:
        """
        Saves the configuration and weights of the pooling layer to the specified directory.
        
        Args:
            save_dir (str): The directory where the configuration and weights will be saved.
            **kwargs: Additional keyword arguments.

        The method performs the following steps:
        1. Saves the configuration dictionary to a file named "config.json" in the specified directory.
        2. Depending on the pooling type, saves the weights of the pooling layer to a file named "multihead_pooling_weights.pt" in the specified directory.
            - For ADDITIVE pooling type, saves the weights of P, W1, and W2.
            - For DOT_PRODUCT pooling type, saves the weights of P_K and Q.
        """
        # Save configuration as before
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=4)
        
        pooling_weights = {}

        if self.pooling_type == self.ADDITIVE :
            # Save weights of the pooling layer (P, W1, W2)
            pooling_weights = {
                "P": [p.weight.data for p in self.P],
                "W1": [w.weight.data for w in self.W1],
                "W2": [w.weight.data for w in self.W2]
            }
        
        elif self.pooling_type == self.DOT_PRODUCT :
            # Save weights of the pooling layer (P, W1, W2)
            pooling_weights = {
                "P_K": [p.weight.data for p in self.P_K],
                "Q": [w.weight.data for w in self.Q],
            }
        else :
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        # Save as separate files
        torch.save(pooling_weights, os.path.join(save_dir, "multihead_pooling_weights.pt"))

        
    @staticmethod
    def load(load_dir: str, device: str = 'cpu', **kwargs) -> "MultiHeadGeneralizedPooling":
        """
        Load a MultiHeadGeneralizedPooling model from a specified directory.
        Args:
            load_dir (str): The directory from which to load the model configuration and weights.
            device (str, optional): The device on which to load the model (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            **kwargs: Additional keyword arguments.
        Returns:
            MultiHeadGeneralizedPooling: The loaded model with the specified configuration and weights.
        """

        # Load configuration as before
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        # Load the model with configuration
        model = MultiHeadGeneralizedPooling(**config)

        # Load the weights for the pooling layer
        pooling_weights = torch.load(os.path.join(load_dir, "multihead_pooling_weights.pt"),
                                     map_location=torch.device(device))
        
        if model.pooling_type == model.ADDITIVE :
            # Assign loaded weights to the pooling layers
            for i in range(model.num_heads):
                model.P[i].weight.data = pooling_weights["P"][i]
                model.W1[i].weight.data = pooling_weights["W1"][i]
                model.W2[i].weight.data = pooling_weights["W2"][i]

        elif model.pooling_type == model.DOT_PRODUCT :
            # Assign loaded weights to the pooling layers
            for i in range(model.num_heads):
                model.P_K[i].weight.data = pooling_weights["P_K"][i]
                model.Q[i].weight.data = pooling_weights["Q"][i]
        else :
            raise ValueError(f"Unsupported pooling type: {model.pooling_type}")

        return model

