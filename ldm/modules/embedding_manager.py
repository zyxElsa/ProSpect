import torch
from torch import nn
import itertools
from ldm.data.personalized import per_img_token_list
from functools import partial
import numpy as np
from ldm.modules.attention import CrossAttention,FeedForward
import PIL
from PIL import Image
import time
DEFAULT_PLACEHOLDER_TOKEN = ["*"]
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    # return tokens
    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token)


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_per_img=None,
            per_image_tokens=False,
            num_vectors_per_token=10,
            progressive_words=False,
            initializer_words=None,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()
        self.placeholder_embedding = None
        self.embedder=embedder

        self.init = True

        self.cond_stage_model = embedder

        self.progressive_words = progressive_words

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings.token_embedding)
            # get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280
        
        self.get_token_for_string = get_token_for_string
        self.get_embedding_for_tkn = get_embedding_for_tkn
        self.token_dim = token_dim
        self.attention = TransformerBlock(dim=token_dim, n_heads=8, d_head=64, dropout = 0.1,dim_out=self.max_vectors_per_token*token_dim) 
        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(1, 1), requires_grad=True)
                self.initial_embeddings = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(1, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(1, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
            self,
            tokenized_text,
            embedded_text,    
            initializer_words=None,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        print('batch',b)

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            if self.initial_embeddings is None:
                print('Working with NO IMGAE mode')
                placeholder_embedding = self.get_embedding_for_tkn('').unsqueeze(0).repeat(self.max_vectors_per_token, 1).to(device)
            else:
                print('Working with IMGAE GUIDING mode')
                placeholder_embedding = self.attention(self.initial_embeddings.view(b,1,768).to(device), self.initial_embeddings.view(b,1,768).to(device))[-1].view(self.max_vectors_per_token,768)  
            
            self.placeholder_embedding = placeholder_embedding
            self.placeholder_embeddings=[]
            self.embedded_texts=[]
            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            if self.string_to_param_dict is not None:
                self.string_to_param_dict[placeholder_string] = torch.nn.Parameter(placeholder_embedding, requires_grad=False).to(device)
        
            # plus
            for i in range(self.max_vectors_per_token):
                self.placeholder_embeddings.append(placeholder_embedding[i].view(1,768))
                new_embedded_text = embedded_text.clone().to(device)
                new_embedded_text[placeholder_idx] = placeholder_embedding[i].view(1,768).float()
                self.embedded_texts.append(new_embedded_text)

            if initializer_words is not None:
                if isinstance(initializer_words, list) and len(initializer_words)==self.max_vectors_per_token:
                    print('Find word list:',initializer_words)
                    for i in range(len(initializer_words)):   
                        if isinstance(initializer_words[i],str):
                            words = initializer_words[i].split(' ')
                            if len(words)==1:
                                if words[0] is not '*':
                                    none_word_token = self.get_token_for_string(words[0]).to(device)
                                    with torch.no_grad():
                                        new_word_embedding = self.get_embedding_for_tkn(none_word_token).to(device)
                                    new_embeddings = new_word_embedding.unsqueeze(0).view(-1,768).to(device)
                                    new_embedded_text = embedded_text.clone().to(device)   
                                    new_embedded_text[placeholder_idx] = new_embeddings.float()    
                                    self.embedded_texts[i]=new_embedded_text
                            else:
                                for j in range(len(words)):     
                                    if words[j] is not '*': 
                                        none_word_token = self.get_token_for_string(words[j]).to(device)
                                        with torch.no_grad():
                                            new_word_embedding = self.get_embedding_for_tkn(none_word_token).to(device)
                                    else:
                                        new_word_embedding = placeholder_embedding[i]
                                    if j==0:
                                        new_embeddings = new_word_embedding.unsqueeze(0).view(-1,768).to(device)  
                                    else:
                                        new_embeddings = torch.cat((new_embeddings,new_word_embedding.unsqueeze(0).view(-1,768).to(device)),dim=0) 
                                new_embedded_text = embedded_text.clone().to(device)
                                num_vectors_for_token = new_embeddings.shape[0]
                                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                                sorted_rows = placeholder_rows[sort_idx]

                                for idx in range(len(sorted_rows)):
                                    row = sorted_rows[idx]
                                    col = sorted_cols[idx]
                                    new_embed_row = torch.cat([new_embedded_text[row][:col], new_embeddings[:num_vectors_for_token], new_embedded_text[row][col + 1:]], axis=0)[:n]
                                    new_embedded_text[row]  = new_embed_row
                                self.embedded_texts[i]=new_embedded_text
                        elif isinstance(initializer_words[i],int):
                                new_embedded_text = self.embedded_texts[initializer_words[i]].clone().to(device) 
                                self.embedded_texts[i]=new_embedded_text

        return self.embedded_texts

    def save(self, ckpt_path):
        torch.save({
                    "attention": self.attention,
                    "initial_embeddings": self.initial_embeddings,
                    }, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if 'attention' in ckpt.keys():
            self.attention = ckpt["attention"]
        else:
            self.attention = None

        if 'initial_embeddings' in ckpt.keys():
            self.initial_embeddings = ckpt["initial_embeddings"]
        else:
            self.initial_embeddings = None

    def embedding_parameters(self):
        return self.attention.parameters()

    def embedding_to_coarse_loss(self):        
        pass

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,dim_out=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff,dim_out=dim_out)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x))
        return x
