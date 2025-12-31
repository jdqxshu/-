"""
unified_extractor.py
多模态特征提取脚本
功能：提取 Text (LLM), Audio (Wav2Vec/HuBERT/WavLM) 特征，并对齐到 fMRI TR 时间点。
包含双重 Pooling 策略验证实验 (Mean vs Max)。
"""

import os
import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor

# ================= 全局配置 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CSV_FILE = '21styear_align.csv'     
AUDIO_FILE = '21styear_audio.wav'   
TR_LEN = 1.5                        
N_TRS = 2249                        

# ================= 工具函数 =================

def align_text_to_tr(word_features, df, n_trs, pooling='mean'):
    """
    将 Word Level 的离散特征对齐到 TR Level 的固定时间网格。
    
    Args:
        word_features: (N_Words, Dim)
        df: 包含 start_ts 的对齐表
        n_trs: 目标 TR 总数
        pooling: 聚合策略 'mean' (平均), 'max' (最大值), 'last' (取最后一个词)
    """
    
    feat_dim = word_features.shape[1]
    tr_features = np.zeros((n_trs, feat_dim))
    
    # 填充缺失的时间戳并计算 TR 索引
    df.start_ts = df.start_ts.bfill()
    word_tr_indices = (df.start_ts.values / TR_LEN).astype(int)
    
    last_valid = np.zeros(feat_dim)
    
    for tr_idx in range(n_trs):
        mask = (word_tr_indices == tr_idx)
        if np.any(mask):
            feats_in_tr = word_features[mask]
            
            # TR 内部的聚合策略
            if pooling == 'mean':
                tr_feat = feats_in_tr.mean(0)
            elif pooling == 'max':
                tr_feat = feats_in_tr.max(0)
            elif pooling == 'last':
                tr_feat = feats_in_tr[-1] 
            else:
                tr_feat = feats_in_tr.mean(0)
                
            tr_features[tr_idx] = tr_feat
            last_valid = tr_feat
        else:
            # 如果当前 TR 无单词，保持上一时刻状态 (Forward Fill)
            tr_features[tr_idx] = last_valid 
            
    return tr_features

# ================= 核心提取逻辑 =================

def extract_llm(model_name, alias, layers, windows, model_pooling='last', tr_pooling='mean'):
    """
    提取 Causal LM 特征。包含两个阶段的 Pooling 操作。
    
    Args:
        model_pooling: 模型输出层的聚合 (建议 'last' 以捕获因果语义)
        tr_pooling:    对齐到 fMRI TR 时的聚合 ('mean' 或 'max')
    """
    print(f"\n[LLM] {alias} | Win={windows} | ModelPool={model_pooling} | TRPool={tr_pooling}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='right')
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).eval().to(DEVICE)
    except Exception as e:
        print(f"Load Error: {e}")
        return
    
    df = pd.read_csv(CSV_FILE, header=None, names=["cased", "uncased", "start_ts", "end_ts"])
    df.cased = df.cased.fillna("")
    
    for win in windows:
        print(f"   >>> Window: {win}")
        
        # 1. 预处理：构建滑动窗口上下文
        full_tokens = [] 
        word_boundaries = [0]
        for txt in df.cased.values:
            t = tokenizer.encode(str(txt), add_special_tokens=False)
            full_tokens.extend(t)
            word_boundaries.append(len(full_tokens))
            
        token_ctx_batch = []
        for i in range(len(df)):
            end_pos = word_boundaries[i+1]
            start_pos = max(0, end_pos - win)
            token_ctx_batch.append(full_tokens[start_pos:end_pos])

        # 2. 批量推理
        batch_size = 16
        word_feats_dict = {l: [] for l in layers}
        
        for i in tqdm(range(0, len(token_ctx_batch), batch_size), desc="Infer"):
            batch = token_ctx_batch[i:i+batch_size]
            max_len = max(len(x) for x in batch)
            padded = [x + [tokenizer.pad_token_id]*(max_len-len(x)) for x in batch]
            inputs = torch.tensor(padded).to(DEVICE)
            attn_mask = (inputs != tokenizer.pad_token_id).long().to(DEVICE)
            
            with torch.no_grad():
                outputs = model(inputs, attention_mask=attn_mask, output_hidden_states=True)
            
            # 定位最后一个有效 Token
            last_idx = attn_mask.sum(1) - 1
            
            for layer in layers:
                h = outputs.hidden_states[layer]
                
                # --- 阶段一：模型输出聚合 (Model Pooling) ---
                if model_pooling == 'last':
                    pooled = h[torch.arange(h.shape[0]), last_idx]
                elif model_pooling == 'mean':
                    mask_expanded = attn_mask.unsqueeze(-1).expand(h.size()).float()
                    sum_embeddings = torch.sum(h * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                
                word_feats_dict[layer].append(pooled.cpu().float().numpy())
        
        # --- 阶段二：时序对齐聚合 (TR Pooling) ---
        for layer in layers:
            raw_feats = np.vstack(word_feats_dict[layer])
            
            tr_feats = align_text_to_tr(raw_feats, df, N_TRS, pooling=tr_pooling)
            
            # 文件名记录双重 Pooling 策略
            fname = f"{alias}_win{win}_layer{layer}_M{model_pooling}_TR{tr_pooling}.npy"
            np.save(fname, tr_feats)
            print(f"      Saved: {fname}")

def extract_audio(model_type, layers, windows, pooling='mean'):
    """
    提取 Audio 特征 (Wav2Vec/HuBERT/WavLM)。
    通过滑动窗口切分音频，并在窗口内进行 Pooling。
    """
    print(f"\n[Audio] {model_type} | Win={windows} | Pooling={pooling}")
    
    if model_type == 'wav2vec':
        name = "facebook/wav2vec2-base-960h"
    elif model_type == 'wavlm':
        name = "microsoft/wavlm-base"
    elif model_type == 'hubert':
        name = "facebook/hubert-base-ls960"
    else:
        raise ValueError(f"Unknown type: {model_type}")
    
    print(f"   -> Loading Model: {name} ...")
    from transformers import Wav2Vec2FeatureExtractor
    
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval().to(DEVICE)
    except Exception as e:
        print(f"Cache load failed, force downloading... Error: {e}")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(name, force_download=True)
        model = AutoModel.from_pretrained(name, force_download=True).eval().to(DEVICE)

    print(f"   -> Loading Audio...")
    wav, sr = librosa.load(AUDIO_FILE, sr=16000) 
    wav_tensor = torch.from_numpy(wav) 
    
    tr_samples = int(16000 * TR_LEN)
    
    for win in windows:
        print(f"   >>> Window: {win} TRs ({win*1.5}s)")
        
        win_samples = int(tr_samples * win)
        # unfold 实现滑动窗口，Stride 固定为 TR_LEN
        
        chunks = wav_tensor.unfold(0, win_samples, tr_samples)
        
        if chunks.shape[0] < N_TRS:
            pad = chunks[-1].unsqueeze(0).repeat(N_TRS - chunks.shape[0], 1)
            chunks = torch.cat([chunks, pad], dim=0)
        
        chunks = chunks[:N_TRS]
        
        layer_feats = {l: [] for l in layers}
        batch_size = 16 
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Extracting"):
            batch_wav = chunks[i:i+batch_size].numpy()
            
            inputs = extractor(batch_wav, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            for layer in layers:
                h = outputs.hidden_states[layer]
                
                # 窗口内的 Pooling 策略
                if pooling == 'mean':
                    pooled = h.mean(dim=1).cpu().numpy()
                elif pooling == 'max':
                    pooled = h.max(dim=1).values.cpu().numpy()
                elif pooling == 'last':
                    pooled = h[:, -1, :].cpu().numpy()
                
                layer_feats[layer].append(pooled)
                
        # 保存结果
        for layer in layers:
            final = np.vstack(layer_feats[layer])
            fname = f"{model_type}_win{win}_layer{layer}_{pooling}.npy"
            np.save(fname, final)
            print(f"      Saved: {fname} {final.shape}")

# ================= 主程序 =================
if __name__ == "__main__":
    print("主入口：执行全模型 Pooling 策略对比实验 (Mean vs Max)。")
    

    # ================= 1. Text Model Group =================
    # 策略：固定 Model Pooling 为 'last'，对比 TR Pooling 的 'mean' vs 'max'
    
    # (A) DeepSeek
    print("\n--- 1.1 DeepSeek (Text) ---")
    # extract_llm("deepseek-ai/deepseek-coder-1.3b-instruct", "deepseek", 
    #             layers=[10], windows=[200], model_pooling='last', tr_pooling='mean')
    extract_llm("deepseek-ai/deepseek-coder-1.3b-instruct", "deepseek", 
                layers=[10], windows=[200], model_pooling='last', tr_pooling='max')

    # (B) GPT-2
    print("\n--- 1.2 GPT-2 (Text) ---")
    extract_llm("gpt2", "gpt2", 
                layers=[10], windows=[200], model_pooling='last', tr_pooling='mean')
    extract_llm("gpt2", "gpt2", 
                layers=[10], windows=[200], model_pooling='last', tr_pooling='max')

    # ================= 2. Audio Model Group =================
    # 策略：对比特征提取窗口内的 'mean' vs 'max'
    # 窗口固定为 Win=2 (3秒)
    
    # (A) HuBERT
    print("\n--- 2.1 HuBERT (Audio) ---")
    extract_audio('hubert', layers=[9], windows=[2], pooling='mean')
    extract_audio('hubert', layers=[9], windows=[2], pooling='max')

    # (B) Wav2Vec 2.0
    print("\n--- 2.2 Wav2Vec 2.0 (Audio) ---")
    extract_audio('wav2vec', layers=[7], windows=[2], pooling='mean')
    extract_audio('wav2vec', layers=[7], windows=[2], pooling='max')
    
    print("\n任务完成。请运行 analysis 脚本查看性能对比。")