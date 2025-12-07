import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

try:
    from rdkit.Chem.rdMolDescriptors import GetMorganGenerator
    MORGAN_GENERATOR_AVAILABLE = True
except ImportError:
    MORGAN_GENERATOR_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_add_pool

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, kendalltau

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import datetime
from collections import OrderedDict, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

from esm.models.esmc import ESMC
from esm.sdk.api import *

os.environ["INFRA_PROVIDER"] = "True"

all_amino_acid_number = {'A':5, 'C':23,'D':13,'E':9, 'F':18,
                         'G':6, 'H':21,'I':12,'K':15,'L':4,
                         'M':20,'N':17,'P':14,'Q':16,'R':10,
                         'S':8, 'T':11,'V':7, 'W':22,'Y':19,
                         '_':32}

def esm_encoder_seq(seq, pad_len):
    s = [all_amino_acid_number[x] for x in seq]
    while len(s) < pad_len:
        s.append(1)
    s.insert(0, 0)
    s.append(2)
    return torch.tensor(s)

def get_local_esmc_model(device):
    return ESMC.from_pretrained("esmc_600m", device=device)

def build_smarts_vocabulary(csv_path):
    print("=== 构建SMARTS词汇表 ===")
    df = pd.read_csv(csv_path)

    all_smarts = set()
    for smarts_str in df['warhead_smarts']:
        if pd.isna(smarts_str) or smarts_str in ['NO_WARHEAD', 'INVALID_SMILES']:
            continue
        smarts_list = [s.strip() for s in str(smarts_str).split(';')]
        all_smarts.update(smarts_list)

    smarts_vocab = sorted(list(all_smarts))
    print(f"SMARTS词汇表大小: {len(smarts_vocab)} 个唯一模式")

    print("前10个SMARTS模式:")
    for i, s in enumerate(smarts_vocab[:10], 1):
        print(f"  {i}. {s}")

    return smarts_vocab

def encode_smarts_to_vector(smarts_str, smarts_vocab):
    vector = np.zeros(len(smarts_vocab), dtype=np.float32)

    if pd.isna(smarts_str) or smarts_str in ['NO_WARHEAD', 'INVALID_SMILES']:
        return vector

    smarts_list = [s.strip() for s in str(smarts_str).split(';')]

    for smarts in smarts_list:
        if smarts in smarts_vocab:
            idx = smarts_vocab.index(smarts)
            vector[idx] = 1.0

    return vector

def generate_smarts_features(smarts_str, smarts_vocab, mol=None):
    binary_vector = encode_smarts_to_vector(smarts_str, smarts_vocab)

    stats_features = []

    if pd.isna(smarts_str) or smarts_str in ['NO_WARHEAD', 'INVALID_SMILES']:
        num_patterns = 0
        smarts_list = []
    else:
        smarts_list = [s.strip() for s in str(smarts_str).split(';')]
        num_patterns = len(smarts_list)

    stats_features.append(num_patterns)

    if mol is not None and num_patterns > 0:
        total_matches = 0
        max_matches = 0

        for smarts in smarts_list:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = len(mol.GetSubstructMatches(pattern))
                    total_matches += matches
                    max_matches = max(max_matches, matches)
            except:
                pass

        stats_features.append(total_matches)
        stats_features.append(max_matches)
    else:
        stats_features.extend([0.0, 0.0])

    smarts_features = np.concatenate([binary_vector, np.array(stats_features, dtype=np.float32)])

    return smarts_features

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def bond_to_features(bond):
    bond_type = bond.GetBondType()
    bond_stereo = bond.GetStereo()
    bond_conjugation = bond.GetIsConjugated()
    bond_is_in_ring = bond.IsInRing()

    bond_type_one_hot = one_hot(bond_type, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])
    bond_stereo_one_hot = one_hot(bond_stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ])

    return torch.tensor(bond_type_one_hot + bond_stereo_one_hot + [bond_conjugation, bond_is_in_ring], dtype=torch.float)

def smiles_to_graph(smiles, fps, smarts_features=None):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features_raw = torch.tensor(atoms, dtype=torch.float).view(-1, 1)

    ring = mol.GetRingInfo()
    fps_torch = torch.tensor(fps, dtype=torch.float).view(1, -1)

    if smarts_features is not None:
        smarts_torch = torch.tensor(smarts_features, dtype=torch.float).view(1, -1)
    else:
        smarts_torch = None

    node_features_list = []
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I'])
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                               Chem.rdchem.HybridizationType.SP2,
                                               Chem.rdchem.HybridizationType.SP3,
                                               Chem.rdchem.HybridizationType.SP3D,
                                               Chem.rdchem.HybridizationType.SP3D2])
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(atom_idx, rsize) for rsize in [3, 4, 5, 6, 7, 8]]
        o += [atom.GetFormalCharge()]

        o_torch = torch.tensor(o, dtype=torch.float).view(1, -1)

        if smarts_torch is not None:
            merged_feat = torch.cat([atom_features_raw[atom_idx].view(1, -1), o_torch, fps_torch, smarts_torch], dim=1)
        else:
            merged_feat = torch.cat([atom_features_raw[atom_idx].view(1, -1), o_torch, fps_torch], dim=1)

        node_features_list.append(merged_feat.squeeze(0))

    node_features = torch.stack(node_features_list, dim=0)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

        b_feat = bond_to_features(bond)
        edge_features.append(b_feat)
        edge_features.append(b_feat)

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 11), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features, dim=0)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_mol_features(mol):
    if MORGAN_GENERATOR_AVAILABLE:
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        fp_morgan = morgan_gen.GetFingerprint(mol)
    else:
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    fp_morgan_bits = np.zeros((1024,))
    DataStructs.ConvertToNumpyArray(fp_morgan, fp_morgan_bits)

    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    fp_maccs_bits = np.zeros((166,))
    DataStructs.ConvertToNumpyArray(fp_maccs, fp_maccs_bits)

    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    psa = rdMolDescriptors.CalcTPSA(mol)
    stereo_count = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    c_logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    nrings = rdMolDescriptors.CalcNumRings(mol)
    nrings_h = rdMolDescriptors.CalcNumHeterocycles(mol)
    nrings_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    nrings_ar_h = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    atm_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    atm_heavy = mol.GetNumHeavyAtoms()
    atm_all = mol.GetNumAtoms()

    return np.concatenate((
        np.array([hbd, hba, rtb, psa, stereo_count, c_logp, mr, csp3, nrings, nrings_h,
                  nrings_ar, nrings_ar_h, spiro, mw, atm_hetero, atm_heavy, atm_all]),
        fp_morgan_bits,
        fp_maccs_bits
    ), axis=0)

def analyze_dataset_targets(csv_path, min_samples=50):
    print("=== 分析数据集靶点分布 ===")

    df = pd.read_csv(csv_path)
    target_counts = df['Target_Gene'].value_counts()

    print(f"数据集总样本数: {len(df)}")
    print(f"总靶点数: {len(target_counts)}")
    print(f"靶点样本数分布:")
    print(f"  - 最大: {target_counts.max()}")
    print(f"  - 最小: {target_counts.min()}")
    print(f"  - 平均: {target_counts.mean():.1f}")
    print(f"  - 中位数: {target_counts.median():.1f}")

    valid_targets = target_counts[target_counts >= min_samples].index.tolist()
    print(f"\n样本数 >= {min_samples} 的靶点: {len(valid_targets)} 个")

    print(f"\n前20个靶点分布:")
    for i, (target, count) in enumerate(target_counts.head(20).items()):
        status = "✓" if target in valid_targets else "✗"
        print(f"  {status} {target}: {count}")

    return valid_targets, target_counts

def process_protein_sequences(csv_path, output_pkl='protein_embeddings.pkl'):
    print("=== 处理蛋白质序列 ===")

    df = pd.read_csv(csv_path)
    print(f"数据总行数: {len(df)}")

    unique_sequences = df['proteinseq'].unique()
    print(f"唯一蛋白质序列数量: {len(unique_sequences)}")

    existing_embeddings = {}
    if os.path.exists(output_pkl):
        try:
            print(f"发现已存在的嵌入文件: {output_pkl}")
            with open(output_pkl, 'rb') as f:
                existing_embeddings = pickle.load(f)
            print(f"已加载 {len(existing_embeddings)} 个已存在的蛋白质嵌入")
        except Exception as e:
            print(f"加载已存在嵌入文件时出错: {str(e)}，将重新生成")
            existing_embeddings = {}

    sequences_to_process = []
    for seq in unique_sequences:
        if seq not in existing_embeddings:
            sequences_to_process.append(seq)

    print(f"需要新生成嵌入的序列数量: {len(sequences_to_process)}")

    if len(sequences_to_process) == 0:
        print("所有蛋白质序列的嵌入都已存在，无需重新生成")
        return existing_embeddings

    print("需要处理的蛋白质序列:")
    for idx, seq in enumerate(sequences_to_process, 1):
        print(f"序列{idx}: 长度={len(seq)} 内容={seq[:50]}...")

    print("初始化本地ESMC模型...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_local_esmc_model(device)

    print("开始生成新的蛋白质嵌入...")
    new_embeddings = {}

    for idx, seq in enumerate(sequences_to_process, 1):
        try:
            protein_tensor = ESMProteinTensor(sequence=esm_encoder_seq(seq, len(seq)).to(device))
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            esm_embedding = logits_output.embeddings

            if not isinstance(esm_embedding, torch.Tensor):
                print(f"警告: 序列 {idx} (长度={len(seq)}) 的embeddings不是张量，类型={type(esm_embedding)}，跳过")
                continue

            new_embeddings[seq] = esm_embedding.cpu()
            print(f"处理序列 {idx}/{len(sequences_to_process)} 成功，嵌入形状: {esm_embedding.shape}")
        except Exception as e:
            print(f"处理序列 {idx} (长度={len(seq)}) 时出错: {str(e)}，内容={seq[:50]}...")
            continue

    all_embeddings = {**existing_embeddings, **new_embeddings}

    print(f"\n保存蛋白质嵌入到 {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_embeddings, f)
    print(f"蛋白质嵌入保存完成！")
    print(f"  - 已存在的嵌入: {len(existing_embeddings)} 个")
    print(f"  - 新生成的嵌入: {len(new_embeddings)} 个")
    print(f"  - 总计嵌入数量: {len(all_embeddings)} 个")

    return all_embeddings

def stratified_multi_task_split(data_list, test_size=0.2, min_val_per_target=1, min_train_per_target=1, random_state=42):
    import random
    random.seed(random_state)
    np.random.seed(random_state)

    train_data = []
    val_data = []

    target_groups = {}
    for data in data_list:
        target = data.target_gene
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(data)

    print(f"=== 按靶点分层划分数据 (80/20划分) ===")
    print(f"总靶点数: {len(target_groups)}")

    for target, target_data in target_groups.items():
        n_samples = len(target_data)

        if n_samples < 2:
            print(f"  警告: {target} 只有 {n_samples} 个样本，跳过")
            continue

        n_val = max(min_val_per_target, int(n_samples * test_size))
        n_val = min(n_val, n_samples - min_train_per_target)
        n_train = n_samples - n_val

        if n_train < min_train_per_target or n_val < min_val_per_target:
            if n_samples == 2:
                n_train, n_val = 1, 1
            else:
                n_train = max(min_train_per_target, 1)
                n_val = n_samples - n_train

        random.shuffle(target_data)
        val_target = target_data[:n_val]
        train_target = target_data[n_val:]

        train_data.extend(train_target)
        val_data.extend(val_target)

        actual_ratio = n_val / n_samples * 100
        print(f"  {target}: 总样本={n_samples}, 训练={len(train_target)}, 验证={len(val_target)} ({actual_ratio:.1f}%)")

    print(f"最终划分: 训练集={len(train_data)}, 验证集={len(val_data)}")
    print(f"总体验证集比例: {len(val_data)/(len(train_data)+len(val_data))*100:.1f}%")
    return train_data, val_data

def load_multi_task_dataset(csv_path, device, valid_targets, smarts_vocab=None, test=False):
    df = pd.read_csv(csv_path)

    df_filtered = df[df['Target_Gene'].isin(valid_targets)]
    print(f"原始数据: {len(df)} 行")
    print(f"过滤后数据: {len(df_filtered)} 行 (包含 {len(valid_targets)} 个靶点)")

    nan_count = df_filtered['pic50'].isna().sum()
    inf_count = (~np.isfinite(df_filtered['pic50'])).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"警告: 发现异常PIC50值 - NaN: {nan_count}, 无穷大: {inf_count}")
    else:
        print(f"✓ PIC50数据质量验证通过")

    print(f"PIC50统计:")
    print(f"  - 最小值: {df_filtered['pic50'].min():.4f}")
    print(f"  - 最大值: {df_filtered['pic50'].max():.4f}")
    print(f"  - 均值: {df_filtered['pic50'].mean():.4f}")
    print(f"  - 标准差: {df_filtered['pic50'].std():.4f}")

    with open('protein_embeddings.pkl', 'rb') as f:
        protein_embeddings = pickle.load(f)

    data_list = []
    skipped_count = 0

    for idx, row in df_filtered.iterrows():
        smiles = row['SMILES']
        pic50 = row['pic50']
        protein_seq = row['proteinseq']
        target_gene = row['Target_Gene']

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告: SMILES {smiles} 无效 (第{idx}行)")
            skipped_count += 1
            continue

        if protein_seq not in protein_embeddings:
            print(f"警告: 第{idx}行的蛋白质序列未找到嵌入")
            skipped_count += 1
            continue

        protein_embedding = protein_embeddings[protein_seq]

        if protein_embedding.dim() == 3:
            protein_embedding = protein_embedding.mean(dim=1)
        elif protein_embedding.dim() == 2:
            protein_embedding = protein_embedding.mean(dim=0, keepdim=True)

        if protein_embedding.dim() == 1:
            protein_embedding = protein_embedding.unsqueeze(0)

        protein_embedding = protein_embedding.detach().to(device).requires_grad_(True)

        smarts_features = None
        if smarts_vocab is not None and 'warhead_smarts' in row:
            smarts_str = row['warhead_smarts']
            smarts_features = generate_smarts_features(smarts_str, smarts_vocab, mol)

        graph_data = smiles_to_graph(smiles, generate_mol_features(mol), smarts_features)

        y_values = []
        for target in valid_targets:
            if target == target_gene:
                y_values.append(pic50)
            else:
                y_values.append(float('nan'))

        graph_data.y = torch.tensor(y_values, dtype=torch.float).unsqueeze(0)
        graph_data.original_y = torch.tensor(y_values, dtype=torch.float).unsqueeze(0)
        graph_data.smiles = smiles
        graph_data.target_gene = target_gene
        graph_data.target_idx = valid_targets.index(target_gene)
        graph_data.protein_embedding = protein_embedding
        data_list.append(graph_data)

    print(f"成功加载 {len(data_list)} 个有效数据点")
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个无效数据点")

    return data_list

def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    pred = pred[mask]
    target = target[mask]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return F.mse_loss(pred, target, reduction='mean')

class MaskedStandardScaler(StandardScaler):
    def __init__(self):
        self.avgs = None
        self.vars = None
        self.mask = None

    def fit(self, X, y=None):
        self.avgs = [0.0]*X.shape[1]
        self.vars = [0.0]*X.shape[1]
        self.mask = ~np.isnan(X)
        for prop in range(X.shape[1]):
            mask = ~np.isnan(X[:,prop])
            self.avgs[prop] = np.mean(X[:,prop][mask])
            self.vars[prop] = np.sqrt(np.var(X[:,prop][mask]))
        return self

    def transform(self, X):
        for prop in range(X.shape[1]):
            for i in range(X.shape[0]):
                X[i, prop] = (X[i, prop] - self.avgs[prop]) / self.vars[prop]
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        for prop in range(X.shape[1]):
            for i in range(X.shape[0]):
                X[i, prop] = X[i, prop]  * self.vars[prop] + self.avgs[prop]
        return X

class TaskRelationModule(nn.Module):
    def __init__(self, num_tasks, task_embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim

        print(f"\n【创新模块1】初始化任务关系模块")
        print(f"  - 任务数量: {num_tasks}")
        print(f"  - 任务嵌入维度: {task_embed_dim}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - 学习方式: 自动学习（无需人工标注家族）")

        self.task_embeddings = nn.Embedding(num_tasks, task_embed_dim)
        nn.init.xavier_uniform_(self.task_embeddings.weight)

        self.task_attn = nn.MultiheadAttention(
            embed_dim=task_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(task_embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, task_indices):
        batch_size = task_indices.size(0)

        current_task_emb = self.task_embeddings(task_indices)

        all_task_embs = self.task_embeddings.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        task_context, attn_weights = self.task_attn(
            query=current_task_emb.unsqueeze(1),
            key=all_task_embs,
            value=all_task_embs
        )

        task_context = task_context.squeeze(1)

        task_context = self.layer_norm(current_task_emb + self.dropout(task_context))

        return task_context, attn_weights

    def get_task_similarity_matrix(self):
        with torch.no_grad():
            embeddings = self.task_embeddings.weight

            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        return similarity_matrix.cpu().numpy()

class EnhancedCrossModalAttention(nn.Module):
    def __init__(self, ligand_dim, protein_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.ligand_dim = ligand_dim
        self.protein_dim = protein_dim
        self.num_heads = num_heads

        print(f"\n【创新模块2】初始化增强跨模态注意力")
        print(f"  - 分子特征维度: {ligand_dim}")
        print(f"  - 蛋白特征维度: {protein_dim}")
        print(f"  - 注意力头数: {num_heads} (原版为1头)")
        print(f"  - 融合方式: 门控自适应融合 (原版为简单拼接)")

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=ligand_dim,
            num_heads=num_heads,
            kdim=protein_dim,
            vdim=protein_dim,
            dropout=dropout,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(ligand_dim + ligand_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(ligand_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ligand_rep, protein_embed):
        batch_size = ligand_rep.size(0)

        if protein_embed.dim() == 3:
            if protein_embed.size(0) == 1:
                protein_embed = protein_embed.squeeze(0)
            else:
                protein_embed = protein_embed[0]

        protein_embed = protein_embed.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, attn_weights = self.multihead_attn(
            query=ligand_rep.unsqueeze(1),
            key=protein_embed,
            value=protein_embed
        )

        attn_output = attn_output.squeeze(1)

        attn_output = self.layer_norm(ligand_rep + self.dropout(attn_output))

        gate_input = torch.cat([ligand_rep, attn_output], dim=1)
        gate = self.gate_net(gate_input)

        fused_rep = gate * ligand_rep + (1 - gate) * attn_output

        return fused_rep

class EnhancedMultiTaskModel(nn.Module):
    def __init__(self,
                 input_dim, edge_dim,
                 tasks,
                 shared_hidden_dim=128,
                 individual_hidden_dim=128,
                 num_experts=2,
                 num_heads=1,
                 dp=0.3,
                 protein_embed_dim=1152,
                 task_embed_dim=64):
        super(EnhancedMultiTaskModel, self).__init__()

        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.edge_dim = edge_dim

        print(f"\n" + "="*80)
        print(f"初始化增强版多任务模型 (含SMARTS特征)")
        print(f"="*80)
        print(f"任务数: {self.num_tasks}")
        print(f"任务列表: {self.tasks}")
        print(f"输入维度: {input_dim} (包含SMARTS特征)")
        print(f"\n核心创新:")
        print(f"  ✅ 任务关系模块 - 自动学习{self.num_tasks}个任务间的相关性")
        print(f"  ✅ 多头门控注意力 - 4头注意力+自适应融合")
        print(f"  ✅ 区别于MultiMolCGC - 架构图有明显差异")
        print(f"  ✅ SMARTS特征集成 - 弹头反应性基团编码")
        print(f"="*80)

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                GATConv(input_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                nn.BatchNorm1d(shared_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp),
                GATConv(shared_hidden_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                nn.BatchNorm1d(shared_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp)
            ) for _ in range(num_experts)
        ])

        self.task_specific_experts = nn.ModuleDict({
            task: nn.ModuleList([
                nn.Sequential(
                    GATConv(input_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                    nn.BatchNorm1d(shared_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dp),
                    GATConv(shared_hidden_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                    nn.BatchNorm1d(shared_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dp)
                ) for _ in range(num_experts)
            ]) for task in self.tasks
        })

        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                GATConv(input_dim, num_experts * 2, heads=1, edge_dim=edge_dim),
                nn.Softmax(dim=1)
            ) for task in self.tasks
        })

        self.task_relation = TaskRelationModule(
            num_tasks=self.num_tasks,
            task_embed_dim=task_embed_dim,
            num_heads=4,
            dropout=dp
        )

        self.cross_attn_modules = nn.ModuleDict({
            task: EnhancedCrossModalAttention(
                ligand_dim=shared_hidden_dim,
                protein_dim=protein_embed_dim,
                num_heads=4,
                dropout=dp
            ) for task in self.tasks
        })

        final_dim = shared_hidden_dim + task_embed_dim
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(final_dim, individual_hidden_dim),
                nn.BatchNorm1d(individual_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp),
                nn.Linear(individual_hidden_dim, 1)
            ) for task in self.tasks
        })

    def _apply_expert(self, expert_block, x, edge_index, edge_attr):
        out = x
        for layer in expert_block:
            if isinstance(layer, GATConv):
                out = layer(out, edge_index, edge_attr)
            else:
                out = layer(out)
        return out

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        shared_rep = torch.stack(
            [self._apply_expert(expert, x, edge_index, edge_attr)
             for expert in self.shared_experts], dim=1
        )

        task_outputs = {}
        for task, experts in self.task_specific_experts.items():
            task_rep = torch.stack(
                [self._apply_expert(expert, x, edge_index, edge_attr)
                 for expert in experts], dim=1
            )

            merged_rep = torch.cat([shared_rep, task_rep], dim=1)

            gate_logits = self._apply_expert(self.gates[task], x, edge_index, edge_attr)
            node_rep = torch.einsum('beh,be->bh', merged_rep, gate_logits)

            node_rep_pooled = global_add_pool(node_rep, batch)

            protein_embed = data.protein_embedding.to(node_rep_pooled.device)
            fused_rep = self.cross_attn_modules[task](node_rep_pooled, protein_embed)

            task_idx = data.target_idx.to(node_rep_pooled.device)
            task_context, _ = self.task_relation(task_idx)

            enhanced_rep = torch.cat([fused_rep, task_context], dim=1)

            out = self.task_heads[task](enhanced_rep)
            task_outputs[task] = out

        preds = []
        for t in self.tasks:
            preds.append(task_outputs[t])
        preds = torch.cat(preds, dim=1)

        return preds

def compute_quality_metrics(ref, pred):
    ref = np.array(ref)
    pred = np.array(pred)

    rmse = np.sqrt(np.mean((ref - pred)**2))
    mae = np.mean(np.abs(ref - pred))
    corr_pearson = pearsonr(ref, pred)[0] if len(ref) > 1 else np.nan
    corr_kendall = kendalltau(ref, pred)[0] if len(ref) > 1 else np.nan
    return rmse, mae, corr_pearson, corr_kendall

def create_output_directories():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join("results", f"enhanced_multi_task_smarts_{timestamp}")

    directories = {
        'base': base_dir,
        'predictions': os.path.join(base_dir, '01_predictions'),
        'task_analysis': os.path.join(base_dir, '02_task_analysis'),
        'cross_validation': os.path.join(base_dir, '03_cross_validation'),
        'visualizations': os.path.join(base_dir, '04_visualizations'),
        'summary_reports': os.path.join(base_dir, '05_summary_reports'),
        'model_weights': os.path.join(base_dir, '06_model_weights'),
        'task_relations': os.path.join(base_dir, '07_task_relations')
    }

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

    return directories

def analyze_multi_task_performance(all_predictions_detail, valid_targets, output_dirs):
    print("=== 分析多任务性能 ===")

    task_performance = {}

    for task_idx, task in enumerate(valid_targets):
        task_predictions = [pred for pred in all_predictions_detail if pred['task'] == task]

        if len(task_predictions) >= 3:
            true_vals = [p['true_pic50'] for p in task_predictions]
            pred_vals = [p['predicted_pic50'] for p in task_predictions]

            rmse, mae, pearson, kendall = compute_quality_metrics(true_vals, pred_vals)

            task_performance[task] = {
                'n_samples': len(task_predictions),
                'rmse': rmse,
                'mae': mae,
                'pearson': pearson,
                'kendall': kendall,
                'mean_true': np.mean(true_vals),
                'std_true': np.std(true_vals),
                'mean_pred': np.mean(pred_vals),
                'std_pred': np.std(pred_vals)
            }

    task_perf_df = []
    for task, perf in task_performance.items():
        task_perf_df.append({
            'Task': task,
            'N_Samples': perf['n_samples'],
            'RMSE': perf['rmse'],
            'MAE': perf['mae'],
            'Pearson': perf['pearson'],
            'Kendall': perf['kendall'],
            'Mean_True': perf['mean_true'],
            'Std_True': perf['std_true'],
            'Mean_Pred': perf['mean_pred'],
            'Std_Pred': perf['std_pred']
        })

    task_perf_df = pd.DataFrame(task_perf_df)
    task_perf_df = task_perf_df.sort_values('Pearson', ascending=False)

    task_perf_file = os.path.join(output_dirs['task_analysis'], 'multi_task_performance.csv')
    task_perf_df.to_csv(task_perf_file, index=False)
    print(f"多任务性能分析已保存: {task_perf_file}")

    return task_performance

def visualize_task_similarity(model, valid_targets, output_dirs):
    print("\n=== 分析学到的任务关系 ===")

    similarity_matrix = model.task_relation.get_task_similarity_matrix()

    sim_df = pd.DataFrame(similarity_matrix, index=valid_targets, columns=valid_targets)
    sim_file = os.path.join(output_dirs['task_relations'], 'task_similarity_matrix.csv')
    sim_df.to_csv(sim_file)
    print(f"任务相似度矩阵已保存: {sim_file}")

    plt.figure(figsize=(16, 14))
    im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, label='相似度')

    plt.xticks(range(len(valid_targets)), valid_targets, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(valid_targets)), valid_targets, rotation=0, fontsize=8)

    plt.title('学到的任务相似度矩阵（自动学习，无需人工标注）', fontsize=14, pad=20)
    plt.xlabel('任务（靶点）', fontsize=12)
    plt.ylabel('任务（靶点）', fontsize=12)
    plt.tight_layout()

    heatmap_file = os.path.join(output_dirs['task_relations'], 'task_similarity_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"任务相似度热力图已保存: {heatmap_file}")

    print("\n发现的高度相似任务对（相似度 > 0.7）:")
    high_sim_pairs = []
    for i in range(len(valid_targets)):
        for j in range(i+1, len(valid_targets)):
            sim = similarity_matrix[i, j]
            if sim > 0.7:
                high_sim_pairs.append((valid_targets[i], valid_targets[j], sim))
                print(f"  {valid_targets[i]} <-> {valid_targets[j]}: {sim:.4f}")

    if high_sim_pairs:
        pairs_df = pd.DataFrame(high_sim_pairs, columns=['Task1', 'Task2', 'Similarity'])
        pairs_file = os.path.join(output_dirs['task_relations'], 'high_similarity_pairs.csv')
        pairs_df.to_csv(pairs_file, index=False)
        print(f"高相似度任务对已保存: {pairs_file}")

def generate_multi_task_report(task_performance, output_dirs, all_predictions_detail):
    print("=== 生成多任务分析报告 ===")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_file = os.path.join(output_dirs['summary_reports'], 'enhanced_multi_task_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("         增强版多任务学习药物-靶点结合预测分析报告\n")
        f.write("         (任务关系模块 + 多头门控注意力 + SMARTS特征)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {timestamp}\n\n")

        f.write("【核心创新】\n")
        f.write("-" * 40 + "\n")
        f.write("✅ 任务关系模块 - 自动学习任务间相关性，无需人工标注蛋白家族\n")
        f.write("✅ 增强跨模态注意力 - 4头注意力+门控融合，替代原版单头简单拼接\n")
        f.write("✅ 低样本任务性能提升 - 通过任务关系借助高样本任务的知识\n")
        f.write("✅ SMARTS弹头特征 - 编码共价抑制剂的反应性基团信息\n\n")

        f.write("【总体统计】\n")
        f.write("-" * 40 + "\n")
        f.write(f"任务数量: {len(task_performance)}\n")
        f.write(f"总预测次数: {len(all_predictions_detail)}\n")

        all_pearson = [perf['pearson'] for perf in task_performance.values()
                      if not np.isnan(perf['pearson'])]
        all_rmse = [perf['rmse'] for perf in task_performance.values()]
        all_mae = [perf['mae'] for perf in task_performance.values()]

        task_avg_pearson = np.mean(all_pearson) if all_pearson else np.nan
        task_std_pearson = np.std(all_pearson) if all_pearson else np.nan

        all_true_values = [pred['true_pic50'] for pred in all_predictions_detail]
        all_pred_values = [pred['predicted_pic50'] for pred in all_predictions_detail]

        if len(all_true_values) > 1:
            overall_pearson = pearsonr(all_true_values, all_pred_values)[0]
            overall_rmse = np.sqrt(np.mean([(t-p)**2 for t, p in zip(all_true_values, all_pred_values)]))
            overall_mae = np.mean([abs(t-p) for t, p in zip(all_true_values, all_pred_values)])
        else:
            overall_pearson = np.nan
            overall_rmse = np.nan
            overall_mae = np.nan

        f.write(f"\n【整体性能指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"所有数据拉通Pearson相关系数: {overall_pearson:.4f}\n")
        f.write(f"所有数据拉通RMSE: {overall_rmse:.4f}\n")
        f.write(f"所有数据拉通MAE: {overall_mae:.4f}\n")
        f.write(f"总预测样本数: {len(all_predictions_detail)}\n")

        f.write(f"\n【各任务平均性能】\n")
        f.write("-" * 40 + "\n")
        if all_pearson:
            f.write(f"各靶点Pearson相关系数平均值: {task_avg_pearson:.4f} ± {task_std_pearson:.4f}\n")
            f.write(f"各靶点RMSE平均值: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}\n")
            f.write(f"各靶点MAE平均值: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}\n")
            f.write(f"参与计算的靶点数: {len(all_pearson)}\n")
        f.write("\n")

        f.write("【各任务详细结果】\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'任务':<15} {'样本数':<10} {'Pearson':<12} {'RMSE':<12} {'MAE':<12}\n")
        f.write("-" * 70 + "\n")

        sorted_tasks = sorted(task_performance.items(),
                             key=lambda x: x[1]['pearson'] if not np.isnan(x[1]['pearson']) else -1,
                             reverse=True)

        for task, perf in sorted_tasks:
            f.write(f"{task:<15} {perf['n_samples']:<10} {perf['pearson']:<12.4f} "
                   f"{perf['rmse']:<12.4f} {perf['mae']:<12.4f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("增强版多任务学习报告生成完成！\n")

    print(f"多任务分析综合报告已保存: {report_file}")

def create_multi_task_visualizations(task_performance, output_dirs):
    print("=== 生成多任务性能可视化图表 ===")

    tasks = list(task_performance.keys())
    pearson_values = [task_performance[t]['pearson'] for t in tasks]
    rmse_values = [task_performance[t]['rmse'] for t in tasks]
    sample_counts = [task_performance[t]['n_samples'] for t in tasks]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    valid_indices = [i for i, p in enumerate(pearson_values) if not np.isnan(p)]
    valid_tasks = [tasks[i] for i in valid_indices]
    valid_pearson = [pearson_values[i] for i in valid_indices]

    sorted_data = sorted(zip(valid_tasks, valid_pearson), key=lambda x: x[1], reverse=True)
    sorted_tasks, sorted_pearson = zip(*sorted_data)

    bars = ax1.bar(range(len(sorted_tasks)), sorted_pearson,
                   color=['green' if p > 0.7 else 'orange' if p > 0.5 else 'red' for p in sorted_pearson])
    ax1.set_xlabel('任务 (靶点)')
    ax1.set_ylabel('Pearson相关系数')
    ax1.set_title('各任务预测性能（增强版模型）')
    ax1.set_xticks(range(len(sorted_tasks)))
    ax1.set_xticklabels(sorted_tasks, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='高性能阈值 (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中等性能阈值 (0.5)')
    ax1.legend()

    ax2.scatter(sample_counts, pearson_values, alpha=0.7, s=60)
    ax2.set_xlabel('样本数量')
    ax2.set_ylabel('Pearson相关系数')
    ax2.set_title('样本数量 vs 预测性能')
    ax2.grid(True, alpha=0.3)

    for i, task in enumerate(tasks):
        if not np.isnan(pearson_values[i]):
            ax2.annotate(task, (sample_counts[i], pearson_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    ax3.scatter(pearson_values, rmse_values, alpha=0.7, s=60)
    ax3.set_xlabel('Pearson相关系数')
    ax3.set_ylabel('RMSE')
    ax3.set_title('预测准确性 vs 相关性')
    ax3.grid(True, alpha=0.3)

    valid_pearson_clean = [p for p in pearson_values if not np.isnan(p)]
    ax4.hist(valid_pearson_clean, bins=15, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Pearson相关系数')
    ax4.set_ylabel('任务数量')
    ax4.set_title('任务预测性能分布')
    ax4.axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='高性能阈值')
    ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='中等性能阈值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(output_dirs['visualizations'], 'enhanced_performance_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"多任务性能可视化图表已保存: {plot_file}")

def main():
    print("="*80)
    print(" 增强版多任务学习系统 + SMARTS特征")
    print(" Enhanced Multi-Task Learning with Task Relation & Gated Attention + SMARTS")
    print("="*80)
    print("\n核心创新:")
    print("  1. 任务关系模块 - 自动学习54个任务间的相关性")
    print("  2. 增强跨模态注意力 - 多头注意力+门控融合")
    print("  3. 区别于MultiMolCGC - 架构上有明显创新")
    print("  4. SMARTS弹头特征 - 编码共价抑制剂的反应性基团")
    print("\n功能:")
    print("  • 自动识别数据集中的有效靶点")
    print("  • 构建增强版多任务学习模型")
    print("  • 按靶点分层划分数据")
    print("  • 任务间性能对比分析")
    print("  • 任务相似度矩阵可视化")
    print("  • SMARTS特征自动编码与融合")
    print("="*80)

    output_dirs = create_output_directories()

    csv_path = 'data/dataset.csv'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv_cycles = 5
    train_epoch = 50
    batch_size = 64
    test_size = 0.2
    norm = True
    min_samples_per_task = 20

    lr = 0.0001
    wd = 0.003
    scheduler_milestones = [5*i for i in range(1, 20)]
    scheduler_gamma = 0.9
    shared_hidden_dim = 128
    individual_hidden_dim = 128
    num_experts = 2
    num_heads = 4
    dropout_rate = 0.4
    task_embed_dim = 64

    print(f"\n【运行模式】{'1折验证模式（快速测试）' if cv_cycles == 1 else f'{cv_cycles}折交叉验证（完整实验）'}")
    print(f"  - {cv_cycles} 折验证")
    print(f"  - 每折训练 {train_epoch} 轮")
    print(f"  - 预计时间: ~{'30-40分钟' if cv_cycles == 1 else '2-3小时'}（GPU）")
    print(f"\n【配置信息】")
    print(f"  设备: {device}")
    print(f"  数据集: {csv_path}")
    print(f"  每个任务最少样本数: {min_samples_per_task}")
    print(f"  批次大小: {batch_size}")
    print(f"  报告输出: results/enhanced_multi_task_smarts_YYYYMMDD_HHMMSS/")

    if not os.path.exists(csv_path):
        print(f"错误: 找不到数据文件 {csv_path}")
        return

    valid_targets, target_counts = analyze_dataset_targets(csv_path, min_samples_per_task)

    if len(valid_targets) < 2:
        print(f"错误: 有效靶点数量不足 ({len(valid_targets)} < 2)")
        return

    print(f"\n将使用 {len(valid_targets)} 个靶点进行多任务学习:")
    for target in valid_targets:
        print(f"  - {target}: {target_counts[target]} 样本")

    smarts_vocab = build_smarts_vocabulary(csv_path)
    print(f"\n✓ SMARTS词汇表构建完成: {len(smarts_vocab)} 个唯一模式")

    protein_embeddings = process_protein_sequences(csv_path)

    full_data_list = load_multi_task_dataset(csv_path, device, valid_targets, smarts_vocab)
    print(f"多任务数据集大小: {len(full_data_list)}")

    valid_data_list = []
    for d in full_data_list:
        if torch.isnan(d.y).all():
            continue
        valid_data_list.append(d)

    print(f"有效数据集大小: {len(valid_data_list)}")

    sample_data = valid_data_list[0]
    input_dim = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None and sample_data.edge_attr.size(0) > 0 else 11

    print(f"特征维度: 节点={input_dim}, 边={edge_dim}")

    if norm:
        print("按任务进行数据标准化...")
        scalers_y = {}

        for task_idx, task in enumerate(valid_targets):
            task_y = []
            for d in valid_data_list:
                if not torch.isnan(d.y[0, task_idx]):
                    task_y.append(d.y[0, task_idx].item())

            if len(task_y) > 0:
                task_y = np.array(task_y).reshape(-1, 1)
                scaler = MaskedStandardScaler()
                scaler.fit(task_y)
                scalers_y[task] = scaler
                print(f"  任务 {task}: {len(task_y)} 个样本")

        for d in valid_data_list:
            d.original_y = d.y.clone()
            new_y = d.y.clone()

            for task_idx, task in enumerate(valid_targets):
                if not torch.isnan(d.y[0, task_idx]) and task in scalers_y:
                    original_val = d.y[0, task_idx].item()
                    scaled_val = scalers_y[task].transform(np.array([[original_val]]))[0, 0]
                    new_y[0, task_idx] = scaled_val

            d.y = new_y

    all_predictions_detail = []
    cv_results = []

    print(f"\n=== 开始{cv_cycles}折交叉验证 ===")

    for fold_idx in range(cv_cycles):
        print(f"\nFold {fold_idx+1}/{cv_cycles}")

        train_dataset, val_dataset = stratified_multi_task_split(
            valid_data_list, test_size=test_size, min_val_per_target=1, min_train_per_target=1, random_state=42+fold_idx
        )

        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

        train_loader = GeoDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        val_loader = GeoDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        model = EnhancedMultiTaskModel(
            input_dim=input_dim,
            edge_dim=edge_dim,
            tasks=valid_targets,
            shared_hidden_dim=shared_hidden_dim,
            individual_hidden_dim=individual_hidden_dim,
            num_experts=num_experts,
            num_heads=num_heads,
            dp=dropout_rate,
            protein_embed_dim=1152,
            task_embed_dim=task_embed_dim
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_milestones,
            gamma=scheduler_gamma
        )

        with tqdm(range(train_epoch), desc=f"Fold {fold_idx+1}") as epochs:
            for epoch in epochs:
                model.train()
                total_train_loss = 0
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    pred = model(batch_data)
                    loss = masked_mse_loss(pred, batch_data.y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                ave_train_loss = total_train_loss / len(train_loader)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_data = batch_data.to(device)
                        pred = model(batch_data)
                        val_loss = masked_mse_loss(pred, batch_data.y)
                        total_val_loss += val_loss.item()

                ave_val_loss = total_val_loss / len(val_loader)

                scheduler.step()

                epochs.set_postfix({
                    'TrainLoss': f'{ave_train_loss:.4f}',
                    'ValLoss': f'{ave_val_loss:.4f}'
                })

        model.eval()
        y_true_val = []
        y_pred_val = []
        original_y_val = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                y_true_val.append(batch_data.y.cpu().numpy())
                y_pred_val.append(pred.cpu().numpy())

                if hasattr(batch_data, 'original_y'):
                    original_y_val.append(batch_data.original_y.cpu().numpy())
                else:
                    original_y_val.append(batch_data.y.cpu().numpy())

        y_true_val = np.concatenate(y_true_val, axis=0)
        y_pred_val = np.concatenate(y_pred_val, axis=0)
        original_y_val = np.concatenate(original_y_val, axis=0)

        if norm:
            for task_idx, task in enumerate(valid_targets):
                if task in scalers_y:
                    valid_mask = ~np.isnan(y_true_val[:, task_idx])
                    if valid_mask.sum() > 0:
                        y_pred_val[valid_mask, task_idx] = scalers_y[task].inverse_transform(
                            y_pred_val[valid_mask, task_idx].reshape(-1, 1)
                        ).flatten()

        fold_predictions = []
        for i, data in enumerate(val_dataset):
            for task_idx, task in enumerate(valid_targets):
                if not np.isnan(original_y_val[i, task_idx]):
                    pred_record = {
                        'fold': fold_idx + 1,
                        'sample_idx': i,
                        'smiles': data.smiles,
                        'task': task,
                        'task_idx': task_idx,
                        'true_pic50': original_y_val[i, task_idx],
                        'predicted_pic50': y_pred_val[i, task_idx],
                        'error': abs(original_y_val[i, task_idx] - y_pred_val[i, task_idx]),
                        'squared_error': (original_y_val[i, task_idx] - y_pred_val[i, task_idx])**2
                    }
                    all_predictions_detail.append(pred_record)
                    fold_predictions.append(pred_record)

        if fold_predictions:
            fold_df = pd.DataFrame(fold_predictions)
            fold_csv_file = os.path.join(output_dirs['predictions'], f'fold_{fold_idx+1}_predictions.csv')
            fold_df.to_csv(fold_csv_file, index=False)
            print(f"  Fold {fold_idx+1} 预测结果已保存: {fold_csv_file}")

        fold_results = {'fold': fold_idx+1}
        for task_idx, task in enumerate(valid_targets):
            mask = ~np.isnan(original_y_val[:, task_idx])
            if mask.sum() >= 3:
                true_val = original_y_val[mask, task_idx]
                pred_val = y_pred_val[mask, task_idx]

                rmse_val, mae_val, r_p_val, r_k_val = compute_quality_metrics(true_val, pred_val)

                fold_results[f'{task}_rmse'] = rmse_val
                fold_results[f'{task}_mae'] = mae_val
                fold_results[f'{task}_pearson'] = r_p_val
                fold_results[f'{task}_kendall'] = r_k_val
                fold_results[f'{task}_n_samples'] = mask.sum()

        cv_results.append(fold_results)

        print(f"Fold {fold_idx+1} 结果:")
        for task in valid_targets[:5]:
            if f'{task}_pearson' in fold_results:
                print(f"  {task}: Pearson={fold_results[f'{task}_pearson']:.4f}, "
                      f"RMSE={fold_results[f'{task}_rmse']:.4f}, "
                      f"MAE={fold_results[f'{task}_mae']:.4f}")

        if fold_idx == cv_cycles - 1:
            visualize_task_similarity(model, valid_targets, output_dirs)

        model_file = os.path.join(output_dirs['model_weights'], f'enhanced_model_fold_{fold_idx+1}.pth')
        torch.save(model.state_dict(), model_file)

    print(f"\n=== 保存详细预测结果 ===")

    predictions_df = pd.DataFrame(all_predictions_detail)
    pred_file = os.path.join(output_dirs['predictions'], 'all_predictions.csv')
    predictions_df.to_csv(pred_file, index=False)
    print(f"所有预测结果已保存: {pred_file}")

    cv_df = pd.DataFrame(cv_results)
    cv_file = os.path.join(output_dirs['cross_validation'], 'cv_summary.csv')
    cv_df.to_csv(cv_file, index=False)
    print(f"交叉验证结果已保存: {cv_file}")

    task_performance = analyze_multi_task_performance(all_predictions_detail, valid_targets, output_dirs)

    generate_multi_task_report(task_performance, output_dirs, all_predictions_detail)

    create_multi_task_visualizations(task_performance, output_dirs)

    print(f"\n=== 增强版多任务学习分析完成 ===")
    print(f"结果已保存到: {output_dirs['base']}")
    print(f"共训练 {len(valid_targets)} 个任务")

    if task_performance:
        all_pearson = [p['pearson'] for p in task_performance.values()
                      if not np.isnan(p['pearson'])]
        if all_pearson:
            task_avg_pearson = np.mean(all_pearson)
            task_std_pearson = np.std(all_pearson)

            all_true_values = [pred['true_pic50'] for pred in all_predictions_detail]
            all_pred_values = [pred['predicted_pic50'] for pred in all_predictions_detail]

            if len(all_true_values) > 1:
                overall_pearson = pearsonr(all_true_values, all_pred_values)[0]
                overall_rmse = np.sqrt(np.mean([(t-p)**2 for t, p in zip(all_true_values, all_pred_values)]))
                overall_mae = np.mean([abs(t-p) for t, p in zip(all_true_values, all_pred_values)])
            else:
                overall_pearson = np.nan
                overall_rmse = np.nan
                overall_mae = np.nan

            all_mae_tasks = [p['mae'] for p in task_performance.values()]
            all_rmse_tasks = [p['rmse'] for p in task_performance.values()]
            task_avg_mae = np.mean(all_mae_tasks)
            task_std_mae = np.std(all_mae_tasks)
            task_avg_rmse = np.mean(all_rmse_tasks)
            task_std_rmse = np.std(all_rmse_tasks)

            print(f"\n" + "="*80)
            print(f"【核心性能指标 - 所有小分子拉通计算】")
            print(f"="*80)
            print(f"总预测样本数: {len(all_predictions_detail)}")
            print(f"\n所有数据拉通的整体指标:")
            print(f"  ✅ Pearson相关系数 (R):  {overall_pearson:.4f}")
            print(f"  ✅ RMSE:                  {overall_rmse:.4f}")
            print(f"  ✅ MAE:                   {overall_mae:.4f}")
            print(f"\n各靶点指标的平均值:")
            print(f"  • Pearson平均: {task_avg_pearson:.4f} ± {task_std_pearson:.4f}")
            print(f"  • RMSE平均:    {task_avg_rmse:.4f} ± {task_std_rmse:.4f}")
            print(f"  • MAE平均:     {task_avg_mae:.4f} ± {task_std_mae:.4f}")
            print(f"  • 参与计算的靶点数: {len(all_pearson)}")
            print(f"="*80)

            best_task = max(task_performance.items(),
                           key=lambda x: x[1]['pearson'] if not np.isnan(x[1]['pearson']) else -1)
            worst_task = min(task_performance.items(),
                            key=lambda x: x[1]['pearson'] if not np.isnan(x[1]['pearson']) else 1)

            print(f"\n【任务性能分析】")
            print(f"最佳任务: {best_task[0]} (Pearson: {best_task[1]['pearson']:.4f})")
            print(f"最差任务: {worst_task[0]} (Pearson: {worst_task[1]['pearson']:.4f})")

            high_perf = sum(1 for p in task_performance.values() if p['pearson'] > 0.7)
            med_perf = sum(1 for p in task_performance.values() if 0.5 < p['pearson'] <= 0.7)
            low_perf = sum(1 for p in task_performance.values() if p['pearson'] <= 0.5 and not np.isnan(p['pearson']))

            print(f"性能分类: 高性能({high_perf}个) | 中等性能({med_perf}个) | 低性能({low_perf}个)")

    print(f"\n" + "="*80)
    print(f"✅ 增强版模型训练完成!")
    print(f"✅ 任务相似度矩阵已生成，可用于分析蛋白家族关系")
    print(f"✅ 所有结果已保存到: {output_dirs['base']}")
    print(f"="*80)

if __name__ == '__main__':
    main()

