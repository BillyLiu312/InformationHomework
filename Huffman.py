import math
import heapq
from collections import Counter, namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 定义基础数据结构与 Huffman 类
# ==========================================

class Node(namedtuple("Node", ["freq", "symbol", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

def calculate_entropy(probabilities):
    """计算香农熵 H(X) = -sum(p * log2(p))"""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def build_huffman_tree(frequency_map):
    """标准的 Huffman 树构建过程"""
    heap = [Node(freq, sym, None, None) for sym, freq in frequency_map.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        # 合并节点
        merged = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, merged)
        
    return heap[0]

def generate_codes(node, prefix="", code_map=None):
    """递归生成 Huffman 编码表"""
    if code_map is None:
        code_map = {}
    
    if node.symbol is not None:
        code_map[node.symbol] = prefix or "0" # 处理单节点情况
    else:
        if node.left: generate_codes(node.left, prefix + "0", code_map)
        if node.right: generate_codes(node.right, prefix + "1", code_map)
    
    return code_map

# ==========================================
# 2. 模拟场景：基于信息熵的非对称脱敏
# ==========================================

def run_privacy_experiment():
    # --- A. 模拟原始数据 (长尾分布) ---
    # 假设这是某医院一周的诊断记录
    # 常见病: Flu (流感), Cold (感冒)
    # 罕见病 (隐私敏感): HIV, Rare_Genetic_Disorder, Ebola
    raw_data = (
        ['Flu'] * 500 + 
        ['Common_Cold'] * 300 + 
        ['Gastritis'] * 150 + 
        ['Diabetes'] * 40 + 
        ['HIV'] * 5 +                  # 敏感/高自信息量
        ['Ebola'] * 2 +                # 极度敏感
        ['Rare_Genetic_Type_A'] * 2 +  # 极度敏感
        ['Rare_Genetic_Type_B'] * 1    # 极度敏感
    )
    
    total_count = len(raw_data)
    counts = Counter(raw_data)
    
    # --- B. 设定隐私策略 ---
    # 策略：如果某疾病的出现概率极低（自信息量过高），则将其归类为 "Other_Sensitive"
    # 这相当于在 Huffman 树构建前，强制合并叶子节点
    
    PRIVACY_THRESHOLD_COUNT = 10  # 类似 K-Anonymity, k=10
    
    # 处理数据
    safe_counts = Counter()
    merged_items = []
    
    for disease, count in counts.items():
        if count < PRIVACY_THRESHOLD_COUNT:
            safe_counts['[MASKED_DISEASE]'] += count
            merged_items.append(disease)
        else:
            safe_counts[disease] = count
            
    # --- C. 构建 Huffman 树并编码 ---
    # 对 "脱敏后" 的分布进行编码
    huffman_tree = build_huffman_tree(safe_counts)
    huffman_codes = generate_codes(huffman_tree)
    
    # --- D. 结果分析与计算 ---
    
    # 1. 计算原始熵 vs 脱敏后熵
    raw_probs = [c/total_count for c in counts.values()]
    safe_probs = [c/total_count for c in safe_counts.values()]
    
    H_original = calculate_entropy(raw_probs)
    H_safe = calculate_entropy(safe_probs)
    
    # 2. 生成报告表格
    report_data = []
    for disease, count in counts.items():
        prob = count / total_count
        self_info = -math.log2(prob) # 自信息量
        
        is_masked = disease in merged_items
        final_code = huffman_codes['[MASKED_DISEASE]'] if is_masked else huffman_codes[disease]
        
        report_data.append({
            "Disease (Symbol)": disease,
            "Count": count,
            "Self-Info (bits)": round(self_info, 2),
            "Status": "PROTECTED/MERGED" if is_masked else "Original",
            "Assigned Code": final_code
        })
        
    df = pd.DataFrame(report_data).sort_values(by="Count", ascending=False)
    
    # ==========================================
    # 3. 输出与可视化
    # ==========================================
    
    print("=== Group Project Experiment Result ===")
    print(f"Original Entropy H(X): {H_original:.4f} bits")
    print(f"Post-Privacy Entropy H(Y): {H_safe:.4f} bits")
    print(f"Information Loss (Privacy Gain): {H_original - H_safe:.4f} bits")
    print("\n--- Coding Table (Top Rows) ---")
    print(df.to_string(index=False))
    
    # 可视化绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制频率分布条形图
    colors = ['red' if x == 'PROTECTED/MERGED' else 'skyblue' for x in df['Status']]
    plt.barh(df['Disease (Symbol)'], df['Count'], color=colors)
    plt.xlabel('Frequency (Count)')
    plt.title('Distribution of Diseases: Red bars are merged for Privacy')
    plt.gca().invert_yaxis() # 频率高的在上面
    
    # 添加文字标注
    plt.figtext(0.5, 0.01, 
                f"Privacy Constraint: Merge if count < {PRIVACY_THRESHOLD_COUNT}\n"
                f"Rare diseases (High Self-Info) share the code '{huffman_codes.get('[MASKED_DISEASE]')}'", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_privacy_experiment()