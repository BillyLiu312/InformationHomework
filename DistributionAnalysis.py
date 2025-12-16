import math
import heapq
from collections import Counter, namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格，使其看起来更像学术论文
plt.style.use('ggplot') 
# 解决中文显示问题 (如果有乱码，可以尝试注释掉这两行，或者换成英文字体)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 基础 Huffman 类与工具函数 (保持不变)
# ==========================================

class Node(namedtuple("Node", ["freq", "symbol", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

def calculate_entropy(probabilities):
    """计算香农熵 H(X) = -sum(p * log2(p))"""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def build_huffman_tree(frequency_map):
    heap = [Node(freq, sym, None, None) for sym, freq in frequency_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, prefix="", code_map=None):
    if code_map is None: code_map = {}
    if node.symbol is not None:
        code_map[node.symbol] = prefix or "0"
    else:
        if node.left: generate_codes(node.left, prefix + "0", code_map)
        if node.right: generate_codes(node.right, prefix + "1", code_map)
    return code_map

# ==========================================
# 2. 核心实验与高级绘图函数
# ==========================================

def get_masked_data(counts, threshold):
    """根据阈值生成脱敏后的数据统计"""
    safe_counts = Counter()
    merged_items = []
    masked_count = 0
    
    for disease, count in counts.items():
        if count < threshold:
            masked_count += count
            merged_items.append(disease)
        else:
            safe_counts[disease] = count
            
    if masked_count > 0:
        safe_counts['[MASKED]'] = masked_count
        
    return safe_counts, merged_items

def run_advanced_experiment():
    # --- A. 数据生成 ---
    raw_data = (
        ['Flu'] * 500 + 
        ['Cold'] * 300 + 
        ['Gastritis'] * 150 + 
        ['Diabetes'] * 40 + 
        ['Hypertension'] * 30 + 
        ['HIV'] * 5 +                  
        ['Ebola'] * 2 +                
        ['Rare_Genetic_A'] * 2 +  
        ['Rare_Genetic_B'] * 1 +
        ['Unknown_Virus'] * 1
    )
    total_count = len(raw_data)
    counts = Counter(raw_data)
    
    # 按照频率排序，为了画图好看
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    freqs = [x[1] for x in sorted_items]

    # --- B. 场景演示：阈值=10 ---
    THRESHOLD = 10
    safe_counts, merged_items = get_masked_data(counts, THRESHOLD)
    
    # 计算 Huffman 码
    tree = build_huffman_tree(safe_counts)
    codes = generate_codes(tree)
    
    # 计算熵
    raw_probs = [c/total_count for c in counts.values()]
    safe_probs = [c/total_count for c in safe_counts.values()]
    H_raw = calculate_entropy(raw_probs)
    H_safe = calculate_entropy(safe_probs)

    print(f"原始熵: {H_raw:.4f} | 脱敏后熵: {H_safe:.4f}")
    print(f"被合并的敏感类别: {merged_items}")

    # ==========================================
    # C. 高级绘图 1: 长尾分布与截断可视化
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图 1: 长尾分布 Log-Scale Plot
    indices = np.arange(len(labels))
    bars = ax1.bar(indices, freqs, color='#4A90E2', alpha=0.8, edgecolor='black')
    
    # 标记被合并的部分 (红色)
    for idx, count in enumerate(freqs):
        if count < THRESHOLD:
            bars[idx].set_color('#E74C3C') # 红色表示危险/被屏蔽
            bars[idx].set_label('Merged (Privacy Risk)') if idx == len(freqs)-1 else None
        else:
            bars[idx].set_label('Retained (Utility)') if idx == 0 else None

    # 画一条阈值线
    ax1.axhline(y=THRESHOLD, color='gray', linestyle='--', linewidth=2, label=f'Threshold k={THRESHOLD}')
    
    ax1.set_yscale('log') # 关键！使用对数坐标展示长尾
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Frequency (Log Scale)')
    ax1.set_title(f'Distribution & Privacy Cut-off (k={THRESHOLD})\n"Cutting the Long Tail"', fontsize=14)
    ax1.legend()
    ax1.text(len(labels)-1, 1.5, "High Privacy Risk\n(High Self-Info)", color='#E74C3C', ha='center', fontweight='bold')

    # ==========================================
    # D. 高级绘图 2: 敏感度分析 (Trade-off Curve)
    # ==========================================
    
    # 动态计算不同阈值下的指标
    threshold_range = range(1, 51)
    info_losses = []
    masked_ratios = []
    
    for k in threshold_range:
        s_counts, _ = get_masked_data(counts, k)
        s_probs = [c/total_count for c in s_counts.values()]
        h_s = calculate_entropy(s_probs)
        
        loss = H_raw - h_s
        masked_ratio = sum(c for tag, c in s_counts.items() if tag == '[MASKED]') / total_count
        
        info_losses.append(loss)
        masked_ratios.append(masked_ratio * 100) # 百分比

    # 双轴图表
    color_loss = 'tab:blue'
    color_priv = 'tab:orange'

    ax2.set_xlabel('Privacy Threshold (k-anonymity parameter)')
    ax2.set_title('Privacy vs. Utility Trade-off Analysis', fontsize=14)

    # 左轴：信息损失
    ax2.plot(threshold_range, info_losses, color=color_loss, linewidth=3, label='Info Loss (bits)')
    ax2.set_ylabel('Information Loss (Bits)', color=color_loss, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.fill_between(threshold_range, 0, info_losses, color=color_loss, alpha=0.1)

    # 右轴：被屏蔽的人群比例
    ax3 = ax2.twinx()  
    ax3.plot(threshold_range, masked_ratios, color=color_priv, linestyle='--', linewidth=2, label='Masked Population %')
    ax3.set_ylabel('Masked Population (%)', color=color_priv, fontsize=12)
    ax3.tick_params(axis='y', labelcolor=color_priv)
    
    # 标注当前选择的点
    current_loss = H_raw - H_safe
    ax2.scatter([THRESHOLD], [current_loss], s=100, c='red', zorder=10)
    ax2.annotate(f'Current k={THRESHOLD}', xy=(THRESHOLD, current_loss), xytext=(THRESHOLD+5, current_loss+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 布局调整
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_advanced_experiment()