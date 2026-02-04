import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def visualize_anchors_nature_style(json_path, canvas_size=416):

    # 1. 读取 JSON 数据
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    anchors_rel = data.get("anchors_rel", [])
    
    # 2. 准备画布
    # 设置 DPI 和大小
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # 设置背景色 (米白色，类似纸张，比纯白更护眼且自然)
    bg_color = "#F5F5F0" 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 绘制画布边框 (代表 416x416 的输入图像大小)
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(0, canvas_size)
    ax.invert_yaxis() # 图像坐标系：原点在左上角
    
    # 辅助网格 (淡灰色)
    ax.grid(True, which='both', color='#D3D3D3', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 3. 定义 Nature 色系调色板 (HEX)
    # 顺序对应：最小 -> 最大 (或者反过来，这里为了美观，我们根据面积排序分配)
    # 颜色：苔藓绿, 湖泊蓝, 陶土红, 大地褐, 深岩灰
    nature_palette = [
        "#4A7058",  # Moss Green (苔藓绿)
        "#4F7CA2",  # Lake Blue (湖泊蓝)
        "#C87F56",  # Terracotta (陶土色)
        "#8B5A2B",  # Earth Brown (大地褐)
        "#2F4F4F"   # Dark Slate Gray (深岩灰)
    ]
    
    # 4. 数据处理：计算像素尺寸并排序
    # 我们按面积从大到小绘制，这样小框不会被大框完全遮挡（虽然我们要用透明度）
    anchors_px = []
    for w_rel, h_rel in anchors_rel:
        w_px = w_rel * canvas_size
        h_px = h_rel * canvas_size
        area = w_px * h_px
        anchors_px.append({'w': w_px, 'h': h_px, 'area': area})
    
    # 按面积从大到小排序，确保大框在底部
    anchors_px.sort(key=lambda x: x['area'], reverse=True)
    
    # 中心点坐标
    cx, cy = canvas_size / 2, canvas_size / 2

    print(f"{'Anchor Size (Px)':<20} | {'Area':<10}")
    print("-" * 35)

    # 5. 循环绘制
    for i, anc in enumerate(anchors_px):
        w = anc['w']
        h = anc['h']
        
        # 计算左上角坐标 (让 anchor 居中)
        x0 = cx - w / 2
        y0 = cy - h / 2
        
        # 选择颜色 (循环使用调色板)
        color = nature_palette[i % len(nature_palette)]
        
        # 创建矩形 Patch
        # facecolor: 填充色 (带透明度)
        # edgecolor: 边框色 (不透明，深色)
        rect = patches.Rectangle(
            (x0, y0), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.2 + (0.1 * i), # 越小的框越不透明，稍微增加对比度
            linestyle='-'
        )
        
        ax.add_patch(rect)
        
        # 添加中心点 (红点)
        ax.scatter(cx, cy, c='#8B0000', s=10, zorder=10)

        # 添加文字标注
        # 文字位置：矩形的左上角，稍微偏移一点
        label_text = f"{int(w)}x{int(h)}"
        
        # 为了防止文字重叠，交替文字的对齐位置或者颜色
        text_y_offset = -5 if i % 2 == 0 else 15
        
        ax.text(
            x0 + 5, y0 + text_y_offset, 
            label_text, 
            color=color, 
            fontsize=10, 
            fontweight='bold',
            bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.7, pad=1)
        )

        print(f"{int(w):>4} x {int(h):<4}         | {int(anc['area'])}")

    # 6. 装饰与保存
    plt.title(f"YOLOv2 Learned Anchors (Canvas: {canvas_size}x{canvas_size})", fontsize=14, color="#333333", pad=20)
    plt.xlabel("Width (px)", color="#555555")
    plt.ylabel("Height (px)", color="#555555")
    
    # 去掉多余的刻度线，保留边框
    ax.tick_params(colors='#555555')
    for spine in ax.spines.values():
        spine.set_color('#888888')

    plt.tight_layout()
    
    # 保存结果
    save_path = "anchors_visualization_nature.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可视化图片已保存至: {save_path}")

if __name__ == "__main__":
    json_file = r"dataset\anchors_k5.json"
    visualize_anchors_nature_style(json_file)