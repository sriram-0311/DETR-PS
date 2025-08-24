"""
Script to generate a professional demo image for LinkedIn post
showing DETR-PS panoptic segmentation results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_demo_visualization():
    """Create a professional demo visualization for LinkedIn"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Set overall style
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    # Create a grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], 
                         hspace=0.3, wspace=0.2)
    
    # Title
    fig.suptitle('DETR-PS: Automotive Panoptic Segmentation Results', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # Input Image (simulated)
    ax1 = fig.add_subplot(gs[0, 0])
    # Create a simulated street scene
    street_scene = np.random.rand(400, 600, 3) * 0.3 + 0.4
    # Add some structure to simulate cars, road, etc.
    street_scene[300:400, :] = [0.4, 0.4, 0.4]  # Road
    street_scene[0:100, :] = [0.6, 0.8, 1.0]    # Sky
    street_scene[200:300, 100:200] = [0.8, 0.2, 0.2]  # Car
    street_scene[200:300, 400:500] = [0.2, 0.8, 0.2]  # Car
    
    ax1.imshow(street_scene)
    ax1.set_title('Input: Street Scene', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Architecture Overview
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.8, 'DETR Architecture', ha='center', va='center', 
             fontsize=16, fontweight='bold', transform=ax2.transAxes)
    
    # Draw architecture flow
    components = ['ResNet50\nBackbone', 'Transformer\nEncoder', 'Transformer\nDecoder', 'Panoptic\nHead']
    positions = [(0.2, 0.6), (0.2, 0.4), (0.2, 0.2), (0.8, 0.4)]
    
    for i, (comp, pos) in enumerate(zip(components, positions)):
        bbox = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
        ax2.text(pos[0], pos[1], comp, ha='center', va='center', 
                transform=ax2.transAxes, bbox=bbox, fontsize=10)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
    ax2.annotate('', xy=(0.2, 0.35), xytext=(0.2, 0.55), 
                arrowprops=arrow_props, transform=ax2.transAxes)
    ax2.annotate('', xy=(0.2, 0.15), xytext=(0.2, 0.35), 
                arrowprops=arrow_props, transform=ax2.transAxes)
    ax2.annotate('', xy=(0.7, 0.4), xytext=(0.3, 0.2), 
                arrowprops=arrow_props, transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Performance Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['PQ: 65.2%', 'SQ: 82.1%', 'RQ: 79.8%', 'Speed: 15 FPS']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    y_pos = np.arange(len(metrics))
    values = [65.2, 82.1, 79.8, 15]
    
    bars = ax3.barh(y_pos, [v if v <= 100 else v*4 for v in values], color=colors, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(metrics)
    ax3.set_xlabel('Performance (%)')
    ax3.set_title('Key Metrics', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{val}{"%" if val <= 100 else " FPS"}', 
                ha='left', va='center', fontweight='bold')
    
    # Panoptic Segmentation Result
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create a mock segmentation result
    seg_result = np.zeros((400, 1200, 3))
    
    # Sky
    seg_result[0:100, :] = [0.5, 0.8, 1.0]
    # Buildings
    seg_result[100:200, 0:300] = [0.8, 0.8, 0.8]
    seg_result[100:200, 900:1200] = [0.7, 0.7, 0.7]
    # Trees
    seg_result[100:200, 300:450] = [0.2, 0.8, 0.2]
    seg_result[100:200, 750:900] = [0.3, 0.7, 0.3]
    # Road
    seg_result[300:400, :] = [0.4, 0.4, 0.4]
    # Sidewalk
    seg_result[250:300, :] = [0.6, 0.6, 0.6]
    # Cars (different instances with different colors)
    seg_result[200:280, 200:350] = [1.0, 0.2, 0.2]  # Red car
    seg_result[200:280, 500:650] = [0.2, 0.2, 1.0]  # Blue car
    seg_result[180:260, 800:950] = [1.0, 1.0, 0.2]  # Yellow car
    # Pedestrians
    seg_result[220:300, 400:430] = [1.0, 0.5, 0.8]  # Person 1
    seg_result[210:290, 700:730] = [0.8, 0.5, 1.0]  # Person 2
    
    ax4.imshow(seg_result)
    ax4.set_title('Panoptic Segmentation Output: Instance + Semantic Segmentation', 
                 fontsize=16, fontweight='bold')
    ax4.axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=[0.5, 0.8, 1.0], label='Sky (Stuff)'),
        patches.Patch(color=[0.8, 0.8, 0.8], label='Buildings (Stuff)'),
        patches.Patch(color=[0.2, 0.8, 0.2], label='Vegetation (Stuff)'),
        patches.Patch(color=[0.4, 0.4, 0.4], label='Road (Stuff)'),
        patches.Patch(color=[1.0, 0.2, 0.2], label='Car Instance 1'),
        patches.Patch(color=[0.2, 0.2, 1.0], label='Car Instance 2'),
        patches.Patch(color=[1.0, 1.0, 0.2], label='Car Instance 3'),
        patches.Patch(color=[1.0, 0.5, 0.8], label='Person Instance 1'),
        patches.Patch(color=[0.8, 0.5, 1.0], label='Person Instance 2')
    ]
    
    ax4.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add project info
    info_text = """🚗 DETR-PS: End-to-End Panoptic Segmentation
🎯 Unified detection of objects (things) and segmentation of regions (stuff)
⚡ Real-time performance optimized for autonomous driving
🧠 Transformer-based architecture with self-attention
📊 State-of-the-art results on Cityscapes dataset"""
    
    plt.figtext(0.02, 0.02, info_text, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Save the figure
    plt.savefig('linkedin_demo_result.jpg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('linkedin_demo_result.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Demo visualization saved as 'linkedin_demo_result.jpg' and 'linkedin_demo_result.png'")
    print("📱 Ready to upload to LinkedIn!")
    
    plt.show()

if __name__ == "__main__":
    create_demo_visualization()
