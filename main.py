"""
图神经网络(GNN)入门项目 - 主运行脚本
一键运行完整流程：下载数据 -> 训练模型 -> 可视化结果

使用方法：
    python main.py              # 运行完整流程
    python main.py --skip-download  # 跳过数据下载
    python main.py --skip-viz   # 跳过可视化
    python main.py --quick      # 快速模式（较少的训练轮数）
"""

import os
import sys
import argparse
from utils.download_dataset import download_cora_dataset
from data_loader import CoraDataLoader
from model import GCN
from train import Trainer
import torch


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图神经网络入门项目')
    parser.add_argument('--skip-download', action='store_true', help='跳过数据集下载')
    parser.add_argument('--skip-viz', action='store_true', help='跳过结果可视化')
    parser.add_argument('--quick', action='store_true', help='快速模式（训练50轮）')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数（默认200）')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("图神经网络(GNN)入门项目 - 节点分类任务")
    print("=" * 60)
    
    # ===================== 步骤1: 初始化项目结构 =====================
    print("\n【步骤 1/5】初始化项目结构...")
    os.makedirs('outputs/images', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    print("项目目录已就绪")
    
    # ===================== 步骤2: 下载数据集 =====================
    if not args.skip_download:
        print("\n【步骤 2/5】下载Cora数据集...")
        try:
            success = download_cora_dataset(force_redownload=False)
            if not success:
                print("\n警告: 数据集下载不完整，但会尝试继续运行...")
        except Exception as e:
            print(f"\n警告: 下载出错: {e}")
            print("提示：如果网络问题，请手动下载数据集")
    else:
        print("\n【步骤 2/5】跳过数据下载")
    
    # ===================== 步骤3: 加载数据并可视化 =====================
    print("\n【步骤 3/5】加载数据集...")
    try:
        loader = CoraDataLoader()
        data, num_classes = loader.get_data()
        
        # 生成数据可视化
        print("\n生成数据集可视化图片...")
        loader.visualize_label_distribution()
        loader.visualize_graph(num_nodes=300)
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("\n解决方案：")
        print("  1. 检查data/Cora/raw/目录下是否有8个数据文件")
        print("  2. 运行: python utils/download_dataset.py")
        print("  3. 或手动下载数据集（见README.md）")
        sys.exit(1)
    
    # ===================== 步骤4: 训练模型 =====================
    print("\n【步骤 4/5】训练GCN模型...")
    
    # 创建模型
    model = GCN(
        num_features=data.num_node_features,
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.5
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        data=data,
        lr=0.01,
        weight_decay=5e-4
    )
    
    # 训练模型
    epochs = 50 if args.quick else args.epochs
    trainer.train(
        epochs=epochs,
        early_stopping=True,
        patience=20
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 保存最终模型
    torch.save(model.state_dict(), 'outputs/models/final_model.pth')
    print("模型已保存")
    
    # ===================== 步骤5: 可视化结果 =====================
    if not args.skip_viz:
        print("\n【步骤 5/5】生成结果可视化...")
        
        try:
            from visualize import Visualizer
            
            visualizer = Visualizer(model, data)
            
            print("  - 生成PCA降维可视化...")
            visualizer.visualize_embeddings_pca()
            
            print("  - 生成预测结果对比图...")
            visualizer.visualize_predictions()
            
            print("  - 生成混淆矩阵...")
            visualizer.plot_confusion_matrix()
            
            print("  - 生成t-SNE降维可视化（较慢）...")
            visualizer.visualize_embeddings_tsne()
            
        except Exception as e:
            print(f"警告: 可视化部分失败: {e}")
    else:
        print("\n【步骤 5/5】跳过结果可视化")
    
    # ===================== 完成 =====================
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  图片文件：outputs/images/")
    print("  模型文件：outputs/models/")
    print("\n下一步：")
    print("  - 查看outputs/images/目录下的可视化结果")
    print("  - 尝试修改model.py中的模型结构")
    print("  - 尝试调整train.py中的超参数")
    print("  - 阅读README.md了解更多")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n程序出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

