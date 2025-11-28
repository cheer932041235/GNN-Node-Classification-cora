"""
数据集下载工具
自动下载Cora数据集
"""

import os
import urllib.request
import ssl
import shutil

def download_cora_dataset(force_redownload=False):
    """
    下载Cora数据集
    
    Args:
        force_redownload: 是否强制重新下载
    """
    # 数据目录
    data_dir = './data/Cora'
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # 如果已存在且不强制重新下载，检查文件完整性
    if os.path.exists(raw_dir) and not force_redownload:
        files = os.listdir(raw_dir)
        if len(files) == 8:
            print("Cora数据集已存在，跳过下载")
            return True
    
    print("=" * 50)
    print("开始下载Cora数据集...")
    print("=" * 50)
    
    # 如果强制重新下载，清理旧数据
    if force_redownload and os.path.exists(data_dir):
        print("\n正在清理旧数据...")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
            print("已删除processed目录")
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
            print("已删除raw目录")
    
    # 创建目录
    os.makedirs(raw_dir, exist_ok=True)
    
    # Cora数据集的文件列表
    files = [
        'ind.cora.x',
        'ind.cora.tx',
        'ind.cora.allx',
        'ind.cora.y',
        'ind.cora.ty',
        'ind.cora.ally',
        'ind.cora.graph',
        'ind.cora.test.index'
    ]
    
    # 使用原始数据源
    base_url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    
    # SSL设置
    ssl_context = ssl._create_unverified_context()
    
    success_count = 0
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(raw_dir, filename)
        
        # 如果文件已存在且不强制重新下载，跳过
        if os.path.exists(file_path) and not force_redownload:
            print(f"[{i}/{len(files)}] {filename} 已存在，跳过")
            success_count += 1
            continue
        
        url = f"{base_url}/{filename}"
        print(f"\n[{i}/{len(files)}] 下载: {filename}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                data = response.read()
                
                # 验证数据不为空
                if len(data) == 0:
                    print(f"    文件为空")
                    continue
                
                with open(file_path, 'wb') as out_file:
                    out_file.write(data)
                
                print(f"    下载成功 ({len(data)} 字节)")
                success_count += 1
                
        except Exception as e:
            print(f"    下载失败: {e}")
    
    print("\n" + "=" * 50)
    if success_count == len(files):
        print("所有文件下载完成！")
        return True
    else:
        print(f"警告: 只下载了 {success_count}/{len(files)} 个文件")
        return False


if __name__ == '__main__':
    download_cora_dataset(force_redownload=False)

