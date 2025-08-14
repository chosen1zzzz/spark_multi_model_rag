"""
图片向量化脚本：从all_pdf_enhanced_chunks.json提取图片并生成CLIP embeddings
输入: outputs/output_v1_3_with_chunk/all_pdf_enhanced_chunks.json
输出: outputs/output_v1_5_vlm/image_embeddings.json
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    print("警告: 需要安装依赖包")
    print("pip install sentence-transformers pillow")
    CLIP_AVAILABLE = False


class ImageEmbeddingGenerator:
    """图片向量化生成器"""
    
    def __init__(self, model_name: str = "/mnt/workspace/AISumerCamp_multiModal_RAG/models/clip-ViT-B-32"):
        """
        初始化CLIP模型
        Args:
            model_name: CLIP模型名称，默认使用clip-ViT-B-32
        """
        if not CLIP_AVAILABLE:
            raise ImportError("请先安装必要的依赖包")
            
        print(f"正在加载CLIP模型: {model_name}")

        # 检测设备：有CUDA则用GPU，否则用CPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"检测到设备: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print(f"CLIP模型加载完成，运行在: {device}")
    
    def extract_image_info_from_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """
        从chunks文件中提取图片信息
        Args:
            chunks_file: all_pdf_enhanced_chunks.json文件路径
        Returns:
            图片信息列表
        """
        print(f"正在读取chunks文件: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        image_info = []
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+\.(?:jpg|jpeg|png|gif))\)')
        
        for chunk in tqdm(chunks, desc="提取图片信息"):
            content = chunk.get('content', '')
            chunk_type = chunk.get('type', '')
            metadata = chunk.get('metadata', {})
            
            # 查找图片路径
            matches = image_pattern.findall(content)
            
            for description, image_path in matches:
                # 构建完整的图片路径
                # 从metadata中获取文件名，构建对应的图片目录路径
                file_name = metadata.get('file_name', '')
                if file_name:
                    # 移除.pdf扩展名
                    base_name = file_name.replace('.pdf', '')
                    # 构建图片路径：data_base_json_content/文件名/文件名/auto/images/xxx.jpg
                    full_image_path = os.path.join(
                        os.path.dirname(chunks_file),
                        'data_base_json_content',
                        base_name,
                        base_name,
                        'auto',
                        image_path
                    )
                else:
                    # 如果没有文件名，尝试通用路径
                    full_image_path = os.path.join(os.path.dirname(chunks_file), 'data_base_json_content', image_path)

                full_image_path = os.path.normpath(full_image_path)
                
                image_info.append({
                    'image_path': image_path,  # 相对路径
                    'full_path': full_image_path,  # 绝对路径
                    'description': description,
                    'chunk_id': chunk.get('id', ''),
                    'chunk_type': chunk_type,
                    'file_name': metadata.get('file_name', ''),
                    'page': metadata.get('page', ''),
                    'content_type': metadata.get('content_type', ''),
                    'chunk_content': content  # 保存完整内容作为上下文
                })
        
        print(f"提取到 {len(image_info)} 张图片")
        return image_info
    
    def generate_embeddings(self, image_info: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        批量生成图片embeddings
        Args:
            image_info: 图片信息列表
            batch_size: 批处理大小
        Returns:
            包含embeddings的图片信息列表
        """
        print(f"开始生成图片embeddings，批大小: {batch_size}")
        
        results = []
        valid_images = []
        
        # 首先检查图片文件是否存在
        print("检查图片文件...")
        for info in tqdm(image_info, desc="验证图片"):
            if os.path.exists(info['full_path']):
                valid_images.append(info)
            else:
                print(f"警告: 图片文件不存在: {info['full_path']}")
        
        print(f"有效图片数量: {len(valid_images)}")
        
        # 批量处理图片
        for i in tqdm(range(0, len(valid_images), batch_size), desc="生成embeddings"):
            batch = valid_images[i:i + batch_size]
            batch_images = []
            batch_info = []
            
            # 加载批次图片
            for info in batch:
                try:
                    image = Image.open(info['full_path']).convert('RGB')
                    batch_images.append(image)
                    batch_info.append(info)
                except Exception as e:
                    print(f"警告: 无法加载图片 {info['full_path']}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # 生成多模态embeddings
            try:
                # 1. 图片embeddings
                image_embeddings = self.model.encode(batch_images, convert_to_numpy=True)

                # 2. 图片描述文本embeddings
                descriptions = [info.get('description', '') for info in batch_info]
                text_embeddings = self.model.encode(descriptions, convert_to_numpy=True)

                # 3. 上下文文本embeddings
                contexts = [info.get('chunk_content', '') for info in batch_info]
                context_embeddings = self.model.encode(contexts, convert_to_numpy=True)

                # 保存结果
                for j in range(len(batch_info)):
                    info = batch_info[j].copy()
                    info['image_embedding'] = image_embeddings[j].tolist()
                    info['text_embedding'] = text_embeddings[j].tolist()
                    info['context_embedding'] = context_embeddings[j].tolist()
                    info['embedding_dim'] = len(image_embeddings[j])
                    results.append(info)
                    
            except Exception as e:
                print(f"警告: 批次embedding生成失败: {e}")
                continue
        
        print(f"成功生成 {len(results)} 个图片embeddings")
        return results
    
    def save_embeddings(self, embeddings_data: List[Dict[str, Any]], output_file: str):
        """
        保存embeddings到JSON文件
        Args:
            embeddings_data: 包含embeddings的数据
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 构建最终的JSON结构
        output_data = {
            "metadata": {
                "model": self.model_name,
                "total_images": len(embeddings_data),
                "embedding_dim": embeddings_data[0]['embedding_dim'] if embeddings_data else 0,
                "created_time": datetime.now().isoformat(),
                "description": "多模态CLIP embeddings：图片、描述文本、上下文文本",
                "embedding_types": ["image_embedding", "text_embedding", "context_embedding"]
            },
            "embeddings": embeddings_data
        }
        
        print(f"正在保存embeddings到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 计算文件大小
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"embeddings已保存，文件大小: {file_size:.2f} MB")


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent
    chunks_file = project_root / "outputs" / "output_v1_3_with_chunk" / "all_pdf_enhanced_chunks.json"
    output_file = project_root / "outputs" / "output_v1_5_vlm" / "image_embeddings.json"
    
    # 检查输入文件
    if not chunks_file.exists():
        print(f"错误: 输入文件不存在: {chunks_file}")
        return
    
    try:
        # 创建图片向量化生成器
        generator = ImageEmbeddingGenerator()
        
        # 提取图片信息
        image_info = generator.extract_image_info_from_chunks(str(chunks_file))
        
        if not image_info:
            print("未找到任何图片，退出")
            return
        
        # 生成embeddings
        embeddings_data = generator.generate_embeddings(image_info, batch_size=16)  # 减小批大小以适应CPU
        
        if not embeddings_data:
            print("未能生成任何embeddings，退出")
            return
        
        # 保存结果
        generator.save_embeddings(embeddings_data, str(output_file))
        
        print("图片向量化完成！")
        print(f"输入文件: {chunks_file}")
        print(f"输出文件: {output_file}")
        print(f"处理图片数量: {len(embeddings_data)}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
