"""
多模态检索器：支持文本+图片的联合检索
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    print("警告: 需要安装 sentence-transformers")
    CLIP_AVAILABLE = False


class MultiModalRetriever:
    """多模态检索器"""
    
    def __init__(self, 
                 text_rag_system,  # 现有的文本RAG系统
                 image_embeddings_file: str,
                 clip_model_name: str = "clip-ViT-B-32"):
        """
        初始化多模态检索器
        Args:
            text_rag_system: 现有的文本RAG系统（EnhancedRAG实例）
            image_embeddings_file: 图片embeddings文件路径
            clip_model_name: CLIP模型名称
        """
        self.text_rag = text_rag_system
        self.image_embeddings_file = image_embeddings_file
        
        # 加载CLIP模型用于查询向量化
        if CLIP_AVAILABLE:
            print(f"加载CLIP模型用于查询: {clip_model_name}")
            self.clip_model = SentenceTransformer(clip_model_name)
        else:
            self.clip_model = None
            print("警告: CLIP模型不可用，将跳过图片检索")
        
        # 加载图片embeddings
        self.image_data = self._load_image_embeddings()
        
    def _load_image_embeddings(self) -> Dict[str, Any]:
        """加载图片embeddings数据"""
        if not os.path.exists(self.image_embeddings_file):
            print(f"警告: 图片embeddings文件不存在: {self.image_embeddings_file}")
            return {"metadata": {}, "embeddings": []}
        
        print(f"加载图片embeddings: {self.image_embeddings_file}")
        with open(self.image_embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将embeddings转换为numpy数组以提高计算效率
        for item in data.get("embeddings", []):
            if "embedding" in item:
                item["embedding"] = np.array(item["embedding"])
        
        print(f"加载了 {len(data.get('embeddings', []))} 个图片embeddings")
        return data
    
    def _compute_similarity(self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]) -> List[float]:
        """计算余弦相似度"""
        if len(candidate_embeddings) == 0:
            return []
        
        # 归一化查询向量
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # 计算相似度
        similarities = []
        for candidate in candidate_embeddings:
            candidate_norm = candidate / np.linalg.norm(candidate)
            similarity = np.dot(query_norm, candidate_norm)
            similarities.append(float(similarity))
        
        return similarities
    
    def search_images(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        基于文本查询检索相关图片
        Args:
            query: 查询文本
            top_k: 返回top-k结果
        Returns:
            相关图片列表
        """
        if not self.clip_model or not self.image_data.get("embeddings"):
            return []
        
        try:
            # 使用CLIP对查询文本进行编码
            query_embedding = self.clip_model.encode([query], convert_to_numpy=True)[0]
            
            # 获取所有图片embeddings
            image_embeddings = [item["embedding"] for item in self.image_data["embeddings"]]
            
            # 计算相似度
            similarities = self._compute_similarity(query_embedding, image_embeddings)
            
            # 排序并获取top-k
            indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            indexed_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 构建结果
            results = []
            for i, (idx, similarity) in enumerate(indexed_similarities[:top_k]):
                image_info = self.image_data["embeddings"][idx].copy()
                image_info["similarity_score"] = similarity
                image_info["rank"] = i + 1
                # 移除embedding以减少内存占用
                if "embedding" in image_info:
                    del image_info["embedding"]
                results.append(image_info)
            
            return results
            
        except Exception as e:
            print(f"图片检索错误: {e}")
            return []
    
    def search_multimodal(self, 
                         query: str, 
                         text_top_k: int = 10, 
                         image_top_k: int = 5,
                         text_weight: float = 0.7,
                         image_weight: float = 0.3) -> Dict[str, Any]:
        """
        多模态联合检索
        Args:
            query: 查询文本
            text_top_k: 文本检索top-k
            image_top_k: 图片检索top-k
            text_weight: 文本结果权重
            image_weight: 图片结果权重
        Returns:
            包含文本和图片结果的字典
        """
        print(f"执行多模态检索: {query}")
        
        # 文本检索
        print("执行文本检索...")
        text_results = self.text_rag.search_with_rerank(query, recall_k=text_top_k*2, final_k=text_top_k)
        
        # 图片检索
        print("执行图片检索...")
        image_results = self.search_images(query, top_k=image_top_k)
        
        # 为结果添加权重分数
        for result in text_results:
            result["modality"] = "text"
            result["weighted_score"] = result.get("rerank_score", result.get("similarity", 0.5)) * text_weight
        
        for result in image_results:
            result["modality"] = "image"
            result["weighted_score"] = result["similarity_score"] * image_weight
        
        return {
            "query": query,
            "text_results": text_results,
            "image_results": image_results,
            "total_text": len(text_results),
            "total_images": len(image_results)
        }
    
    def format_multimodal_context(self, search_results: Dict[str, Any], max_text_chunks: int = 5, max_images: int = 3) -> str:
        """
        格式化多模态检索结果为上下文字符串
        Args:
            search_results: 多模态检索结果
            max_text_chunks: 最大文本chunk数量
            max_images: 最大图片数量
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 添加文本内容
        text_results = search_results.get("text_results", [])[:max_text_chunks]
        if text_results:
            context_parts.append("=== 相关文本信息 ===")
            for i, result in enumerate(text_results, 1):
                metadata = result.get("metadata", {})
                file_name = metadata.get("file_name", "未知文件")
                page = metadata.get("page", "未知页码")
                content = result.get("content", "")
                
                context_parts.append(f"[文本{i}] 文件: {file_name}, 页码: {page}")
                context_parts.append(f"内容: {content}")
                context_parts.append("")
        
        # 添加图片信息
        image_results = search_results.get("image_results", [])[:max_images]
        if image_results:
            context_parts.append("=== 相关图片信息 ===")
            for i, result in enumerate(image_results, 1):
                file_name = result.get("file_name", "未知文件")
                page = result.get("page", "未知页码")
                description = result.get("description", "")
                image_path = result.get("image_path", "")
                chunk_content = result.get("chunk_content", "")
                
                context_parts.append(f"[图片{i}] 文件: {file_name}, 页码: {page}")
                context_parts.append(f"图片路径: {image_path}")
                if description:
                    context_parts.append(f"图片描述: {description}")
                if chunk_content:
                    context_parts.append(f"上下文: {chunk_content}")
                context_parts.append("")
        
        return "\n".join(context_parts)


def create_multimodal_retriever(text_rag_system, 
                               image_embeddings_file: str = None,
                               clip_model_name: str = "clip-ViT-B-32") -> MultiModalRetriever:
    """
    创建多模态检索器的工厂函数
    Args:
        text_rag_system: 现有的文本RAG系统
        image_embeddings_file: 图片embeddings文件路径
        clip_model_name: CLIP模型名称
    Returns:
        MultiModalRetriever实例
    """
    if image_embeddings_file is None:
        # 默认路径
        project_root = Path(__file__).parent
        image_embeddings_file = project_root / "outputs" / "output_v1_5_vlm" / "image_embeddings.json"
    
    return MultiModalRetriever(
        text_rag_system=text_rag_system,
        image_embeddings_file=str(image_embeddings_file),
        clip_model_name=clip_model_name
    )


if __name__ == "__main__":
    # 测试代码
    print("多模态检索器模块")
    print("使用方法:")
    print("1. 首先运行 generate_image_embeddings.py 生成图片embeddings")
    print("2. 然后使用 create_multimodal_retriever() 创建检索器")
    print("3. 调用 search_multimodal() 进行多模态检索")
