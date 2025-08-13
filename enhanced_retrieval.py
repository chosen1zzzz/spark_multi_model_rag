"""
增强检索系统：支持重排和多策略检索
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """重排模型基类"""
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重新排序候选结果"""
        pass


class LocalBGEReranker(BaseReranker):
    """基于本地BGE-reranker-v2-m3的重排模型"""

    def __init__(self, model_name: str = None, device: str = "cuda"):
        # 从环境变量读取模型名称
        if model_name is None:
            model_name = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device)
            print(f"已加载本地BGE重排模型: {model_name} (设备: {device})")
        except ImportError:
            print("警告: sentence_transformers未安装，请运行: pip install sentence_transformers")
            self.model = None
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用BGE重排模型重新排序"""
        if not self.model or not candidates:
            return candidates
        
        try:
            # 准备输入对：(query, candidate_content)
            pairs = []
            for candidate in candidates:
                content = candidate['content']
                # 限制内容长度，避免超过模型限制
                if len(content) > 500:
                    content = content[:500] + "..."
                pairs.append([query, content])
            
            # 计算相关性分数
            scores = self.model.predict(pairs)
            
            # 按分数排序
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 添加重排分数到结果
            reranked_results = []
            for candidate, score in scored_candidates:
                candidate_copy = candidate.copy()
                candidate_copy['rerank_score'] = float(score)
                reranked_results.append(candidate_copy)
            
            return reranked_results

        except Exception as e:
            print(f"本地BGE重排失败: {e}")
            return candidates


class APIReranker(BaseReranker):
    """基于API的重排模型"""

    def __init__(self):
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.model_name = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')

        if not all([self.api_key, self.base_url, self.model_name]):
            raise ValueError('请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、RERANKER_MODEL')

        print(f"已初始化API重排模型: {self.model_name}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用Rerank API进行重排"""
        if not candidates:
            return candidates

        try:
            import requests

            # 准备文档列表
            documents = []
            for candidate in candidates:
                content = candidate['content']
                # 限制内容长度，避免超过API限制
                if len(content) > 500:
                    content = content[:500] + "..."
                documents.append(content)

            # 调用rerank API
            response = requests.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents
                }
            )

            if response.status_code != 200:
                raise Exception(f"Error code: {response.status_code} - {response.text}")

            result = response.json()

            # 解析rerank结果
            reranked_results = []
            if 'results' in result:
                # 按rerank分数排序
                for item in result['results']:
                    idx = item['index']
                    score = item['relevance_score']

                    if 0 <= idx < len(candidates):
                        candidate_copy = candidates[idx].copy()
                        candidate_copy['rerank_score'] = float(score)
                        reranked_results.append(candidate_copy)

                # 按分数降序排序
                reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            else:
                # 如果API返回格式不符合预期，返回原始顺序
                reranked_results = candidates.copy()
                for candidate in reranked_results:
                    candidate['rerank_score'] = 0

            print(f"API重排完成: {len(reranked_results)} 个结果")
            return reranked_results

        except Exception as e:
            print(f"API重排失败: {e}")
            # 返回原始顺序，但添加默认分数
            for candidate in candidates:
                candidate['rerank_score'] = 0
            return candidates


class EnhancedVectorStore:
    """增强的向量存储，支持两阶段检索"""
    
    def __init__(self, reranker: BaseReranker = None):
        self.embeddings = []
        self.chunks = []
        self.reranker = reranker
        print(f"初始化增强向量存储，重排器: {type(reranker).__name__ if reranker else 'None'}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """添加chunks和对应的embeddings"""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        print(f"已添加 {len(chunks)} 个chunks，总计 {len(self.chunks)} 个")
    
    def search_with_rerank(self, query: str, query_embedding: List[float], 
                          recall_k: int = 20, final_k: int = 5) -> List[Dict[str, Any]]:
        """两阶段检索：向量召回 + 重排"""
        
        # 阶段1：向量召回更多候选
        candidates = self._vector_recall(query_embedding, recall_k)
        print(f"向量召回阶段: 获得 {len(candidates)} 个候选")
        
        if not self.reranker or len(candidates) <= final_k:
            print("跳过重排阶段")
            return candidates[:final_k]
        
        # 阶段2：重排精选
        print(f"重排阶段: 从 {len(candidates)} 个候选中选择 {final_k} 个")
        reranked_candidates = self.reranker.rerank(query, candidates)
        
        return reranked_candidates[:final_k]
    
    def search_by_type(self, query: str, query_embedding: List[float], 
                      content_types: List[str] = None, 
                      recall_k: int = 20, final_k: int = 5) -> List[Dict[str, Any]]:
        """按内容类型筛选检索"""
        
        # 先按类型筛选
        if content_types:
            filtered_indices = []
            for i, chunk in enumerate(self.chunks):
                chunk_type = chunk.get('type', 'text')
                if chunk_type in content_types:
                    filtered_indices.append(i)
            print(f"类型筛选: {content_types} -> {len(filtered_indices)} 个候选")
        else:
            filtered_indices = list(range(len(self.chunks)))
        
        if not filtered_indices:
            return []
        
        # 在筛选后的chunks中进行向量检索
        filtered_embeddings = np.array([self.embeddings[i] for i in filtered_indices])
        filtered_chunks = [self.chunks[i] for i in filtered_indices]
        
        query_emb = np.array(query_embedding)
        sims = filtered_embeddings @ query_emb / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        # 获取top candidates
        top_indices = sims.argsort()[::-1][:min(recall_k, len(filtered_chunks))]
        candidates = []
        for idx in top_indices:
            chunk = filtered_chunks[idx].copy()
            chunk['vector_score'] = float(sims[idx])
            candidates.append(chunk)
        
        # 重排
        if self.reranker and len(candidates) > final_k:
            candidates = self.reranker.rerank(query, candidates)
        
        return candidates[:final_k]
    
    def _vector_recall(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """向量召回阶段"""
        if not self.embeddings:
            return []
        
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (
            np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        idxs = sims.argsort()[::-1][:top_k]
        
        results = []
        for i in idxs:
            chunk = self.chunks[i].copy()
            chunk['vector_score'] = float(sims[i])
            results.append(chunk)
        
        return results



def create_enhanced_rag_system(chunk_json_path: str,
                             use_reranker: bool = True,
                             use_local_reranker_model: bool = False,
                             reranker_device: str = "auto"):
    """创建增强RAG系统的工厂函数"""

    # 自动选择设备
    if reranker_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                # 检查显存是否充足（至少需要2GB空闲显存）
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_memory > 2 * 1024**3:  # 2GB
                    reranker_device = "cuda"
                    print(f"检测到充足GPU显存 ({free_memory/1024**3:.1f}GB空闲)，使用GPU运行重排模型")
                else:
                    reranker_device = "cpu"
                    print(f"GPU显存不足 ({free_memory/1024**3:.1f}GB空闲)，使用CPU运行重排模型")
            else:
                reranker_device = "cpu"
                print("未检测到GPU，使用CPU运行重排模型")
        except ImportError:
            reranker_device = "cpu"
            print("PyTorch未安装，使用CPU运行重排模型")
        except Exception as e:
            reranker_device = "cpu"
            print(f"设备检测失败: {e}，使用CPU运行重排模型")

    # 初始化重排模型
    reranker = None
    if use_reranker:
        try:
            if use_local_reranker_model:
                reranker = LocalBGEReranker(device=reranker_device)
            else:
                reranker = APIReranker()
        except Exception as e:
            print(f"重排模型初始化失败: {e}")
            print("将使用无重排的基础检索")
    
    # 创建增强RAG系统
    from rag_from_page_chunks import PageChunkLoader, EmbeddingModel
    
    class EnhancedRAG:
        def __init__(self, chunk_json_path: str, reranker: BaseReranker = None):
            self.loader = PageChunkLoader(chunk_json_path)
            self.embedding_model = EmbeddingModel()
            self.vector_store = EnhancedVectorStore(reranker)
            # 禁用查询分类器，使用通用重排策略
            self.use_local_llm = False
        
        def setup(self):
            """构建增强向量库"""
            chunks = self.loader.load_chunks()
            print(f"加载了 {len(chunks)} 个chunks")
            
            print("生成embeddings...")
            embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
            
            print("构建增强向量库...")
            self.vector_store.add_chunks(chunks, embeddings)
            print("增强RAG系统构建完成！")
        
        def generate_answer_enhanced(self, question: str) -> Dict[str, Any]:
            """增强的问答生成"""

            print(f"问题: {question}")
            print("使用通用重排检索策略")

            # 生成查询embedding
            q_emb = self.embedding_model.embed_text(question)

            # 使用通用重排检索
            chunks = self.vector_store.search_with_rerank(
                question, q_emb, recall_k=20, final_k=5
            )
            
            # 构建上下文
            context = self._build_enhanced_context(chunks)
            
            # 生成回答（使用原有的API调用逻辑）
            response_text = self._generate_with_api(question, context)
            
            # 解析结果
            try:
                import json
                result = json.loads(response_text)
                result['retrieval_info'] = {
                    'strategy': 'universal_rerank',
                    'chunks_count': len(chunks),
                    'chunk_types': [c.get('type', 'text') for c in chunks],
                    'rerank_scores': [c.get('rerank_score', 0) for c in chunks],
                    'vector_scores': [c.get('vector_score', 0) for c in chunks]
                }
                result['retrieval_chunks'] = chunks
                return result
            except:
                return {
                    "answer": response_text,
                    "filename": chunks[0]['metadata']['file_name'] if chunks else "",
                    "page": chunks[0]['metadata']['page'] if chunks else "",
                    "retrieval_info": {
                        'strategy': 'universal_rerank',
                        'chunks_count': len(chunks)
                    },
                    "retrieval_chunks": chunks
                }
        
        def _build_enhanced_context(self, chunks: List[Dict[str, Any]]) -> str:
            """构建增强上下文"""
            context_parts = []
            for i, chunk in enumerate(chunks):
                chunk_type = chunk.get('type', 'text')
                rerank_score = chunk.get('rerank_score', 0)
                vector_score = chunk.get('vector_score', 0)
                
                # 构建前缀信息
                prefix = f"[排名{i+1}][{chunk_type.upper()}]"
                if rerank_score > 0:
                    prefix += f"[重排分数:{rerank_score:.3f}]"
                if vector_score > 0:
                    prefix += f"[向量分数:{vector_score:.3f}]"
                
                context_parts.append(
                    f"{prefix}[文件]{chunk['metadata']['file_name']} "
                    f"[页码]{chunk['metadata']['page']}\n{chunk['content']}"
                )
            
            return "\n\n".join(context_parts)
        
        def _generate_with_api(self, question: str, context: str) -> str:
            """使用API生成回答"""
            from openai import OpenAI
            
            api_key = os.getenv('LOCAL_API_KEY')
            base_url = os.getenv('LOCAL_BASE_URL')
            model = os.getenv('LOCAL_TEXT_MODEL')
            
            if not all([api_key, base_url, model]):
                raise ValueError('请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL')
            
            prompt = f"""你是一名高级金融分析助手，具备多源信息整合能力。
你的核心职责是：利用重排序后的高质量信息源，提供最精确的分析结果。
重要原则：宁可承认信息不足，也不能产生任何未经验证的内容。

信息可信度评估标准：
- 重排分数 ≥ 0.7：高可信度信息，可直接使用
- 重排分数 0.3-0.7：中等可信度，需要交叉验证
- 重排分数 < 0.3：低可信度，谨慎使用
- 排名前2的chunk如果内容一致，可信度提升

回答决策流程：
1. 优先使用重排分数最高的信息
2. 检查前3个chunk的信息一致性
3. 高可信度信息 ≥ 1个 + 中等可信度信息 ≥ 1个且一致 → 可以回答
4. 仅有低可信度信息或信息冲突 → "根据现有信息无法回答"
5. 必须引用最可靠的信息来源

严格按照JSON格式输出：
{{"answer": "你的答案或'根据现有信息无法回答'", "filename": "最可靠来源的文件名", "page": "对应页码"}}

检索内容：
{context}

问题：{question}

请确保输出内容为合法JSON字符串，不要输出多余内容。"""
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一名专业的金融分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            
            return completion.choices[0].message.content.strip()

        def setup_multimodal(self):
            """设置多模态功能"""
            try:
                from pathlib import Path
                import os

                # 检查图片embeddings文件
                project_root = Path(__file__).parent
                image_embeddings_file = project_root / "outputs" / "output_v1_5_vlm" / "image_embeddings.json"

                if not image_embeddings_file.exists():
                    print("图片embeddings文件不存在，将只使用文本检索")
                    return False

                # 创建多模态检索器
                from multimodal_retriever import create_multimodal_retriever
                self.multimodal_retriever = create_multimodal_retriever(
                    text_rag_system=self,
                    image_embeddings_file=str(image_embeddings_file)
                )

                # 创建VLM生成器
                from vlm_generator import VLMGenerator
                self.vlm_generator = VLMGenerator()

                print("多模态功能已启用")
                return True

            except Exception as e:
                print(f"多模态功能初始化失败: {e}")
                print("将使用文本模式")
                return False

        def generate_answer_multimodal(self, question: str, text_top_k: int = 10, image_top_k: int = 5) -> Dict[str, Any]:
            """多模态问答生成"""
            try:
                if not hasattr(self, 'multimodal_retriever') or not hasattr(self, 'vlm_generator'):
                    print("多模态功能未初始化，回退到文本模式")
                    return self.generate_answer_enhanced(question)

                print(f"多模态问题: {question}")

                # 使用多模态检索
                search_results = self.multimodal_retriever.search_multimodal(
                    query=question,
                    text_top_k=text_top_k,
                    image_top_k=image_top_k
                )

                # 使用GLM-4.1V生成答案
                answer = self.vlm_generator.generate_multimodal_answer(
                    question=question,
                    multimodal_context=search_results,
                    max_images=3
                )

                # 解析结果
                try:
                    import json
                    result = json.loads(answer)
                    result['retrieval_info'] = {
                        'strategy': 'multimodal',
                        'text_chunks': len(search_results.get("text_results", [])),
                        'image_chunks': len(search_results.get("image_results", [])),
                        'model': 'GLM-4.1V'
                    }
                    result['retrieval_chunks'] = {
                        'text_results': search_results.get("text_results", []),
                        'image_results': search_results.get("image_results", [])
                    }
                    return result
                except json.JSONDecodeError:
                    return {
                        "answer": answer,
                        "filename": "multimodal_result",
                        "page": "unknown",
                        "retrieval_info": {
                            'strategy': 'multimodal',
                            'text_chunks': len(search_results.get("text_results", [])),
                            'image_chunks': len(search_results.get("image_results", []))
                        }
                    }

            except Exception as e:
                print(f"多模态生成失败，回退到文本模式: {e}")
                return self.generate_answer_enhanced(question)

    return EnhancedRAG(chunk_json_path, reranker)
