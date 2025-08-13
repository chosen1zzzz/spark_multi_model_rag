"""
GLM-4.1V多模态生成器：支持文本+图片的联合推理
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("警告: 需要安装 openai 包")
    OPENAI_AVAILABLE = False


class VLMGenerator:
    """视觉语言模型生成器"""
    
    def __init__(self):
        """初始化GLM-4.1V客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装 openai 包: pip install openai")
        
        # 从环境变量读取配置
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.vl_model = os.getenv('VL_MODEL', 'THUDM/GLM-4.1V-9B-Thinking')
        
        if not self.api_key or not self.base_url:
            raise ValueError("请在.env文件中配置 LOCAL_API_KEY 和 LOCAL_BASE_URL")
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"GLM-4.1V客户端初始化完成")
        print(f"模型: {self.vl_model}")
        print(f"API地址: {self.base_url}")
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        将图片编码为base64字符串
        Args:
            image_path: 图片文件路径
        Returns:
            base64编码的图片字符串，失败返回None
        """
        try:
            if not os.path.exists(image_path):
                print(f"警告: 图片文件不存在: {image_path}")
                return None
            
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"图片编码失败 {image_path}: {e}")
            return None
    
    def build_multimodal_prompt(self, 
                               question: str,
                               text_context: str,
                               image_results: List[Dict[str, Any]],
                               max_images: int = 3) -> List[Dict[str, Any]]:
        """
        构建多模态prompt
        Args:
            question: 用户问题
            text_context: 文本上下文
            image_results: 图片检索结果
            max_images: 最大图片数量
        Returns:
            OpenAI格式的消息列表
        """
        # 系统提示词
        system_prompt = """你是一名高级金融分析助手，具备多模态信息整合能力。
你的核心职责是：基于提供的文本信息和图片信息，提供最精确的分析结果。

重要原则：
1. 优先使用文本信息进行分析
2. 图片信息用于补充和验证文本信息
3. 如果文本和图片信息冲突，以文本信息为准
4. 宁可承认信息不足，也不能产生任何未经验证的内容

严格按照JSON格式输出：
{"answer": "你的答案或'根据现有信息无法回答'", "filename": "最可靠来源的文件名", "page": "对应页码"}

请确保输出内容为合法JSON字符串，不要输出多余内容。"""

        # 构建用户消息
        user_content = []
        
        # 添加文本内容
        user_text = f"""问题：{question}

文本信息：
{text_context}

"""
        
        # 处理图片
        valid_images = []
        project_root = Path(__file__).parent
        
        for i, img_result in enumerate(image_results[:max_images]):
            image_path = img_result.get('image_path', '')
            if not image_path:
                continue
            
            # 构建完整图片路径
            full_image_path = project_root / image_path
            if not full_image_path.exists():
                # 尝试其他可能的路径
                alt_path = project_root / "outputs" / "output_v1_3_with_chunk" / image_path
                if alt_path.exists():
                    full_image_path = alt_path
                else:
                    print(f"警告: 图片文件不存在: {image_path}")
                    continue
            
            # 编码图片
            base64_image = self.encode_image_to_base64(str(full_image_path))
            if base64_image:
                valid_images.append({
                    'path': image_path,
                    'base64': base64_image,
                    'description': img_result.get('description', ''),
                    'file_name': img_result.get('file_name', ''),
                    'page': img_result.get('page', ''),
                    'context': img_result.get('chunk_content', '')
                })
        
        # 添加图片信息到用户文本
        if valid_images:
            user_text += "图片信息：\n"
            for i, img in enumerate(valid_images, 1):
                user_text += f"图片{i}: {img['file_name']} 第{img['page']}页\n"
                if img['description']:
                    user_text += f"描述: {img['description']}\n"
                if img['context']:
                    user_text += f"上下文: {img['context']}\n"
                user_text += "\n"
        
        # 构建消息内容
        user_content.append({
            "type": "text",
            "text": user_text
        })
        
        # 添加图片内容
        for img in valid_images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['base64']}"
                }
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return messages
    
    def generate_multimodal_answer(self, 
                                  question: str,
                                  multimodal_context: Dict[str, Any],
                                  max_images: int = 3,
                                  temperature: float = 0.2,
                                  max_tokens: int = 1024) -> str:
        """
        生成多模态答案
        Args:
            question: 用户问题
            multimodal_context: 多模态检索结果
            max_images: 最大图片数量
            temperature: 生成温度
            max_tokens: 最大token数
        Returns:
            生成的答案字符串
        """
        try:
            # 格式化文本上下文
            from multimodal_retriever import MultiModalRetriever
            
            # 创建临时检索器实例用于格式化（不需要实际初始化）
            text_results = multimodal_context.get("text_results", [])
            image_results = multimodal_context.get("image_results", [])
            
            # 格式化文本上下文
            text_context_parts = []
            for i, result in enumerate(text_results[:5], 1):
                metadata = result.get("metadata", {})
                file_name = metadata.get("file_name", "未知文件")
                page = metadata.get("page", "未知页码")
                content = result.get("content", "")
                
                text_context_parts.append(f"[文本{i}] 文件: {file_name}, 页码: {page}")
                text_context_parts.append(f"内容: {content}")
                text_context_parts.append("")
            
            text_context = "\n".join(text_context_parts)
            
            # 构建多模态prompt
            messages = self.build_multimodal_prompt(
                question=question,
                text_context=text_context,
                image_results=image_results,
                max_images=max_images
            )
            
            print(f"调用GLM-4.1V生成答案...")
            print(f"文本chunks: {len(text_results)}, 图片: {len(image_results)}")
            
            # 调用GLM-4.1V
            completion = self.client.chat.completions.create(
                model=self.vl_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response = completion.choices[0].message.content.strip()
            print(f"GLM-4.1V响应完成")
            
            return response
            
        except Exception as e:
            print(f"GLM-4.1V生成错误: {e}")
            # 返回错误格式的JSON
            error_response = {
                "answer": "根据现有信息无法回答",
                "filename": "系统错误",
                "page": "N/A"
            }
            return json.dumps(error_response, ensure_ascii=False)


class MultiModalRAG:
    """完整的多模态RAG系统"""
    
    def __init__(self, text_rag_system, multimodal_retriever, vlm_generator):
        """
        初始化多模态RAG系统
        Args:
            text_rag_system: 文本RAG系统
            multimodal_retriever: 多模态检索器
            vlm_generator: 视觉语言模型生成器
        """
        self.text_rag = text_rag_system
        self.multimodal_retriever = multimodal_retriever
        self.vlm_generator = vlm_generator
        
        print("多模态RAG系统初始化完成")
    
    def answer_question(self, 
                       question: str,
                       text_top_k: int = 10,
                       image_top_k: int = 5,
                       max_images_for_generation: int = 3) -> Dict[str, Any]:
        """
        回答问题（多模态）
        Args:
            question: 用户问题
            text_top_k: 文本检索top-k
            image_top_k: 图片检索top-k
            max_images_for_generation: 生成时使用的最大图片数
        Returns:
            包含答案和元信息的字典
        """
        print(f"处理多模态问题: {question}")
        
        # 多模态检索
        search_results = self.multimodal_retriever.search_multimodal(
            query=question,
            text_top_k=text_top_k,
            image_top_k=image_top_k
        )
        
        # 生成答案
        answer = self.vlm_generator.generate_multimodal_answer(
            question=question,
            multimodal_context=search_results,
            max_images=max_images_for_generation
        )
        
        return {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "metadata": {
                "text_chunks_used": len(search_results.get("text_results", [])),
                "images_used": len(search_results.get("image_results", [])),
                "model": self.vlm_generator.vl_model
            }
        }


def create_multimodal_rag_system(text_rag_system, 
                                image_embeddings_file: str = None) -> MultiModalRAG:
    """
    创建完整多模态RAG系统的工厂函数
    Args:
        text_rag_system: 现有的文本RAG系统
        image_embeddings_file: 图片embeddings文件路径
    Returns:
        MultiModalRAG实例
    """
    from multimodal_retriever import create_multimodal_retriever
    
    # 创建多模态检索器
    multimodal_retriever = create_multimodal_retriever(
        text_rag_system=text_rag_system,
        image_embeddings_file=image_embeddings_file
    )
    
    # 创建VLM生成器
    vlm_generator = VLMGenerator()
    
    # 创建完整系统
    return MultiModalRAG(
        text_rag_system=text_rag_system,
        multimodal_retriever=multimodal_retriever,
        vlm_generator=vlm_generator
    )


if __name__ == "__main__":
    print("GLM-4.1V多模态生成器模块")
    print("使用方法:")
    print("1. 确保.env中配置了VL_MODEL=THUDM/GLM-4.1V-9B-Thinking")
    print("2. 使用 create_multimodal_rag_system() 创建完整系统")
    print("3. 调用 answer_question() 进行多模态问答")
