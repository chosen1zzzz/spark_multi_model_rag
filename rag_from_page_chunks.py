import json
import os

import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import concurrent.futures
import random

from get_text_embedding import get_text_embedding

from dotenv import load_dotenv
from openai import OpenAI
# 统一加载项目根目录的.env
load_dotenv()

class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)




class EmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')
        self.batch_size = batch_size
        if not self.api_key or not self.base_url:
            raise ValueError('请在.env中配置LOCAL_API_KEY和LOCAL_BASE_URL')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]

class SimpleRAG:
    def __init__(self, chunk_json_path: str, batch_size: int = 32,
                 llm_model_path: str = "/mnt/workspace/AISumerCamp_multiModal_RAG/models/Qwen3-8B"):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()

        # 使用本地Qwen模型
        self.use_local_llm = False
        if self.use_local_llm:
            self._load_local_llm(llm_model_path)


    def setup(self):
        print("加载所有页chunk...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 个chunk")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        print("RAG向量库构建完成！")
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {
            "question": question,
            "chunks": results
        }

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        检索+大模型生成式回答，返回结构化结果
        """
        q_emb = self.embedding_model.embed_text(question)
        chunks = self.vector_store.search(q_emb, top_k)

        context = "\n".join([
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}" for c in chunks
        ])

        prompt = f"""你是一名严谨的金融分析助手，专门处理财报和研究报告分析。
你的核心职责是：基于检索到的文档内容提供准确答案，绝不进行推测或编造。

信息可信度评估标准：
- 单个chunk内容完整且直接回答问题 = 可信
- 多个chunk内容一致 = 高度可信
- 多个chunk内容冲突 = 需要选择最相关的
- 所有chunk都不直接相关 = 不可信

回答决策流程：
1. 检查是否有直接相关的信息
2. 如果有冲突，选择最相关且完整的信息
3. 如果信息完整且相关 → 提供答案
4. 如果信息模糊或不相关 → 回答"根据现有信息无法回答"

严格按照JSON格式输出：
{{"answer": "你的答案或'根据现有信息无法回答'", "filename": "最可靠来源的文件名", "page": "对应页码"}}

检索内容：
{context}

问题：{question}

请确保输出内容为合法JSON字符串，不要输出多余内容。"""

        # prompt = (
        #     f"你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
        #     f"请严格按照如下JSON格式输出：\n"
        #     f'{{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}}'"\n"
        #     f"检索内容：\n{context}\n\n问题：{question}\n"
        #     f"请确保输出内容为合法JSON字符串，不要输出多余内容。"
        # )

        if self.use_local_llm:
            # 使用本地模型
            raw = self._generate_with_local_llm(prompt)
        else:
            # 使用API调用（原有逻辑）
            qwen_api_key = os.getenv('LOCAL_API_KEY')
            qwen_base_url = os.getenv('LOCAL_BASE_URL')
            qwen_model = os.getenv('LOCAL_TEXT_MODEL')
            if not qwen_api_key or not qwen_base_url or not qwen_model:
                raise ValueError('请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL')

            client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
            completion = client.chat.completions.create(
                model=qwen_model,
                messages=[
                    {"role": "system", "content": "你是一名专业的金融分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            raw = completion.choices[0].message.content.strip()
        import json as pyjson
        from extract_json_array import extract_json_array
        # 用 extract_json_array 提取 JSON 对象
        json_str = extract_json_array(raw, mode='objects')
        if json_str:
            try:
                arr = pyjson.loads(json_str)
                # 只取第一个对象
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    answer = raw
                    filename = chunks[0]['metadata']['file_name'] if chunks else ''
                    page = chunks[0]['metadata']['page'] if chunks else ''
            except Exception:
                answer = raw
                filename = chunks[0]['metadata']['file_name'] if chunks else ''
                page = chunks[0]['metadata']['page'] if chunks else ''
        else:
            answer = raw
            filename = chunks[0]['metadata']['file_name'] if chunks else ''
            page = chunks[0]['metadata']['page'] if chunks else ''
        # 结构化输出
        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": chunks
        }

    def _load_local_llm(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"正在加载本地LLM模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("本地LLM模型加载完成")

    def _generate_with_local_llm(self, prompt: str) -> str:
        """使用本地LLM生成回答"""
        import torch

        messages = [
            {"role": "system", "content": "你是一名专业的金融分析助手。"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == '__main__':
    # 路径可根据实际情况调整
    # 使用增强分块文件
    chunk_json_path = "./outputs/output_v1_3_with_chunk/all_pdf_enhanced_chunks.json"

    # 选择RAG系统类型
    use_enhanced_rag = True  # 设置为True使用增强检索，False使用基础检索
    use_multimodal = True   # 设置为True使用多模态RAG，False使用纯文本RAG

    if use_enhanced_rag:
        print("=== 使用增强检索系统 ===")
        from enhanced_retrieval import create_enhanced_rag_system

        # 创建增强RAG系统
        rag = create_enhanced_rag_system(
            chunk_json_path=chunk_json_path,
            use_reranker=True,  # 是否使用重排模型
            use_local_reranker_model=False,  # 使用API重排
            reranker_device="auto"  # 自动选择设备（仅本地模型时有效）
        )
        rag.setup()

        # 如果启用多模态，设置多模态功能
        if use_multimodal:
            print("=== 启用多模态功能 ===")
            multimodal_enabled = rag.setup_multimodal()
            if multimodal_enabled:
                print("多模态功能已启用")
            else:
                print("多模态功能启用失败，将使用纯文本模式")
                use_multimodal = False

    else:
        print("=== 使用基础检索系统 ===")
        rag = SimpleRAG(chunk_json_path)
        rag.setup()

    # 控制测试时读取的题目数量，默认只随机抽取10个，实际跑全部时设为None
    TEST_SAMPLE_NUM = None  # 设置为None则全部跑
    FILL_UNANSWERED = True  # 未回答的也输出默认内容

    # 批量评测脚本：读取测试集，检索+大模型生成，输出结构化结果
    test_path = "./datas/test.json"
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # 记录所有原始索引
        all_indices = list(range(len(test_data)))
        # 随机抽取部分题目用于测试
        selected_indices = all_indices
        if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0:
            if len(test_data) > TEST_SAMPLE_NUM:
                selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))

        def process_one(idx):
            import time
            from openai import RateLimitError, APIError

            item = test_data[idx]
            question = item['question']
            tqdm.write(f"[{selected_indices.index(idx)+1}/{len(selected_indices)}] 正在处理: {question[:30]}...")

            # 重试机制
            max_retries = 3
            base_delay = 1  # 基础延迟时间（秒）

            for attempt in range(max_retries):
                try:
                    # 根据配置选择使用多模态或纯文本RAG
                    if use_multimodal and hasattr(rag, 'generate_answer_multimodal'):
                        result = rag.generate_answer_multimodal(question)
                    else:
                        result = rag.generate_answer_enhanced(question)
                    return idx, result

                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        # 指数退避：1秒、2秒、4秒
                        delay = base_delay * (2 ** attempt)
                        tqdm.write(f"[{idx}] 速率限制，第{attempt+1}次重试，等待{delay}秒...")
                        time.sleep(delay)
                    else:
                        tqdm.write(f"[{idx}] 重试{max_retries}次后仍失败: {e}")
                        return idx, {"error": f"RateLimitError after {max_retries} retries: {str(e)}"}

                except APIError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        tqdm.write(f"[{idx}] API错误，第{attempt+1}次重试，等待{delay}秒...")
                        time.sleep(delay)
                    else:
                        tqdm.write(f"[{idx}] API错误重试{max_retries}次后仍失败: {e}")
                        return idx, {"error": f"APIError after {max_retries} retries: {str(e)}"}

                except Exception as e:
                    tqdm.write(f"[{idx}] 未知错误: {e}")
                    return idx, {"error": f"Unknown error: {str(e)}"}

            return idx, {"error": "Max retries exceeded"}

        results = []
        if selected_indices:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(tqdm(executor.map(process_one, selected_indices), total=len(selected_indices), desc='并发批量生成'))

        # 先输出一份未过滤的原始结果（含 idx）
        output_dir = "./outputs/output_v1_5_vlm" if use_multimodal else "./outputs/output_v1_4_reranker"
        os.makedirs(output_dir, exist_ok=True)
        raw_out_path = f"{output_dir}/rag_top1_pred_raw.json"
        with open(raw_out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'已输出原始未过滤结果到: {raw_out_path}')

        # 只保留结果部分，并去除 retrieval_chunks 字段
        idx2result = {idx: {k: v for k, v in r.items() if k != 'retrieval_chunks'} for idx, r in results}
        filtered_results = []
        for idx, item in enumerate(test_data):
            if idx in idx2result:
                filtered_results.append(idx2result[idx])
            elif FILL_UNANSWERED:
                # 未被回答的，补默认内容
                filtered_results.append({
                    "question": item.get("question", ""),
                    "answer": "",
                    "filename": "",
                    "page": "",
                })
        # 输出结构化结果到json
        out_path = f"{output_dir}/rag_top1_pred.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        print(f'已输出结构化检索+大模型生成结果到: {out_path}')
    else:
        print(f'out_path 路径不存在')
    
        