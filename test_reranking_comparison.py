"""
重排效果对比测试脚本
"""
import json
import time
from typing import List, Dict, Any


def test_retrieval_comparison():
    """对比基础检索和重排检索的效果"""
    
    # 测试问题集
    test_questions = [
        {
            "question": "中恒电气2024年营业收入是多少？",
            "keywords": ["营业收入", "2024", "亿元"]
        },
        {
            "question": "HVDC技术相比传统UPS有什么优势？",
            "keywords": ["HVDC", "优势", "效率", "稳定性"]
        },
        {
            "question": "巴拿马电源方案的核心特点是什么？",
            "keywords": ["巴拿马", "电源", "特点", "优点"]
        },
        {
            "question": "公司的股权结构如何？",
            "keywords": ["股权", "结构", "持股"]
        },
        {
            "question": "2025年预计净利润增长率是多少？",
            "keywords": ["2025", "净利润", "增长率", "%"]
        }
    ]
    
    chunk_json_path = "./outputs/output_v1_3_with_chunk/all_pdf_enhanced_chunks.json"
    
    print("=== 重排检索效果对比测试 ===\n")
    
    # 测试基础检索
    print("1. 基础检索测试")
    print("-" * 50)
    
    from rag_from_page_chunks import SimpleRAG
    basic_rag = SimpleRAG(chunk_json_path)
    basic_rag.setup()

    basic_results = []
    for i, test_case in enumerate(test_questions):
        question = test_case["question"]
        print(f"问题 {i+1}: {question}")

        start_time = time.time()
        # 使用基础检索
        q_emb = basic_rag.embedding_model.embed_text(question)
        chunks = basic_rag.vector_store.search(q_emb, top_k=5)
        end_time = time.time()
        print(f"  检索时间: {end_time - start_time:.3f}s")
        print(f"  检索到 {len(chunks)} 个chunks")
        
        # 分析chunk类型分布
        chunk_types = [chunk.get('type', 'text') for chunk in chunks]
        type_counts = {}
        for t in chunk_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Chunk类型: {type_counts}")
        
        # 检查关键词匹配
        keyword_matches = 0
        for chunk in chunks:
            content = chunk['content'].lower()
            for keyword in test_case["keywords"]:
                if keyword.lower() in content:
                    keyword_matches += 1
                    break
        
        print(f"  关键词匹配: {keyword_matches}/{len(chunks)}")
        
        basic_results.append({
            "question": question,
            "chunks": chunks,
            "keyword_matches": keyword_matches,
            "response_time": end_time - start_time,
            "chunk_types": type_counts
        })
        print()
    
    # 测试增强检索
    print("\n2. 增强检索测试（带重排）")
    print("-" * 50)
    
    try:
        from enhanced_retrieval import create_enhanced_rag_system
        
        enhanced_rag = create_enhanced_rag_system(
            chunk_json_path=chunk_json_path,
            use_reranker=True,
            use_local_reranker_model=False,
            reranker_device="auto"
        )
        enhanced_rag.setup()
        
        enhanced_results = []
        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            print(f"问题 {i+1}: {question}")
            
            start_time = time.time()
            
            # 使用增强检索
            q_emb = enhanced_rag.embedding_model.embed_text(question)
            chunks = enhanced_rag.vector_store.search_with_rerank(
                question, q_emb, recall_k=20, final_k=5
            )
            
            end_time = time.time()
            
            print(f"  检索时间: {end_time - start_time:.3f}s")
            print(f"  检索到 {len(chunks)} 个chunks")
            
            # 分析chunk类型分布
            chunk_types = [chunk.get('type', 'text') for chunk in chunks]
            type_counts = {}
            for t in chunk_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  Chunk类型: {type_counts}")
            
            # 显示重排分数
            rerank_scores = [chunk.get('rerank_score', 0) for chunk in chunks]
            if any(s > 0 for s in rerank_scores):
                print(f"  重排分数: {[f'{s:.3f}' for s in rerank_scores]}")
            
            # 检查关键词匹配
            keyword_matches = 0
            for chunk in chunks:
                content = chunk['content'].lower()
                for keyword in test_case["keywords"]:
                    if keyword.lower() in content:
                        keyword_matches += 1
                        break
            
            print(f"  关键词匹配: {keyword_matches}/{len(chunks)}")
            
            enhanced_results.append({
                "question": question,
                "chunks": chunks,
                "keyword_matches": keyword_matches,
                "response_time": end_time - start_time,
                "chunk_types": type_counts,
                "rerank_scores": rerank_scores
            })
            print()
        
        # 对比分析
        print("\n3. 对比分析")
        print("-" * 50)
        
        total_basic_matches = sum(r["keyword_matches"] for r in basic_results)
        total_enhanced_matches = sum(r["keyword_matches"] for r in enhanced_results)
        total_basic_time = sum(r["response_time"] for r in basic_results)
        total_enhanced_time = sum(r["response_time"] for r in enhanced_results)
        
        print(f"关键词匹配率:")
        print(f"  基础检索: {total_basic_matches}/{len(test_questions)*5} = {total_basic_matches/(len(test_questions)*5)*100:.1f}%")
        print(f"  增强检索: {total_enhanced_matches}/{len(test_questions)*5} = {total_enhanced_matches/(len(test_questions)*5)*100:.1f}%")
        print(f"  提升: {((total_enhanced_matches-total_basic_matches)/(len(test_questions)*5)*100):+.1f}%")
        
        print(f"\n平均响应时间:")
        print(f"  基础检索: {total_basic_time/len(test_questions):.3f}s")
        print(f"  增强检索: {total_enhanced_time/len(test_questions):.3f}s")
        print(f"  时间增加: {((total_enhanced_time-total_basic_time)/total_basic_time*100):+.1f}%")
        
        # 详细对比
        print(f"\n详细对比:")
        for i, (basic, enhanced) in enumerate(zip(basic_results, enhanced_results)):
            print(f"问题 {i+1}: {basic['question'][:30]}...")
            print(f"  基础检索匹配: {basic['keyword_matches']}/5")
            print(f"  增强检索匹配: {enhanced['keyword_matches']}/5")
            
            if 'rerank_scores' in enhanced and enhanced['rerank_scores']:
                avg_rerank = sum(enhanced['rerank_scores']) / len(enhanced['rerank_scores'])
                print(f"  平均重排分数: {avg_rerank:.3f}")
            print()
        
    except ImportError as e:
        print(f"增强检索测试失败: {e}")
        print("请安装依赖: pip install sentence_transformers")
    except Exception as e:
        print(f"增强检索测试出错: {e}")


def analyze_chunk_distribution():
    """分析chunk类型分布"""
    chunk_json_path = "./outputs/output_v1_3_with_chunk/all_pdf_enhanced_chunks.json"
    
    print("=== Chunk类型分布分析 ===")
    
    try:
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"总chunk数量: {len(chunks)}")
        
        # 统计类型分布
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print("\nChunk类型分布:")
        for chunk_type, count in sorted(type_counts.items()):
            percentage = count / len(chunks) * 100
            print(f"  {chunk_type}: {count} ({percentage:.1f}%)")
        
        # 分析内容长度分布
        print("\n内容长度分布:")
        length_ranges = {
            "0-200": 0,
            "200-500": 0, 
            "500-1000": 0,
            "1000+": 0
        }
        
        for chunk in chunks:
            length = len(chunk['content'])
            if length <= 200:
                length_ranges["0-200"] += 1
            elif length <= 500:
                length_ranges["200-500"] += 1
            elif length <= 1000:
                length_ranges["500-1000"] += 1
            else:
                length_ranges["1000+"] += 1
        
        for range_name, count in length_ranges.items():
            percentage = count / len(chunks) * 100
            print(f"  {range_name}字符: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"分析失败: {e}")


if __name__ == "__main__":
    # 分析chunk分布
    analyze_chunk_distribution()
    
    print("\n" + "="*60 + "\n")
    
    # 对比检索效果
    test_retrieval_comparison()
