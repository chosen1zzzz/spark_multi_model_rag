import json
from pathlib import Path
from typing import List, Dict, Any
from mineru_advanced_chunker import MinerUAdvancedChunker, handle_cross_page_content


def process_mineru_to_enhanced_chunks(input_json_path: str, output_json_path: str, 
                                     chunk_size: int = 512, overlap_size: int = 50):
    """处理MinerU输出为增强分块
    
    Args:
        input_json_path: MinerU输出的JSON文件路径 (如 all_pdf_page_chunks_mineru.json)
        output_json_path: 增强分块后的输出文件路径
        chunk_size: 目标chunk大小
        overlap_size: 重叠大小
    """
    print(f"开始处理MinerU输出文件: {input_json_path}")
    
    # 加载原始数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        original_chunks = json.load(f)
    
    print(f"加载了 {len(original_chunks)} 个原始页面chunks")
    
    # 初始化分块器
    chunker = MinerUAdvancedChunker(
        chunk_size=chunk_size, 
        overlap_size=overlap_size
    )
    
    enhanced_chunks = []
    
    # 对每个页面进行增强分块
    for i, page_chunk in enumerate(original_chunks):
        content = page_chunk['content']
        metadata = page_chunk['metadata']
        
        print(f"处理第 {i+1}/{len(original_chunks)} 页: {metadata.get('file_name', 'unknown')}")
        
        # 智能分块
        page_enhanced_chunks = chunker.chunk_mineru_content(content, metadata)
        enhanced_chunks.extend(page_enhanced_chunks)
    
    print(f"初步分块完成，生成了 {len(enhanced_chunks)} 个chunks")
    
    # 处理跨页内容
    print("处理跨页内容...")
    final_chunks = handle_cross_page_content(enhanced_chunks)
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"增强分块完成！保存到: {output_json_path}")
    print(f"原始chunks: {len(original_chunks)}")
    print(f"增强后chunks: {len(final_chunks)}")
    
    # 统计不同类型的chunks
    type_stats = {}
    for chunk in final_chunks:
        chunk_type = chunk.get('type', 'unknown')
        type_stats[chunk_type] = type_stats.get(chunk_type, 0) + 1
    
    print("Chunk类型统计:")
    for chunk_type, count in type_stats.items():
        print(f"  {chunk_type}: {count}")
    
    return final_chunks


def analyze_chunk_quality(chunks: List[dict]) -> Dict[str, Any]:
    """分析分块质量"""
    stats = {
        "total_chunks": len(chunks),
        "avg_chunk_size": 0,
        "size_distribution": {"small": 0, "medium": 0, "large": 0},
        "type_distribution": {},
        "cross_page_chunks": 0
    }
    
    total_size = 0
    for chunk in chunks:
        content_size = len(chunk['content'])
        total_size += content_size
        
        # 大小分布
        if content_size < 200:
            stats["size_distribution"]["small"] += 1
        elif content_size < 800:
            stats["size_distribution"]["medium"] += 1
        else:
            stats["size_distribution"]["large"] += 1
        
        # 类型分布
        chunk_type = chunk.get('type', 'unknown')
        stats["type_distribution"][chunk_type] = stats["type_distribution"].get(chunk_type, 0) + 1
        
        # 跨页chunks
        if 'merged' in chunk.get('id', ''):
            stats["cross_page_chunks"] += 1
    
    stats["avg_chunk_size"] = total_size / len(chunks) if chunks else 0
    
    return stats


def compare_chunking_strategies(original_path: str, enhanced_path: str):
    """比较原始分块和增强分块的效果"""
    print("=== 分块策略对比分析 ===")
    
    # 加载数据
    with open(original_path, 'r', encoding='utf-8') as f:
        original_chunks = json.load(f)
    
    with open(enhanced_path, 'r', encoding='utf-8') as f:
        enhanced_chunks = json.load(f)
    
    # 分析原始分块
    print("\n原始分块（按页）:")
    original_stats = analyze_chunk_quality(original_chunks)
    for key, value in original_stats.items():
        print(f"  {key}: {value}")
    
    # 分析增强分块
    print("\n增强分块（语义+结构）:")
    enhanced_stats = analyze_chunk_quality(enhanced_chunks)
    for key, value in enhanced_stats.items():
        print(f"  {key}: {value}")
    
    # 对比分析
    print("\n对比分析:")
    print(f"  chunks数量变化: {original_stats['total_chunks']} -> {enhanced_stats['total_chunks']}")
    print(f"  平均chunk大小: {original_stats['avg_chunk_size']:.1f} -> {enhanced_stats['avg_chunk_size']:.1f}")
    print(f"  跨页处理: {enhanced_stats['cross_page_chunks']} 个跨页chunks")


def main():
    """主函数：演示增强分块流程"""
    # 文件路径
    input_file = "all_pdf_page_chunks_mineru.json"
    output_file = "all_pdf_enhanced_chunks.json"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        print("请确保已经运行了MinerU处理流程")
        return
    
    # 处理增强分块
    try:
        enhanced_chunks = process_mineru_to_enhanced_chunks(
            input_json_path=input_file,
            output_json_path=output_file,
            chunk_size=512,
            overlap_size=50
        )
        
        # 分析对比
        compare_chunking_strategies(input_file, output_file)
        
        print(f"\n✅ 增强分块处理完成！")
        print(f"📁 输出文件: {output_file}")
        print(f"📊 生成了 {len(enhanced_chunks)} 个增强chunks")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
