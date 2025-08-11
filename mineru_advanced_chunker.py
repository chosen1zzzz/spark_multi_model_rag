import re
from typing import List, Dict, Any
import json

class MinerUAdvancedChunker:
    """基于MinerU结构化内容的高级分块器"""
    
    def __init__(self, chunk_size: int = 512, overlap_size: int = 50, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_mineru_content(self, content: str, metadata: dict) -> List[dict]:
        """基于MinerU结构化内容的智能分块"""
        chunks = []
        
        # 1. 分离不同类型的内容
        content_blocks = self._parse_structured_content(content)
        
        current_chunk = ""
        chunk_id = 0
        
        for block in content_blocks:
            if block['type'] == 'table':
                # 表格独立成chunk
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, "text"))
                    chunk_id += 1
                    current_chunk = ""
                
                # 表格chunk
                table_chunk = self._create_chunk(
                    block['content'], metadata, chunk_id, "table"
                )
                chunks.append(table_chunk)
                chunk_id += 1
                
            elif block['type'] == 'heading':
                # 标题处理：与下一段内容合并
                if current_chunk.strip() and len(current_chunk) > self.chunk_size:
                    chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, "text"))
                    chunk_id += 1
                    current_chunk = block['content']
                else:
                    current_chunk += "\n\n" + block['content'] if current_chunk else block['content']
                    
            elif block['type'] == 'text':
                # 普通文本处理
                if len(current_chunk + block['content']) > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, "text"))
                        chunk_id += 1
                        
                        # 添加重叠
                        overlap = self._get_text_overlap(current_chunk)
                        current_chunk = overlap + "\n\n" + block['content']
                    else:
                        current_chunk = block['content']
                else:
                    current_chunk += "\n\n" + block['content'] if current_chunk else block['content']
            
            elif block['type'] == 'image':
                # 图片描述作为独立chunk
                if len(block['content']) > self.min_chunk_size:
                    chunks.append(self._create_chunk(block['content'], metadata, chunk_id, "image"))
                    chunk_id += 1
        
        # 处理最后一个chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_id, "text"))
        
        return chunks
    
    def _parse_structured_content(self, content: str) -> List[dict]:
        """解析MinerU的结构化内容"""
        blocks = []
        lines = content.split('\n')
        current_block = ""
        current_type = "text"
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测表格
            if '<table>' in line:
                if current_block.strip():
                    blocks.append({'type': current_type, 'content': current_block.strip()})
                current_block = line
                current_type = "table"
                in_table = True
            elif '</table>' in line:
                current_block += "\n" + line
                blocks.append({'type': current_type, 'content': current_block.strip()})
                current_block = ""
                current_type = "text"
                in_table = False
            elif in_table:
                current_block += "\n" + line
            # 检测标题
            elif re.match(r'^#+\s', line):
                if current_block.strip():
                    blocks.append({'type': current_type, 'content': current_block.strip()})
                current_block = line
                current_type = "heading"
            # 检测图片
            elif re.match(r'!\[.*\]\(.*\)', line):
                if current_block.strip():
                    blocks.append({'type': current_type, 'content': current_block.strip()})
                blocks.append({'type': 'image', 'content': line})
                current_block = ""
                current_type = "text"
            else:
                if current_type == "heading" and line:
                    current_block += "\n" + line
                elif current_type == "text":
                    current_block += "\n" + line if current_block else line
                else:
                    if current_block.strip():
                        blocks.append({'type': current_type, 'content': current_block.strip()})
                    current_block = line
                    current_type = "text"
        
        if current_block.strip():
            blocks.append({'type': current_type, 'content': current_block.strip()})
        
        return blocks
    
    def _create_chunk(self, content: str, metadata: dict, chunk_id: int, content_type: str) -> dict:
        """创建chunk对象"""
        return {
            "id": f"{metadata['file_name']}_page_{metadata['page']}_chunk_{chunk_id}",
            "content": content,
            "type": content_type,
            "metadata": {
                **metadata,
                "chunk_id": chunk_id,
                "content_type": content_type,
                "chunk_strategy": "mineru_structured"
            }
        }
    
    def _get_text_overlap(self, text: str) -> str:
        """获取文本重叠部分"""
        sentences = re.split(r'[.!?。！？]', text)
        if len(sentences) <= 1:
            return ""
        
        # 取最后几个句子作为重叠
        overlap_sentences = sentences[-2:] if len(sentences) >= 2 else sentences[-1:]
        overlap = "".join(overlap_sentences).strip()
        
        # 限制重叠长度
        if len(overlap) > self.overlap_size:
            overlap = overlap[-self.overlap_size:]
        
        return overlap


def is_table_continuation(chunk1: dict, chunk2: dict) -> bool:
    """检测表格是否跨页"""
    content1 = chunk1['content']
    content2 = chunk2['content']
    
    # 检查第一个chunk是否以不完整表格结尾
    if '<table>' in content1 and '</table>' not in content1:
        return True
    
    # 检查第二个chunk是否以表格行开始
    if content2.strip().startswith('<tr>') or content2.strip().startswith('</table>'):
        return True
    
    return False


def is_paragraph_continuation(chunk1: dict, chunk2: dict) -> bool:
    """检测段落是否跨页"""
    content1 = chunk1['content'].strip()
    content2 = chunk2['content'].strip()
    
    # 检查第一个chunk是否以不完整句子结尾
    if content1 and not re.search(r'[.!?。！？]$', content1):
        # 检查第二个chunk是否以小写字母开始（可能是句子的延续）
        if content2 and content2[0].islower():
            return True
    
    return False


def merge_table_chunks(chunk1: dict, chunk2: dict) -> dict:
    """合并跨页表格"""
    merged_content = chunk1['content'] + "\n" + chunk2['content']
    
    return {
        "id": f"{chunk1['id']}_merged_{chunk2['id']}",
        "content": merged_content,
        "type": "table_merged",
        "metadata": {
            **chunk1['metadata'],
            "merged_from": [chunk1['id'], chunk2['id']],
            "content_type": "table_cross_page"
        }
    }


def merge_paragraph_chunks(chunk1: dict, chunk2: dict) -> dict:
    """合并跨页段落"""
    merged_content = chunk1['content'] + " " + chunk2['content']
    
    return {
        "id": f"{chunk1['id']}_merged_{chunk2['id']}",
        "content": merged_content,
        "type": "text_merged",
        "metadata": {
            **chunk1['metadata'],
            "merged_from": [chunk1['id'], chunk2['id']],
            "content_type": "text_cross_page"
        }
    }


def handle_cross_page_content(chunks: List[dict]) -> List[dict]:
    """处理跨页的表格和段落"""
    enhanced_chunks = chunks.copy()
    
    # 按文件分组
    chunks_by_file = {}
    for chunk in chunks:
        file_name = chunk['metadata']['file_name']
        if file_name not in chunks_by_file:
            chunks_by_file[file_name] = []
        chunks_by_file[file_name].append(chunk)
    
    for file_name, file_chunks in chunks_by_file.items():
        # 按页码排序
        file_chunks.sort(key=lambda x: int(x['metadata']['page']))
        
        # 检测跨页内容
        for i in range(len(file_chunks) - 1):
            current_chunk = file_chunks[i]
            next_chunk = file_chunks[i + 1]
            
            # 检测表格跨页
            if is_table_continuation(current_chunk, next_chunk):
                merged_table = merge_table_chunks(current_chunk, next_chunk)
                enhanced_chunks.append(merged_table)
            
            # 检测段落跨页
            elif is_paragraph_continuation(current_chunk, next_chunk):
                merged_paragraph = merge_paragraph_chunks(current_chunk, next_chunk)
                enhanced_chunks.append(merged_paragraph)
    
    return enhanced_chunks
