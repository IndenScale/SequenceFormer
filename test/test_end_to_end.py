#!/usr/bin/env python3
"""
端到端测试：使用KIMI K2_OPEN AGENTIC INTELLIGENCE.pdf_MinerU_truncated测试heading自动关联
"""
import json
import os
from src.sequence_former.data_models import Chunk, LLMOutput, ProcessingState
from src.sequence_former.llm_processor import build_prompt
from src.sequence_former.postprocessor import finalize_chunks_and_update_state
from src.sequence_former.config import Settings

def test_end_to_end():
    """端到端测试主函数"""
    # 创建settings对象
    settings = Settings(
        target_chunk_size=1000,
        chunk_size_tolerance=0.2,
        enable_vlm=False,
        input_path="test_input.txt"
    )

    # 加载测试文档
    test_doc_path = "E:\\SequenceFormer\\examples\\sample_documents\\KIMI K2_OPEN AGENTIC INTELLIGENCE.pdf_MinerU_truncated\\full.md"

    if not os.path.exists(test_doc_path):
        print(f"❌ 测试文档不存在: {test_doc_path}")
        return False

    # 读取测试文档内容
    with open(test_doc_path, 'r', encoding='utf-8') as f:
        document_content = f.read()

    print(f"✅ 成功加载测试文档: {len(document_content)} 字符")

    # 模拟LLM响应（基于mineru_test_output.jsonl中的实际模式）
    mock_llm_response = {
        "hierarchical_headings": [
            "## 1 Introduction",
            "## 2 Model Architecture",
            "## 3 Training Methodology",
            "## 4 Evaluation Results",
            "## 5 Discussion",
            "## 6 Conclusions"
        ],
        "chunks": [
            {
                "start_page": 1,
                "start_line": 1,
                "end_page": 1,
                "end_line": 10,
                "summary": "Introduction to Kimi K2 as a 1T-parameter model with advanced capabilities",
                "heading": "## 1 Introduction",  # LLM返回了heading
                "metadata": {"type": "introduction", "section": "main"}
            },
            {
                "start_page": 2,
                "start_line": 1,
                "end_page": 2,
                "end_line": 15,
                "summary": "Overview of the transformer-based architecture design",
                "heading": None,  # LLM未返回heading，需要自动关联
                "metadata": {"type": "architecture", "section": "technical"}
            },
            {
                "start_page": 20,
                "start_line": 6,
                "end_page": 20,
                "end_line": 7,
                "summary": "Summarizes Kimi K2 as a 1T-parameter model",
                "heading": "",  # LLM返回空字符串，需要自动关联
                "metadata": {"headings": ["## 6 Conclusions"]}
            }
        ]
    }

    # 创建LLM输出对象
    llm_output = LLMOutput(
        hierarchical_headings=mock_llm_response["hierarchical_headings"],
        chunks=[Chunk(**chunk) for chunk in mock_llm_response["chunks"]]
    )

    print("\n📋 测试数据概览:")
    print(f"  层级标题: {len(llm_output.hierarchical_headings)} 个")
    print(f"  文本块: {len(llm_output.chunks)} 个")

    # 处理前的heading状态
    print("\n🔍 处理前的heading状态:")
    for i, chunk in enumerate(llm_output.chunks):
        print(f"  Chunk {i+1}: heading='{chunk.heading}'")

    # 创建处理状态
    processing_state = ProcessingState(
        doc_id="test_kimi_document",
        hierarchical_headings=llm_output.hierarchical_headings,
        chunks=[]
    )

    # 应用后处理器
    try:        # 创建模拟的combined_enriched_lines
        combined_enriched_lines = []
        for line in document_content.split('\n'):
            combined_enriched_lines.append({
                'page': 1,
                'line': len(combined_enriched_lines) + 1,
                'text': line
            })

        # 修复函数调用，添加缺失的参数
        reliable_chunks, new_state = finalize_chunks_and_update_state(
            llm_output,
            combined_enriched_lines,
            1,  # current_page_boundary
            processing_state,
            settings  # 需要创建settings对象
        )
        processing_state = new_state
        print("\n✅ 后处理器执行成功")
    except Exception as e:
        print(f"❌ 后处理器执行失败: {e}")
        return False

    # 验证结果
    print("\n🎯 处理后的heading状态:")

    # 获取处理后的chunks
    all_chunks = reliable_chunks  # 使用后处理器返回的chunks

    expected_headings = [
        "## 1 Introduction",  # 已有heading保持不变
        "## 6 Conclusions",   # None关联到最后一个heading
        "## 6 Conclusions"    # 空字符串关联到最后一个heading
    ]

    # 验证每个chunk的heading
    for i, (chunk, expected_heading) in enumerate(zip(all_chunks, expected_headings)):
        actual_heading = chunk.heading if chunk.heading else "None"
        print(f"  Chunk {i+1}: heading='{actual_heading}' (期望: '{expected_heading}')")

        if actual_heading != expected_heading:
            print(f"    ❌ 不匹配！")
            return False

    print("✅ heading自动关联验证通过！")

    # 测试metadata中的heading信息
    print("\n📊 测试metadata中的heading信息...")
    for chunk in all_chunks:
        if hasattr(chunk, 'metadata') and chunk.metadata and 'heading' in chunk.metadata:
            print(f"  Chunk heading in metadata: {chunk.metadata['heading']}")

    return True

def test_metadata_heading_integration():
    """测试metadata中的heading信息是否正确处理"""
    print("\n=== 测试metadata中的heading信息集成 ===\n")

    # 创建包含metadata heading的测试数据
    chunk_with_meta_heading = Chunk(
        start_page=1, start_line=1, end_page=1, end_line=5,
        summary="测试摘要",
        heading=None,  # 需要自动关联
        metadata={"headings": ["## 测试标题"]}
    )

    hierarchical_headings = ["## 测试标题", "## 子标题"]

    # 应用heading关联
    from src.sequence_former.postprocessor import _associate_chunk_with_heading
    _associate_chunk_with_heading(chunk_with_meta_heading, hierarchical_headings)

    print(f"Metadata中的headings: {chunk_with_meta_heading.metadata.get('headings', [])}")
    print(f"自动关联后的heading: '{chunk_with_meta_heading.heading}'")

    # 验证是否优先使用hierarchical_headings
    expected = "## 子标题"  # 应该使用最后一个hierarchical_heading
    success = chunk_with_meta_heading.heading == expected

    if success:
        print("✅ 正确优先使用hierarchical_headings而非metadata中的headings")
    else:
        print(f"❌ 期望heading='{expected}'，实际='{chunk_with_meta_heading.heading}'")

    return success

if __name__ == "__main__":
    success = test_end_to_end()
    if success:
        print("\n🎉 所有测试通过！heading自动关联功能正常工作")
    else:
        print("\n❌ 测试失败，请检查代码")