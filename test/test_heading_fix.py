#!/usr/bin/env python3
"""
测试脚本：验证heading自动关联功能的正确性
"""
import json
from src.sequence_former.data_models import Chunk, LLMOutput, ProcessingState
from src.sequence_former.postprocessor import _associate_chunk_with_heading

def test_heading_association():
    """测试heading自动关联逻辑"""
    print("=== 测试heading自动关联功能 ===\n")
    
    # 测试用例1：LLM已返回heading，保持不变
    print("测试用例1：LLM已返回heading")
    chunk1 = Chunk(
        start_page=1, start_line=1, end_page=1, end_line=5,
        summary="测试摘要1", heading="已存在的标题"
    )
    _associate_chunk_with_heading(chunk1, ["主标题", "子标题"])
    print(f"结果：heading='{chunk1.heading}' (期望：已存在的标题)")
    assert chunk1.heading == "已存在的标题", "测试失败：LLM返回的heading被覆盖了"
    
    # 测试用例2：LLM未返回heading，使用最后一个heading
    print("\n测试用例2：LLM未返回heading")
    chunk2 = Chunk(
        start_page=2, start_line=1, end_page=2, end_line=5,
        summary="测试摘要2", heading=None
    )
    _associate_chunk_with_heading(chunk2, ["主标题", "子标题1", "子标题2"])
    print(f"结果：heading='{chunk2.heading}' (期望：子标题2)")
    assert chunk2.heading == "子标题2", "测试失败：未正确关联到最后一个heading"
    
    # 测试用例3：没有可用的heading
    print("\n测试用例3：没有可用的heading")
    chunk3 = Chunk(
        start_page=3, start_line=1, end_page=3, end_line=5,
        summary="测试摘要3", heading=None
    )
    _associate_chunk_with_heading(chunk3, [])
    print(f"结果：heading='{chunk3.heading}' (期望：None)")
    assert chunk3.heading is None, "测试失败：没有heading时应保持为None"
    
    # 测试用例4：空字符串heading视为未提供
    print("\n测试用例4：空字符串heading")
    chunk4 = Chunk(
        start_page=4, start_line=1, end_page=4, end_line=5,
        summary="测试摘要4", heading=""
    )
    _associate_chunk_with_heading(chunk4, ["主标题", "子标题"])
    print(f"结果：heading='{chunk4.heading}' (期望：子标题)")
    assert chunk4.heading == "子标题", "测试失败：空字符串应使用自动关联"
    
    print("\n=== 所有测试用例通过！ ===")

def test_with_real_data():
    """使用实际数据测试"""
    print("\n=== 使用实际数据测试 ===\n")
    
    # 模拟一个LLM输出
    llm_output = LLMOutput(
        hierarchical_headings=["## 6 Conclusions"],
        chunks=[
            Chunk(
                start_page=20, start_line=6, end_page=20, end_line=7,
                summary="Summarizes Kimi K2 as a 1T-parameter model",
                heading=None,  # 模拟LLM未返回heading
                metadata={"headings": ["## 6 Conclusions"]}
            )
        ]
    )
    
    # 应用heading关联
    for chunk in llm_output.chunks:
        _associate_chunk_with_heading(chunk, llm_output.hierarchical_headings)
    
    print("原始数据：")
    print(f"  hierarchical_headings: {llm_output.hierarchical_headings}")
    print(f"  chunk.heading (before): None")
    print(f"  chunk.heading (after): '{llm_output.chunks[0].heading}'")
    
    assert llm_output.chunks[0].heading == "## 6 Conclusions", "实际数据测试失败"
    print("实际数据测试通过！")

if __name__ == "__main__":
    test_heading_association()
    test_with_real_data()
    print("\n🎉 所有测试完成，heading自动关联功能工作正常！")