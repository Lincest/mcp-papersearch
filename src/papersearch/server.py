# src/paper_search/server.py
from typing import Any, List, Dict
import asyncio
import arxiv
import re
from datetime import datetime, timezone, timedelta
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# 创建服务器实例
server = Server("papersearch")

async def search_papers(days: int = 7, query_type: str = "moe", max_results: int = 100, 
                    field: str = None, keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    搜索最近发表的论文
    
    Args:
        days: 搜索最近几天的论文
        query_type: 搜索类型 (默认为 moe) 或 arxiv ID
        max_results: 返回结果的最大数量
        field: 研究领域
        keywords: 标题或摘要中的关键词列表
    """
    def calculate_relevance_score(paper, keywords: List[str]) -> float:
        """
        计算论文与关键词的相关性得分
        """
        if not keywords:
            return 1.0
            
        score = 0
        total_weight = 0
        
        # 标题中关键词权重为2，摘要中权重为1
        title_weight = 2
        summary_weight = 1
        
        for keyword in keywords:
            if keyword.lower() in paper.title.lower():
                score += title_weight
            if keyword.lower() in paper.summary.lower():
                score += summary_weight
            total_weight += (title_weight + summary_weight)
            
        return score / total_weight if total_weight > 0 else 0

    client = arxiv.Client()

    # 检查是否是 arxiv ID 格式
    arxiv_id_pattern = r'\d{4}\.\d{4,5}(?:v\d+)?'
    if query_type and re.match(arxiv_id_pattern, query_type):
        try:
            paper = next(client.results(arxiv.Search(id_list=[query_type])))
            return [{
                'title': paper.title,
                'authors': [str(author) for author in paper.authors],
                'summary': paper.summary,
                'url': paper.pdf_url,
                'published_date': paper.published.strftime('%Y-%m-%d'),
                'categories': paper.categories,
                'relevance_score': 1.0  # 精确匹配的论文相关性设为1
            }]
        except StopIteration:
            return []  # 如果找不到指定 ID 的论文，返回空列表
    
    # 构建查询语句
    query_parts = []
    
    # 添加原有的 MOE 相关查询
    if query_type == "moe inference":
        query_parts.append('(ti:"mixture of experts" OR ti:moe) AND (ti:deployment OR abs:deployment OR ti:inference OR abs:inference OR ti:efficient OR abs:efficient)')
    elif query_type:
        query_parts.append(query_type)
    
    # 添加研究领域查询
    if field:
        field_query = f'(cat:"{field}")'
        query_parts.append(field_query)
    
    # 添加关键词查询
    if keywords:
        keyword_conditions = []
        for keyword in keywords:
            keyword_condition = f'(ti:"{keyword}" OR abs:"{keyword}")'
            keyword_conditions.append(keyword_condition)
        keyword_query = " OR ".join(keyword_conditions)
        query_parts.append(f"({keyword_query})")
    
    # 组合所有查询条件
    final_query = " AND ".join(query_parts) if query_parts else "*:*"
        
    search = arxiv.Search(
        query=final_query,
        max_results=max(100, max_results * 2),  # 获取更多结果用于排序
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=days)
    
    papers = []
    for paper in client.results(search):
        if len(keywords) > 0 or paper.published >= cutoff_date:
            paper_dict = {
                'title': paper.title,
                'authors': [str(author) for author in paper.authors],
                'summary': paper.summary,
                'url': paper.pdf_url,
                'published_date': paper.published.strftime('%Y-%m-%d'),
                'categories': paper.categories
            }
            # 计算相关性得分
            relevance_score = calculate_relevance_score(paper, keywords)
            paper_dict['relevance_score'] = relevance_score
            papers.append(paper_dict)
    
    # 根据相关性得分排序
    papers.sort(key=lambda x: (-x['relevance_score'], x['published_date']), reverse=True)
    
    # 返回指定数量的论文
    return papers[:max_results]

def format_papers(papers: List[Dict[str, Any]], show_score: bool = False) -> str:
    """
    格式化论文列表为易读的文本格式
    
    Args:
        papers: 论文列表
        show_score: 是否显示相关性得分
    """
    if not papers:
        return "在指定时间范围内未找到相关论文。"
    
    formatted = []
    for i, paper in enumerate(papers, 1):
        paper_text = [
            f"{i}. {paper['title']}",
            f"作者: {', '.join(paper['authors'])}",
            f"发布日期: {paper['published_date']}",
            f"领域分类: {', '.join(paper['categories'])}",
            f"链接: {paper['url']}",
            f"摘要: {paper['summary']}"
        ]
        
        if show_score and 'relevance_score' in paper:
            paper_text.insert(1, f"相关性得分: {paper['relevance_score']:.2f}")
            
        formatted.append("\n".join(paper_text) + "\n---")
    
    return "\n\n".join(formatted)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用的工具"""
    return [
        types.Tool(
            name="papersearch",
            description="Arxiv 论文搜索",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "number",
                        "description": "要搜索的时间范围（天数）",
                        "default": 180
                    },
                    "query_type": {
                        "type": "string",
                        "description": "搜索类型 (默认为 moe), 或直接传入 arxiv ID (例如 2103.03404)",
                        "default": "moe"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "返回结果的最大数量",
                        "default": 100
                    },
                    "field": {
                        "type": "string",
                        "description": "研究领域",
                        "default": None
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标题或摘要中的关键词列表",
                        "default": None
                    }
                }
            }
        )
    ]

# 处理工具调用
@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """处理工具调用请求"""
    if not arguments:
        arguments = {}
        
    if name != "papersearch":
        raise ValueError(f"Unknown tool: {name}")
    
    context = server.request_context
    
    try:
        # 获取参数
        days = arguments.get("days", 7)
        query_type = arguments.get("query_type", "moe")
        max_results = arguments.get("max_results", 100)
        field = arguments.get("field")
        keywords = arguments.get("keywords")

        papers = await search_papers(days=days, query_type=query_type, max_results=max_results, field=field, keywords=keywords)
        papers_text = format_papers(papers)
        return [types.TextContent(type="text", text=papers_text)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"出现错误: {str(e)}")]

# 主函数
async def main():
    """运行服务器"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="papersearch",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
