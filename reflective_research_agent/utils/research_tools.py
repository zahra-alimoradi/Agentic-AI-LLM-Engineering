# --- Standard library ---
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote

# --- Third-party ---
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
import wikipedia

load_dotenv()

# A descriptive user-agent is recommended by arXiv.
session = requests.Session()
session.headers.update({
    "User-Agent": "ReflectiveResearchAgent/1.0 (mailto:your.email@example.com)"
})

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv and return paper metadata and abstracts."""
    if not query or not query.strip():
        return [{"error": "query must not be empty"}]

    max_results = max(1, min(int(max_results), 20))
    encoded_query = quote(query.strip(), safe=" ")
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}&start=0&max_results={max_results}"
    )

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        results = []
        for entry in root.findall("atom:entry", ATOM_NS):
            get_text = lambda tag: (
                entry.findtext(f"atom:{tag}", default="", namespaces=ATOM_NS).strip()
            )
            authors = [
                author.findtext("atom:name", default="", namespaces=ATOM_NS)
                for author in entry.findall("atom:author", ATOM_NS)
            ]

            pdf_url = None
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break

            results.append({
                "source": "arXiv",
                "title": get_text("title").replace("\n", " "),
                "authors": authors,
                "published": get_text("published")[:10],
                "url": get_text("id"),
                "summary": get_text("summary").replace("\n", " "),
                "link_pdf": pdf_url
            })
        return results

    except requests.exceptions.RequestException as exc:
        return [{"error": f"arXiv request failed: {exc}"}]
    except ET.ParseError as exc:
        return [{"error": f"arXiv XML parsing failed: {exc}"}]
    except Exception as exc:
        return [{"error": f"Unexpected arXiv error: {exc}"}]


arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Search arXiv for scientific papers and return titles, authors, dates, abstracts, and links.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research keywords or topic."},
                "max_results": {
                    "type": "integer", "description": "Maximum papers to return (1-20).",
                    "default": 5, "minimum": 1, "maximum": 20
                }
            },
            "required": ["query"]
        }
    }
}


def tavily_search_tool(
    query: str,
    max_results: int = 5,
    include_images: bool = False,
    search_depth: str = "basic"
) -> list[dict]:
    """Search the web with Tavily. search_depth can be 'basic' or 'advanced'."""
    if not query or not query.strip():
        return [{"error": "query must not be empty"}]

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "TAVILY_API_KEY not found. Add it to your .env file."}]

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query.strip(),
            max_results=max(1, min(int(max_results), 10)),
            include_images=bool(include_images),
            search_depth=search_depth if search_depth in {"basic", "advanced"} else "basic"
        )

        results = [{
            "source": "Tavily",
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "url": item.get("url", ""),
            "score": item.get("score")
        } for item in response.get("results", [])]

        if include_images:
            for image_url in response.get("images", []):
                results.append({"source": "Tavily image", "image_url": image_url})
        return results

    except Exception as exc:
        return [{"error": f"Tavily search failed: {exc}"}]


tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Search the web for current information, technical explanations, and trustworthy sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query."},
                "max_results": {
                    "type": "integer", "description": "Number of results (1-10).",
                    "default": 5, "minimum": 1, "maximum": 10
                },
                "include_images": {
                    "type": "boolean", "description": "Whether to include image URLs.",
                    "default": False
                },
                "search_depth": {
                    "type": "string", "enum": ["basic", "advanced"],
                    "description": "Use advanced for more in-depth searches.",
                    "default": "basic"
                }
            },
            "required": ["query"]
        }
    }
}


def wikipedia_search_tool(query: str, sentences: int = 5, results: int = 3) -> list[dict]:
    """Search Wikipedia and return summaries for the best matching pages."""
    if not query or not query.strip():
        return [{"error": "query must not be empty"}]

    try:
        matches = wikipedia.search(query.strip(), results=max(1, min(int(results), 5)))
        if not matches:
            return [{"error": f"No Wikipedia pages found for: {query}"}]

        output = []
        for title in matches:
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                summary = wikipedia.summary(
                    page.title,
                    sentences=max(1, min(int(sentences), 15)),
                    auto_suggest=False,
                    redirect=True
                )
                output.append({
                    "source": "Wikipedia",
                    "title": page.title,
                    "summary": summary,
                    "url": page.url
                })
            except wikipedia.DisambiguationError as exc:
                output.append({
                    "source": "Wikipedia",
                    "title": title,
                    "error": "Disambiguation page",
                    "options": exc.options[:10]
                })
            except wikipedia.PageError:
                continue
        return output or [{"error": f"Could not retrieve a Wikipedia page for: {query}"}]

    except Exception as exc:
        return [{"error": f"Wikipedia search failed: {exc}"}]


wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Find introductory background information and summaries from Wikipedia.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or article keywords."},
                "sentences": {
                    "type": "integer", "description": "Summary length in sentences.",
                    "default": 5, "minimum": 1, "maximum": 15
                },
                "results": {
                    "type": "integer", "description": "Number of matching pages to retrieve.",
                    "default": 3, "minimum": 1, "maximum": 5
                }
            },
            "required": ["query"]
        }
    }
}


# Map the function names exposed to the LLM to their Python implementations.
tool_mapping = {
    "arxiv_search_tool": arxiv_search_tool,
    "tavily_search_tool": tavily_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool
}

# All tool schemas, in the same order as the mapping above.
all_tool_definitions = [
    arxiv_tool_def,
    tavily_tool_def,
    wikipedia_tool_def
]
