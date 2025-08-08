#!/usr/bin/env python3
"""
AI IDE Browser Automation System
Playwright-based web scraping and automation for RAG and web capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import aiohttp
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

class BrowserAutomation:
    """Advanced browser automation for AI IDE web capabilities"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_initialized = False
        
    async def initialize(self, headless: bool = True):
        """Initialize Playwright browser"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = await self.context.new_page()
            self.is_initialized = True
            logger.info("Browser automation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def scrape_documentation(self, url: str) -> Dict[str, Any]:
        """Scrape technical documentation for RAG"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            await self.page.goto(url, wait_until='networkidle')
            
            # Extract content
            content = await self.page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(el => el.remove());
                    
                    // Get main content
                    const main = document.querySelector('main, article, .content, #content') || document.body;
                    
                    return {
                        title: document.title,
                        text: main.innerText,
                        html: main.innerHTML,
                        links: Array.from(document.querySelectorAll('a')).map(a => ({
                            text: a.innerText,
                            href: a.href
                        }))
                    };
                }
            """)
            
            return {
                'url': url,
                'title': content['title'],
                'text': content['text'],
                'html': content['html'],
                'links': content['links'],
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {'url': url, 'error': str(e)}    

    async def search_stackoverflow(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Stack Overflow for coding solutions"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            search_url = f"https://stackoverflow.com/search?q={query.replace(' ', '+')}"
            await self.page.goto(search_url, wait_until='networkidle')
            
            results = await self.page.evaluate(f"""
                () => {{
                    const results = [];
                    const items = document.querySelectorAll('.s-post-summary');
                    
                    for (let i = 0; i < Math.min({max_results}, items.length); i++) {{
                        const item = items[i];
                        const titleEl = item.querySelector('.s-link');
                        const excerptEl = item.querySelector('.s-post-summary--content-excerpt');
                        const tagsEl = item.querySelectorAll('.s-tag');
                        const votesEl = item.querySelector('.s-post-summary--stats-item-number');
                        
                        if (titleEl) {{
                            results.push({{
                                title: titleEl.innerText,
                                url: 'https://stackoverflow.com' + titleEl.getAttribute('href'),
                                excerpt: excerptEl ? excerptEl.innerText : '',
                                tags: Array.from(tagsEl).map(tag => tag.innerText),
                                votes: votesEl ? votesEl.innerText : '0'
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search Stack Overflow: {e}")
            return []
    
    async def scrape_github_repo(self, repo_url: str) -> Dict[str, Any]:
        """Scrape GitHub repository information"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            await self.page.goto(repo_url, wait_until='networkidle')
            
            repo_info = await self.page.evaluate("""
                () => {
                    const readme = document.querySelector('#readme');
                    const description = document.querySelector('[data-pjax="#repo-content-pjax-container"] p');
                    const topics = document.querySelectorAll('.topic-tag');
                    const languages = document.querySelectorAll('.BorderGrid-cell .color-fg-default');
                    
                    return {
                        name: document.querySelector('h1 strong a')?.innerText || '',
                        description: description?.innerText || '',
                        readme: readme?.innerText || '',
                        topics: Array.from(topics).map(t => t.innerText),
                        languages: Array.from(languages).map(l => l.innerText),
                        stars: document.querySelector('#repo-stars-counter-star')?.innerText || '0',
                        forks: document.querySelector('#repo-network-counter')?.innerText || '0'
                    };
                }
            """)
            
            return {
                'url': repo_url,
                'type': 'github_repo',
                **repo_info,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape GitHub repo {repo_url}: {e}")
            return {'url': repo_url, 'error': str(e)}
    
    async def automated_research(self, topic: str) -> Dict[str, Any]:
        """Perform automated research on a coding topic"""
        if not self.is_initialized:
            await self.initialize()
            
        results = {
            'topic': topic,
            'stackoverflow': [],
            'documentation': [],
            'github_repos': [],
            'articles': []
        }
        
        try:
            # Search Stack Overflow
            so_results = await self.search_stackoverflow(topic)
            results['stackoverflow'] = so_results
            
            # Search for documentation
            doc_search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}+documentation+site:docs.python.org+OR+site:developer.mozilla.org"
            await self.page.goto(doc_search_url, wait_until='networkidle')
            
            doc_links = await self.page.evaluate("""
                () => {
                    const links = [];
                    const results = document.querySelectorAll('h3 a');
                    for (let i = 0; i < Math.min(5, results.length); i++) {
                        const link = results[i];
                        if (link.href.includes('docs.') || link.href.includes('developer.')) {
                            links.push({
                                title: link.innerText,
                                url: link.href
                            });
                        }
                    }
                    return links;
                }
            """)
            
            # Scrape documentation pages
            for doc_link in doc_links[:3]:  # Limit to 3 docs
                try:
                    doc_content = await self.scrape_documentation(doc_link['url'])
                    results['documentation'].append(doc_content)
                except Exception as e:
                    logger.error(f"Failed to scrape doc {doc_link['url']}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed automated research for {topic}: {e}")
            return results
    
    async def monitor_web_changes(self, urls: List[str], interval: int = 3600) -> None:
        """Monitor web pages for changes (useful for tracking updates)"""
        if not self.is_initialized:
            await self.initialize()
            
        while True:
            try:
                for url in urls:
                    try:
                        await self.page.goto(url, wait_until='networkidle')
                        content_hash = await self.page.evaluate("""
                            () => {
                                const content = document.body.innerText;
                                // Simple hash function
                                let hash = 0;
                                for (let i = 0; i < content.length; i++) {
                                    const char = content.charCodeAt(i);
                                    hash = ((hash << 5) - hash) + char;
                                    hash = hash & hash; // Convert to 32-bit integer
                                }
                                return hash;
                            }
                        """)
                        
                        # Store/compare hash (implement storage mechanism)
                        logger.info(f"Monitored {url}, content hash: {content_hash}")
                        
                    except Exception as e:
                        logger.error(f"Failed to monitor {url}: {e}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

class WebRAGIntegration:
    """Integration between browser automation and RAG system"""
    
    def __init__(self, browser_automation: BrowserAutomation):
        self.browser = browser_automation
        self.knowledge_cache = {}
    
    async def enhance_query_with_web_data(self, query: str) -> Dict[str, Any]:
        """Enhance RAG queries with real-time web data"""
        try:
            # Perform automated research
            research_results = await self.browser.automated_research(query)
            
            # Extract and structure knowledge
            knowledge_items = []
            
            # Process Stack Overflow results
            for so_result in research_results['stackoverflow']:
                knowledge_items.append({
                    'type': 'stackoverflow',
                    'title': so_result['title'],
                    'content': so_result['excerpt'],
                    'url': so_result['url'],
                    'tags': so_result['tags'],
                    'relevance_score': self._calculate_relevance(query, so_result['excerpt'])
                })
            
            # Process documentation
            for doc in research_results['documentation']:
                if 'text' in doc:
                    knowledge_items.append({
                        'type': 'documentation',
                        'title': doc['title'],
                        'content': doc['text'][:1000],  # Truncate for processing
                        'url': doc['url'],
                        'relevance_score': self._calculate_relevance(query, doc['text'])
                    })
            
            # Sort by relevance
            knowledge_items.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'query': query,
                'web_knowledge': knowledge_items[:10],  # Top 10 most relevant
                'research_summary': research_results,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to enhance query with web data: {e}")
            return {'query': query, 'error': str(e)}
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Simple relevance scoring"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

# Example usage and testing
async def main():
    """Test browser automation functionality"""
    browser = BrowserAutomation()
    
    try:
        await browser.initialize()
        
        # Test Stack Overflow search
        results = await browser.search_stackoverflow("python async await")
        print(f"Found {len(results)} Stack Overflow results")
        
        # Test automated research
        research = await browser.automated_research("FastAPI authentication")
        print(f"Research completed for FastAPI authentication")
        
        # Test web RAG integration
        web_rag = WebRAGIntegration(browser)
        enhanced_query = await web_rag.enhance_query_with_web_data("How to implement JWT authentication")
        print(f"Enhanced query with {len(enhanced_query.get('web_knowledge', []))} knowledge items")
        
    finally:
        await browser.cleanup()

if __name__ == "__main__":
    asyncio.run(main())