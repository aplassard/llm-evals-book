"""LangGraph workflow for researching article references."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import List
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REFERER = "https://github.com/andrewplassard/llm-evals-book"
DEFAULT_TITLE = "Writing Assistant Research Agent"


ZOTERO_CHEAT_SHEET = dedent(
    """\
    Zotero item quick reference:
    - journalArticle: title, creators (authors), publicationTitle, date, DOI or url; extras volume, issue, pages.
    - conferencePaper: title, creators, conferenceName or proceedingsTitle, date; extras pages, DOI, url.
    - book: title, creators or editors, publisher, date; extras ISBN, place.
    - bookSection: title (chapter), creators, bookTitle, date, publisher; extras pages.
    - report: title, creators or institution, institution, date, url; extras reportNumber.
    - thesis: title, creators, university, date.
    - webpage: title, creators if available, date, url; extras websiteTitle, accessDate.
    - presentation: title, presenter, date, url if available.
    - videoRecording: title, presenter, date, url.
    - podcast: title, host/guest if known, date, url.
    Return creator objects as {"firstName": "...", "lastName": "...", "creatorType": "author"|...}.
    """
)


@dataclass
class ArticleTask:
    """Represents a single article request extracted from the planning JSON."""

    name: str
    details: str
    status: str

    @property
    def is_known(self) -> bool:
        return self.status.lower() == "known"


def build_system_prompt(task: ArticleTask) -> str:
    """Craft the system instructions for the agent based on task status."""

    base_instructions = [
        "You are an expert research librarian who finds reliable bibliographic data.",
        "Always ground answers in verifiable sources and mention the evidence gathered.",
        "Use the Tavily web search tool whenever you need supporting pages, paying attention to authority.",
        "Produce outputs that can be imported into Zotero without further editing.",
        "Return your final response as strict JSON following the requested schema.",
        "If information is uncertain, transparently mark fields as null and add a note describing the gap.",
    ]

    if task.is_known:
        base_instructions.append(
            "The reference is labeled as 'known', so prioritise confirming canonical metadata for the expected work."
        )
        base_instructions.append(
            "Confirm identifiers (DOI, URL) and publication venue from trusted sources like publisher pages or established indexes."
        )
        base_instructions.append(
            "Do not return unrelated works: ensure the title and metadata correspond to the requested reference name/details before finalising."
        )
    else:
        base_instructions.append(
            "The reference is labeled as 'unknown'. Identify the most relevant publications that satisfy the request."
        )
        base_instructions.append(
            "When multiple plausible sources exist, gather two or three strong candidates with clear reasoning."
        )

    base_instructions.append(ZOTERO_CHEAT_SHEET)

    return "\n".join(base_instructions)


def build_user_prompt(task: ArticleTask, note_summary: str, extra_instruction: str | None = None) -> str:
    """Prepare the user content describing the research need."""

    status_clause = "known reference" if task.is_known else "unknown reference that requires discovery"
    schema_description = dedent(
        """\
        Final JSON schema (UTF-8, minified or pretty):
        {
          "items": [
            {
              "itemType": "journalArticle" | "conferencePaper" | "book" | "bookSection" | "report" | "thesis" | "webpage" | "presentation" | "videoRecording" | "podcast",
              "title": string,
              "creators": [ {"firstName": string, "lastName": string, "creatorType": string} ],
              "date": string | null,
              "publicationTitle": string | null,
              "conferenceName": string | null,
              "proceedingsTitle": string | null,
              "publisher": string | null,
              "institution": string | null,
              "volume": string | null,
              "issue": string | null,
              "pages": string | null,
              "doi": string | null,
              "url": string | null,
              "abstractNote": string | null,
              "tags": [string],
              "notes": [string]
            }
          ],
          "context": {
            "articleName": string,
            "status": "known" | "unknown",
            "evidence": [
              {
                "source": string,
                "snippet": string
              }
            ]
          }
        }
        Always include the "context" block with at least one evidence entry referencing the sources you used.
        """
    )

    validation_clause = (
        "Validate that the title and metadata you return align with the provided name/details; "
        "if you cannot confirm the match, keep researching rather than returning a mismatched work."
        if task.is_known
        else ""
    )

    suggested_query = f"{task.name} {task.details}".strip()

    prompt = dedent(
        f"""\
        A {status_clause} needs metadata suitable for Zotero import.
        Transcript summary (for context):\n{note_summary}

        Requested entry:\n- Name: {task.name}\n- Details: {task.details}\n- Status: {task.status}

        Begin by issuing a Tavily search using the query: "{suggested_query}".
        Use the Tavily search tool to gather supporting evidence as needed. {validation_clause}
        {schema_description}
        """
    )

    if extra_instruction:
        prompt += "\n\nCorrection guidance:\n" + extra_instruction.strip()

    return prompt


def create_article_agent(task: ArticleTask) -> ChatOpenAI:
    """Instantiate the OpenRouter-backed chat model configured for the agent."""

    return ChatOpenAI(
        model="x-ai/grok-4-fast",
        api_key=None,  # taken from environment OPENROUTER_API_KEY
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": DEFAULT_REFERER,
            "X-Title": DEFAULT_TITLE,
        },
        temperature=0.2,
        timeout=120,
    )


def build_graph(task: ArticleTask) -> Runnable:
    """Construct the LangGraph ReAct agent for the given task."""

    llm = create_article_agent(task)
    tools: List[TavilySearch] = [TavilySearch(max_results=5, include_answer=True)]
    system_prompt = build_system_prompt(task)
    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))
    return agent
