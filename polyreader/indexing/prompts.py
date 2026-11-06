"""All prompt templates for PaperIndex."""

# --- Tree Search ---

TREE_SEARCH_PROMPT = """\
You are given a question and a tree structure of a scientific paper.
Each node contains a node_id, title, and a summary (as "summary" for leaf nodes or "prefix_summary" for parent nodes).
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{tree_json}

Please reply in the following JSON format:
{{
    "thinking": "<Your reasoning about which nodes are relevant>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else."""

# --- Tree Search Retry ---

TREE_SEARCH_RETRY_PROMPT = """\
You are given a question and a tree structure of a scientific paper.
Your previous search returned some invalid results. Please try again more carefully.

Question: {query}

Document tree structure:
{tree_json}

Previous issues:
{issues}

Please select ONLY valid node_ids that appear in the tree structure above.
Reply in the following JSON format:
{{
    "thinking": "<Your reasoning about which nodes are relevant, being careful to use exact node_ids>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else."""

# --- Node Summary ---

SUMMARIZE_NODE_PROMPT = """\
You are given text from a section of a scientific paper. The text may contain content \
from adjacent sections on the same page. ONLY summarize content that belongs to the \
section titled "{title}". Ignore any content from other sections.

Generate a concise summary focusing on key findings, materials/methods, and quantitative results. \
At the end, briefly note what types of content the section contains (e.g., tables, figures, \
reaction schemes, NMR/GPC/DSC data, equations, or other characterization results) so that a \
search system can route queries to the right section.

Section title: {title}
Section text: {text}

Directly return the summary, do not include any other text."""

# --- Structure Extraction (LLM-driven, Tier 2 fallback) ---

EXTRACT_STRUCTURE_INIT_PROMPT = """\
You are an expert in extracting hierarchical tree structure from scientific papers,
your task is to generate the section tree structure of the paper.

The structure variable is the numeric system which represents the index of the hierarchy section. \
For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, extract the EXACT original section title from the text, only fix the space inconsistency.

The provided text contains tags like <page_N> to indicate the start of page N.

For the page_num, extract the page number where the section starts.

The response should be in the following format:
[
    {{
        "structure": <structure index, "x.x.x"> (string),
        "title": <title of the section, keep the original title>,
        "page_num": <page number> (keep the format)
    }},
    ...
]

Directly return the final JSON structure. Do not output anything else.

Given text:
{text}"""

EXTRACT_STRUCTURE_CONTINUE_PROMPT = """\
You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, extract the EXACT original section title from the text, only fix the space inconsistency.

The provided text contains tags like <page_N> to indicate the start of page N.

For the page_num, extract the page number where the section starts.

The response should be in the same format:
[
    {{
        "structure": <structure index, "x.x.x"> (string),
        "title": <title of the section, keep the original title>,
        "page_num": <page number> (keep the format)
    }},
    ...
]

Directly return the additional part of the JSON structure. Do not output anything else.

Given text:
{text}

Previous tree structure:
{prev_structure}"""

# --- Fix Section Index (single entry re-extraction) ---

FIX_SECTION_INDEX_PROMPT = """\
You are given a section title and several pages of a document.
Find the page where this section starts.

The text contains tags like <page_N> indicating page starts.

Section title: {title}
Text: {text}

Reply in JSON:
{{
    "thinking": "<explain which page contains the start of this section>",
    "page_num": <page number>
}}

Directly return the JSON. Do not output anything else."""

# --- Check Appear Start (LLM-based) ---

CHECK_APPEAR_START_PROMPT = """\
You will be given a section title and the text of a single page.
Your task is to determine whether this section starts at the very beginning of the page.

If there is other content (from a previous section, figures, tables, etc.) before the section title, answer "no".
If the section title is the first meaningful content on the page (ignoring headers/footers/page numbers), answer "yes".

Do fuzzy matching: ignore minor spacing or formatting differences when locating the title.

Section title: {title}

Page text:
{page_text}

Reply in JSON:
{{
    "thinking": "<explain whether the section title appears at the start of the page>",
    "start_begin": "<yes or no>"
}}

Directly return the JSON. Do not output anything else."""

# --- Metadata Extraction ---

EXTRACT_METADATA_PROMPT = """\
Extract bibliographic metadata from this scientific paper text (first and last pages).
Return JSON:
{{
    "title": "<paper title>",
    "authors": ["<author1>", "<author2>", ...],
    "journal": "<journal name or null>",
    "year": <publication year as integer or null>,
    "doi": "<DOI string or null>",
    "volume": "<volume or null>",
    "pages": "<pages or null>"
}}

Important extraction hints:
- Year: Look for "Published", "Received", "Accepted" dates, copyright year, or any year near the journal name. Extract only the 4-digit year as an integer.
- DOI: Return ONLY the DOI itself (e.g., "10.1234/example"), NOT a URL or markdown link. Look for patterns like "10.xxxx/..." anywhere in the text, including URLs containing "doi.org/". Do NOT extract DOIs from the References section — those belong to cited works, not this paper.
- Journal: Often appears near the top of the first page or in the header/footer.
- Volume/Pages: May appear near the journal name or DOI.
{hints_section}
Paper text:
{text}

Directly return the JSON. Do not output anything else."""

# --- Document Description ---

GENERATE_DOC_DESCRIPTION_PROMPT = """\
Generate a one-sentence description of this scientific paper based on its structure.
Focus on the main topic, materials studied, and type of study.

Document Structure: {structure}

Directly return the description, do not include any other text."""

