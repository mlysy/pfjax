# from bs4 import BeautifulSoup
import re


def on_page_markdown(markdown, *, page, config, files):
    """
    Allow Markdown annotation in csl reference list.

    Does this by adding `markdown="1"` to all divs with
    `class=[csl-bib-body|csl-entry]`.

    ~~Uses BeautifulSoup to perform the substitution, which seems safer than
    regex in plain markdown.  Does this by wrapping markdown in an html div
    first.~~

    Edit: use regex in Markdown, because the converting to html mangles the
    <url> Markdown links.
    """
    # regex for csl divs
    csl_div_pattern = re.compile(
        r'<div([^>]*class="[^"]*(?:csl-bib-body|csl-entry)[^"]*"[^>]*)>',
        re.IGNORECASE,
    )

    def add_markdown(match):
        tag = match.group(0)
        if "markdown=" in tag:
            return tag
        return tag[:-1] + ' markdown="1">'

    return csl_div_pattern.sub(add_markdown, markdown)

    # print(f"Hook running for page: {page.file.src_path}")
    #
    # # wrap markdown in html
    # wrapper = f"<div>{markdown}</div>"
    # soup = BeautifulSoup(wrapper, "html.parser")

    # # add markdown="1" to all bibliography divs and entries
    # modified = 0
    # for div in soup.find_all("div"):
    #     classes = div.get("class", [])
    #     if "csl-bib-body" in classes or "csl-entry" in classes:
    #         div["markdown"] = "1"
    #         print(f"div: {div}")
    #         modified += 1

    # if modified:
    #     print(f"Added markdown='1' to {modified} div(s) in {page.file.src_path}")

    # # return inner HTML of the wrapper as the new Markdown
    # return soup.div.decode_contents()
