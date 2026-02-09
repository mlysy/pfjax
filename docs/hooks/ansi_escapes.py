import re

from ansi2html import Ansi2HTMLConverter

ansi_converter = Ansi2HTMLConverter(inline=True)


def on_page_markdown(markdown, *, page, config, files):
    """
    Apply color highlighting to ansi/bash/text code blocks.

    Does this by first converting codeblock to html, then enclosing in a
    `<div class="language-text hightlight></div>`.
    """

    def ansi_to_html(match):
        ansi_code = match.group(1)
        html = ansi_converter.convert(ansi_code, full=False)
        return f'<div class="language-text highlight"><pre><span></span><code>{html}</code></pre></div>'

    # match code blocks like ```ansi or ```bash
    ansi_block_pattern = re.compile(r"```[ \t]*(?:ansi|bash|text)\n(.*?)```", re.DOTALL)
    return ansi_block_pattern.sub(ansi_to_html, markdown)
