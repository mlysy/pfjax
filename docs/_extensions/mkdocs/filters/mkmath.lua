--- Modify quarto numbered equation for mkdocs output.
--
--  Puts space between `<span>` its contents and `</span>` so html rendered doesn't get confused.
--
--  NOTES:
--
--  - Must run this at `post-quarto` location.
--
--  TODO:
--
--  - [x] Make sure to only process `Span` elements containing `Math`.
--  - [ ] Only apply to `gmf` output.
--
--  @param el Element of type `Span`.
--
--  @return A modified AST `Span` block. 
function Span(el)
  if el.identifier ~= "" and el.content[1].t == "Math" and el.content[1].mathtype == "DisplayMath" then
    el.content = {
      pandoc.RawInline('html', '\n\n'),
      el.content[1],
      pandoc.RawInline('html', '\n\n')
    }
  end
  return el
end

--- Modify include block for latex macros for mkdocs.
--
-- The quarto document must have the latex macros defined using one of the
-- following methods:
--
-- Method 1:
-- ```
-- ::: {.hidden}
-- $$
-- [macro definitions here directly hgere]
-- $$
-- :::
-- ```
--
-- Method 2:
-- ```
-- ::: {.hidden}
-- {{< include _my_macros.qmd >}}
-- :::
-- ```
-- where `_my_macros_qmd` contains the `$$` math delimiters.
--
-- Method 3:
-- ::: {.hidden}
-- {{< include _my_macros.tex >}}
-- :::
-- ```
-- where `_my_macros_tex` does not contain math delimiters.
--
-- This filter will convert these to something like this:
-- ```
-- <span style="display: none">
-- \(
-- [macro definitions]
-- \)
-- </span>
-- ```
function Div(el)
  local block
  local math_tex
  local raw_html
  if el.classes:includes("hidden") then
    block = el.content[1]
    if block and block.t == "RawBlock" and block.format == "tex" then
      -- hidden math inserted via pure tex in .tex file
      local math_text = block.text
      local raw_html =
	'<span style="display: none">\n' ..
	'\\(\n' .. math_text .. '\n\\)\n' ..
	'</span>'
      return pandoc.RawBlock('html', raw_html)
    elseif block and block.t == "Para" and
      block.content[1] and block.content[1].t == "Math" then
      math_text = block.content[1].text
      -- hidden math inserted via DisplayMath in .qmd file
      raw_html =
	'<span style="display: none">\n' ..
	'\\(' .. math_text .. '\\)\n' ..
	'</span>'
      return pandoc.RawBlock('html', raw_html)
    end
  end
end
