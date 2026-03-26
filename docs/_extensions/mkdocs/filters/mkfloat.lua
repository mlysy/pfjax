-- local predicate = function(float)
--   return quarto.doc.is_format("gfm")
-- end

-- local renderer = function(float)
--   local im = quarto.utils.match("Plain/[1]/Image")(float.content)
--   im.caption = {pandoc.Str("Default caption for the image")}
--   -- im.caption = quarto.utils.as_inlines(float.caption_long.content)
--   -- im.caption = float.caption_long.content
--   return pandoc.Para({im})
-- end

-- quarto._quarto.ast.add_renderer(
--   "FloatRefTarget",
--   predicate,
--   renderer
-- )

-- -- Only when writing GFM
-- quarto._quarto.ast.add_renderer("FloatRefTarget",
--   function() return quarto.format.isGitHubMarkdownOutput() end,
--   function(float)
--     -- float.content is the inner Pandoc element (Image/Para/Table/etc.)
--     local blocks = pandoc.List{ float.content }

--     -- Append a caption paragraph like "Figure 1: …" using Quarto’s crossref helpers
--     local caption_para = quarto.crossref.decorate_caption_with_crossref(float)
--     if caption_para then
--       blocks:insert(caption_para)
--     end

--     -- Wrap in a div so GitHub parses inner markdown
--     local attr = pandoc.Attr(float.identifier or "", {}, { markdown = "1" })
--     return pandoc.Div(blocks, attr)
--   end
-- )

--- Prepends "../" to `Image.src` so that `mkdocs build` gets the right path.
function Image(el)
  if not quarto._quarto.format.is_github_markdown_output then
    return nil
  end
  -- modify src
  el.src = "../" .. el.src
  -- quarto.log.output("el.caption_long:\n", el.caption_long)
  return el
end

--- Render `FloatRefTarget` for mkdocs output.
--
--  Usage:
--  
--  NOTES:
--
--  - Only `gmf` output format is supported.
--  - Modification of original renderer in `quarto-cli/src/resources/filters/customnodes/floatreftarget.lua`.
--
--  TODO:
--
--  - [ ] Modify `Image.src` from within the renderer instead of via `Image()`.
--  - [x] Deal with case when there's no caption provided.  Works as long as both `label` and `fig-cap` are empty.
--  - [x] Make sure this works with arbitrary images, not just those created with code.
quarto._quarto.ast.add_renderer(
  "FloatRefTarget",
  function(_)
    -- quarto.log.output("Format: ", FORMAT)
    return quarto._quarto.format.is_github_markdown_output()
  end,
  function(float)
    quarto.doc.crossref.decorate_caption_with_crossref(float)

    local caption_location = quarto.doc.crossref.cap_location(float)

    local open_block = pandoc.RawBlock("markdown", "<div id=\"" .. float.identifier .. "\" markdown=\"1\">\n")
    local close_block = pandoc.RawBlock("markdown", "</div>")
    local result = pandoc.Blocks({open_block})
    local insert_content = function()
      if pandoc.utils.type(float.content) == "Block" then
	result:insert(float.content)
      else
	result:extend(quarto.utils.as_blocks(float.content))
      end
    end
    local insert_caption = function()
      if pandoc.utils.type(float.caption_long) == "Block" then
	result:insert(float.caption_long)
      else
	result:insert(pandoc.Plain(quarto.utils.as_inlines(float.caption_long)))
      end
    end

    if caption_location == "top" then
      insert_caption()
      insert_content()
      result:insert(close_block)
    else
      insert_content()
      result:insert(pandoc.RawBlock("markdown", "\n"))
      insert_caption()
      result:insert(pandoc.RawBlock("markdown", "\n"))
      result:insert(close_block)
    end
    return result
  end
)
