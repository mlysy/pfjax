--- Modify named CodeBlock for mkdocs
--
-- Modifies the first class name to append `title={{filename attribute}}`.
-- This is a hack for gfm output, since it doesn't process attributes. 
function process_title(el)
  if el.attr and el.attr.attributes and el.attr.attributes["filename"] then
    local title = el.attr.attributes["filename"]
    -- el.attr.attributes["title"] = title
    el.attr.attributes["filename"] = nil
    if #el.classes > 0 then
      el.classes[1] = el.classes[1] .. ' title="' .. title .. '"'
    end
    return el
  end
end

--- Convert plain-text CodeBlocks to ansi.
--
-- Fixes indentation, and also allows for colored rendering.
function process_output(el)
  if #el.classes == 0 then
    el.classes[1] = "ansi"
    return el
  end
end

return {
  {CodeBlock = process_title},
  {CodeBlock = process_output}
}

-- function CodeBlock(el)
--   if el.classes == pandoc.List() then
--     el.classes = pandoc.List({""})
--     -- quarto.log.output(el.classes)
--   end
--   return el
-- end
