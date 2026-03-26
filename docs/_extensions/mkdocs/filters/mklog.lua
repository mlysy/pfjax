-- function FloatRefTarget(el)
--   quarto.log.output("=== Handling FloatRefTarget ===")
--   quarto.log.output(el)
--   quarto.log.output("el.type:\n", el.type)
--   quarto.log.output("el.identifier:\n", el.identifier)
--   quarto.log.output("el.attributes:\n", el.attributes)
--   quarto.log.output("el.classes:\n", el.classes)
--   quarto.log.output("el.parent_id:\n", el.parent_id)
--   quarto.log.output("el.caption_long:\n", el.caption_long)
--   quarto.log.output("el.caption_short:\n", el.caption_short)
--   quarto.log.output("el.content:\n", el.content)
-- end

-- function Math(el)
--   quarto.log.output("=== Handling Math ===")
--   quarto.log.output(el)
-- end


-- function Cite(el)
--   quarto.log.output("=== Handling Cite ===")
--   quarto.log.output(el)
-- end

function Div(el)
  quarto.log.output("=== Handling Div ===")
  quarto.log.output(el)
end

-- function Pandoc(doc)
--   quarto.log.output("=== BLOCKS BEFORE WRITER ===")
--   for i, blk in ipairs(doc.blocks) do
--     quarto.log.output(blk)
--   end
--   return doc
-- end


-- function CodeBlock(el)
--   quarto.log.output("=== Handling CodeBlock ===")
--   quarto.log.output(el)
-- end

function RawBlock(el)
  quarto.log.output("=== Handling RawBlock ===")
  quarto.log.output(el)
end

function Math(el)
  quarto.log.output("=== Handling Math ===")
  quarto.log.output(el)
end


-- function Span(el)
--   quarto.log.output("=== Handling Span ===")
--   quarto.log.output(el)
-- end

-- function Para(el)
--   quarto.log.output("=== Handling Para ===")
--   quarto.log.output(el)
-- end

-- -- debug.lua
-- function Pandoc(doc)
--   quarto.log.output("Modules available in 'quarto':")
--   for k,v in pairs(quarto) do
--     quarto.log.output("  " .. k .. " -> " .. type(v))
--   end

--   quarto.log.output("Modules available in 'quarto._quarto':")
--   for k,v in pairs(quarto._quarto) do
--     quarto.log.output("  " .. k .. " -> " .. type(v))
--   end
-- end

-- -- explore-quarto.lua
-- local seen = {}

-- local function dump_table(t, prefix)
--   if seen[t] then return end
--   seen[t] = true
--   for k, v in pairs(t) do
--     local keypath = prefix .. "." .. k
--     quarto.log.output(keypath .. " -> " .. type(v))
--     if type(v) == "table" then
--       dump_table(v, keypath)
--     end
--   end
-- end

-- function Pandoc(doc)
--   quarto.log.output("==== Exploring 'quarto' namespace ====")
--   dump_table(quarto, "quarto")
--   return nil
-- end
