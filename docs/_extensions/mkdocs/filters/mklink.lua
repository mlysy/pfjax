--- Convert .qmd links to .md for mkdocs.
function Link(el)
  if el.target:match("%.qmd$") then
    el.target = el.target:gsub("%.qmd$", ".md")
  end
  return el
end
