
local results_dir = 'result'
local volIdfile = 'trainIdList.txt'
local volIds = {}

-- make webpage
local function makeWebpage(models, modelsdir, axis)
  io.output(paths.concat(results_dir, string.format('compare_%s.html', axis)))
  
  io.write('<table style="text-align:center;">')
  local header = '<tr><td>Volumn Id #</td><td>Ground Truth</td>'
  for k=1, #models do
    header = string.format('%s<td>%s</td>', header, models[k])
  end
  header = string.format('%s</tr>', header)
  io.write(header)
  
  for i=1, #volIds do
    --print(volIds[i])
    io.write('<tr>')
    io.write('<td>' .. volIds[i] .. '</td>')
    io.write(string.format('<td><img src="./%s/images/%s_%s.png"/></td>', modelsdir[1], volIds[i], axis))
    for k=1, #modelsdir do
      io.write(string.format('<td><img src="./%s/images/%s_%s_gen.png"/></td>', modelsdir[k], volIds[i], axis))
    end
    io.write('</tr>')
  end

  io.write('</table>')
end

local function getVolIds(volIdfile)
  print(volIdfile)
  local file = io.open(volIdfile)
  if file then
    for line in file:lines() do
      table.insert(volIds, line)      
    end
  end
end

local models = {'mri_percept_myunet', 'mri_myunet_finetune_percept_on_w50'}
local modelsdir = {'mri_percept_myunet/epoch50', 'mri_myunet_finetune_percept_on_w50/epoch90'}
getVolIds(string.format("%s/%s/IdList.txt", results_dir, modelsdir[1]))
makeWebpage(models, modelsdir, 'N')
makeWebpage(models, modelsdir, 'W')
makeWebpage(models, modelsdir, 'H')
