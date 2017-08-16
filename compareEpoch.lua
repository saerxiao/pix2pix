require 'viewMIP'
require 'image'

local modelId = 'mri_cgan_percept'
local mipdirRoot = '/data/pix2pix/result'

-- print MIP for every 10 epochs
local function makeMIP()
  for i=0, 200, 10 do
    local epoch = i>0 and i or 1
    print("epoch", epoch)
    local checkpointFile = string.format("%d_net_G.t7", epoch)
    plotMIP(modelId, string.format("epoch%d", epoch), checkpointFile)
  end
end

local function getVolId(dir)
  local t = {}
  for file in lfs.dir(dir) do
    local volId = string.match(file, '(.*)_N.png')
    if volId and not t[volId] then
      table.insert(t, volId)
      t[volId] = #t
    end
  end
  return t
end

local function generateCompareImage(vols, whichp)
  for v = 1, #vols do
    local volId = vols[v]
    print(volId)
    local left
    for i = 0, 90, 10 do
      local epoch = i>0 and i or 1
      local fname = string.format("%s/%s/epoch%d/%s_%s.png", mipdirRoot, modelId, epoch, volId, whichp)
      local mip = image.load(fname)
      if epoch == 1 then
        left = mip[{{},{},{257, -1}}]
        left = torch.cat(left,mip[{{},{},{1,256}}], 2)
      else
        left = torch.cat(left, mip[{{},{},{1,256}}], 2)
      end
    end
    local right
    for i = 100, 200, 10 do
      local epoch = i
      local mip = image.load(string.format("%s/%s/epoch%d/%s_%s.png", mipdirRoot, modelId, epoch, volId, whichp))
      if right then
        right = torch.cat(right, mip[{{},{},{1,256}}], 2)
      else
        right = mip[{{},{},{1,256}}]
      end
    end
    local plotname = string.format("%s/%s/compare_epochs/%s_%s.png", mipdirRoot, modelId, volId, whichp)
    image.save(plotname, torch.cat(left, right, 3))
  end
end

makeMIP()
paths.mkdir(string.format("%s/%s/compare_epochs", mipdirRoot, modelId))
local vols = getVolId(string.format("%s/%s/epoch1", mipdirRoot, modelId))
print(#vols)
local volsname = string.format("%s/%s/vols.t7", mipdirRoot, modelId)
torch.save(volsname, vols)

--local vols = torch.load(volsname)
generateCompareImage(vols, 'N')
generateCompareImage(vols, 'H')
generateCompareImage(vols, 'W')
