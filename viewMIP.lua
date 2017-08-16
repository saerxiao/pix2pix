require 'image'
require 'lfs'
require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {
  backend = 'cuda',
  gpu = 1,
  cudnn = 1,
  modelId = 'mri_cgan',
  epochIter = 'epoch200',
  checkpointFile = '200_net_G.t7',
  plotSlice = false,
  batchsize = 4,
  selectVolId = nil   -- 'IXI291'
}

local dir = "/data/mri/data/multi/valid"
local dtype, use_cudnn = util.setup_gpu(opt.gpu-1, opt.backend, opt.cudnn == 1)

local model
local function getModel()
  local checkpointfile = string.format("checkpoints/%s/%s", opt.modelId, opt.checkpointFile)
  model = util.load(checkpointfile, opt)
  model:evaluate()
end

local outputdir, slicedir

local function createOutputdir()
  outputdir = string.format("/data/pix2pix/result/%s/%s", opt.modelId, opt.epochIter)
  if opt.selectVolId then
    outputdir = outputdir .. '-' .. opt.selectVolId
  end
  paths.mkdir(outputdir)
  slicedir = string.format("%s/slice", outputdir)
  if opt.plotSlice then paths.mkdir(slicedir) end
  return outputdir, slicedir
end

local function getVolumnMap()
  local t = {}
  for file in lfs.dir(dir) do
    if string.match(file, 'png') and not string.match(file, "mask") then
      local volId = string.match(file, "(.*)-.*")
      local slices = t[volId]
      if not slices then
        slices = {}
        t[volId] = slices
      end
      table.insert(slices, file)
    end
  end
  return t
end

local function compareIdx(a, b)
  local aId = tonumber(string.match(a, ".*-(.*).png"))
  local bId = tonumber(string.match(b, ".*-(.*).png"))
  return aId < bId
end

local function transform(slices, prefix, sliceprefix)
  table.sort(slices, compareIdx)
  local input = torch.Tensor():resize(#slices, 3, 256, 256)
  local gtrueth = torch.Tensor():resizeAs(input)
  for i=1, #slices do
    input[i] = image.load(string.format("%s/%s", dir, slices[i]))
    local sliceId = tonumber(string.match(slices[i], ".*-(.*).png"))
    local tfile = string.format("%s/%s_mask.png", dir, slices[i]:sub(1, -5))
    local mask = image.load(tfile)
    gtrueth[i][1]:copy(mask[1])
    gtrueth[i][2]:copy(mask[1])
    gtrueth[i][3]:copy(mask[1])
  end
  --print(input:min(), input:max())
  input = util.preprocess_batch(input:float()):type(dtype)
  --print(input:min(), input:max())
  local output_gen = input.new():resizeAs(input)
  local bstart = 1
  while bstart <= #slices do
    local bend = bstart + opt.batchsize
    if bend > #slices then bend = #slices end
    output_gen[{{bstart, bend}}] = model:forward(input[{{bstart, bend}}])
    bstart = bend + 1
  end
  output_gen = util.deprocess_batch(output_gen):float()
  if opt.plotSlice then
    for i=1, output_gen:size(1) do
      local sliceId = tonumber(string.match(slices[i], ".*-(.*).png"))
      local plotname = string.format("%s_%d.png", sliceprefix, sliceId)
      local image3 = torch.cat(torch.cat(util.deprocess(input[i]:float()), output_gen[i],3), gtrueth[i],3) 
      image.save(plotname, image3)
    end
  end
  output_gen = output_gen:transpose(1,2)
  local mN, mH, mW = output_gen:max(2):squeeze(), output_gen:max(3):squeeze(), output_gen:max(4):squeeze()
  gtrueth = gtrueth:transpose(1,2)
  local mN_t, mH_t, mW_t = gtrueth:max(2):squeeze(), gtrueth:max(3):squeeze(), gtrueth:max(4):squeeze()
  image.save(string.format("%s_N.png", prefix), torch.cat(mN, mN_t, 3))
  image.save(string.format("%s_H.png", prefix), torch.cat(mH, mH_t, 3))
  image.save(string.format("%s_W.png", prefix), torch.cat(mW, mW_t, 3))
end

function plotMIP(modelId, epoch, checkpointFile, selectVolId)
  opt.modelId = modelId and modelId or opt.modelId
  opt.epochIter = epoch and epoch or opt.epochIter
  opt.checkpointFile = checkpointFile and checkpointFile or opt.checkpointFile
  opt.selectVolId = selectVolId and selectVolId or opt.selectVolId
  getModel()
  createOutputdir()  
  print(outputdir)
  local volumns = getVolumnMap()
  local N = 1
  local id = 0
  local fileName = "trainIdList.txt"
  local file = io.open(fileName, 'a')
  for volId, slices in pairs(volumns) do
    file:write(string.format("%s\n", volId))
    if not selectVolId or volId == selectVolId then 
      local prefix = string.format("%s/%s", outputdir, volId)
      local sliceprefix = string.format("%s/%s", slicedir, volId)
      transform(slices, prefix, sliceprefix)
      id = id + 1
      print(id)
    end
    --if id == N then break end
  end
  file:close()
end

plotMIP()
