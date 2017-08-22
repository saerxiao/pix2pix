require 'gnuplot'

local modelId = 'facade_uncgan_wgan'
local trainlossname = string.format("checkpoints/%s/loss.t7", modelId)
local trainloss = torch.load(trainlossname)
local validlossname = string.format("checkpoints/%s/loss_val.t7", modelId)
local validloss = torch.load(validlossname)

local function plotfigureOld(file, key)
  local name = string.format("checkpoints/%s/%s_train.png", modelId, key)
  gnuplot.pngfigure(name)
  gnuplot.title(string.format("%s %s, train", modelId, key))
  local vals = torch.Tensor(file[key])
  local mutilplier = 50
  vals = vals:view(-1, mutilplier):mean(2):squeeze()
  gnuplot.plot(torch.range(1,vals:size(1))*mutilplier, vals, '+')
  gnuplot.axis{0,'',0,''}
  gnuplot.plotflush()
end

local function plotfigureTrain(file, key, mutilplier)
  local name = string.format("checkpoints/%s/%s_train.png", modelId, key)
  gnuplot.pngfigure(name)
  gnuplot.title(string.format("%s %s, train", modelId, key))
  local vals = torch.Tensor(file[key])
  gnuplot.plot(torch.range(1,vals:size(1))*mutilplier, vals, '+')
  gnuplot.axis{0,'',0,''}
  gnuplot.plotflush()
end

local function plotfigureValid(file, key)
  local name = string.format("checkpoints/%s/%s_valid.png", modelId, key)
  gnuplot.pngfigure(name)
  gnuplot.title(string.format("%s %s, validate", modelId, key))
  local vals = torch.Tensor(file[key])
  gnuplot.plot(torch.Tensor(file.ts), vals, '+')
  --if yrange then gnuplot.raw(yrange) end
  gnuplot.axis{0,'',0,''}
  gnuplot.plotflush()
end

-- for model mri_cgan, mri_cgan_L1
--plotfigureOld(trainloss, 'errG')
--plotfigureOld(trainloss, 'errD')
--plotfigureTrain('errL1')

plotfigureTrain(trainloss, 'errG', 50)
plotfigureTrain(trainloss, 'errD', 50)
plotfigureTrain(trainloss, 'errContent', 50)

plotfigureValid(validloss, 'errG')
plotfigureValid(validloss, 'errD')
plotfigureValid(validloss, 'errContent')



