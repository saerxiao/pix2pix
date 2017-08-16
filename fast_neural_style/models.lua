require 'nn'
require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'


local M = {}


local function build_conv_block(dim, padding_type, use_instance_norm)
  local conv_block = nn.Sequential()
  local p = 0
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  elseif padding_type == 'zero' then
    p = 1
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  if use_instance_norm == 1 then
    conv_block:add(nn.InstanceNormalization(dim))
  else
    conv_block:add(nn.SpatialBatchNormalization(dim))
  end
  conv_block:add(nn.ReLU(true))
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  if use_instance_norm == 1 then
    conv_block:add(nn.InstanceNormalization(dim))
  else
    conv_block:add(nn.SpatialBatchNormalization(dim))
  end
  return conv_block
end


local function build_res_block(dim, padding_type, use_instance_norm)
  local conv_block = build_conv_block(dim, padding_type, use_instance_norm)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)
  if padding_type == 'none' or padding_type == 'reflect-start' then
    concat:add(nn.ShaveImage(2))
  else
    concat:add(nn.Identity())
  end
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end


function M.build_model(opt)
  local arch = opt.arch:split(',')
  local prev_dim = 3
  local model = nn.Sequential()
  
  for i, v in ipairs(arch) do
    local first_char = string.sub(v, 1, 1)
    local layer, next_dim
    local needs_relu = true
    local needs_bn = true
    if first_char == 'c' then
      -- Convolution
      local f = tonumber(string.sub(v, 2, 2)) -- filter size
      local p = (f - 1) / 2 -- padding
      local s = tonumber(string.sub(v, 4, 4)) -- stride
      next_dim = tonumber(string.sub(v, 6))
      if opt.padding_type == 'reflect' then
        model:add(nn.SpatialReflectionPadding(p, p, p, p))
        p = 0
      elseif opt.padding_type == 'replicate' then
        model:add(nn.SpatialReplicationPadding(p, p, p, p))
        p = 0
      elseif padding_type == 'none' then
        p = 0
      end
      layer = nn.SpatialConvolution(prev_dim, next_dim, f, f, s, s, p, p)
    elseif first_char == 'f' then
      -- Full convolution
      local f = tonumber(string.sub(v, 2, 2)) -- filter size
      local p = (f - 1) / 2 -- padding
      local s = tonumber(string.sub(v, 4, 4)) -- stride
      local a = s - 1 -- adjustment
      next_dim = tonumber(string.sub(v, 6))
      layer = nn.SpatialFullConvolution(prev_dim, next_dim,
                                        f, f, s, s, p, p, a, a)
    elseif first_char == 'd' then
      -- Downsampling (strided convolution)
      next_dim = tonumber(string.sub(v, 2))
      layer = nn.SpatialConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1)
    elseif first_char == 'U' then
      -- Nearest-neighbor upsampling
      next_dim = prev_dim
      local scale = tonumber(string.sub(v, 2))
      layer = nn.SpatialUpSamplingNearest(scale)
    elseif first_char == 'u' then
      -- Learned upsampling (strided full-convolution)
      next_dim = tonumber(string.sub(v, 2))
      layer = nn.SpatialFullConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1, 1, 1)
    elseif first_char == 'C' then
      -- Non-residual conv block
      next_dim = tonumber(string.sub(v, 2))
      layer = build_conv_block(next_dim, opt.padding_type, opt.use_instance_norm)
      needs_bn = false
      needs_relu = true
    elseif first_char == 'R' then
      -- Residual (non-bottleneck) block
      next_dim = tonumber(string.sub(v, 2))
      layer = build_res_block(next_dim, opt.padding_type, opt.use_instance_norm)
      needs_bn = false
      needs_relu = false
    end
    model:add(layer)
    if i == #arch then
      needs_relu = false
      needs_bn = false
    end
    if needs_bn then
      if opt.use_instance_norm == 1 then
        model:add(nn.InstanceNormalization(next_dim))
      else
        model:add(nn.SpatialBatchNormalization(next_dim))
      end
    end
    if needs_relu then
      model:add(nn.ReLU(true))
    end

    prev_dim = next_dim
  end

  --model:add(nn.Tanh())
  --model:add(nn.MulConstant(opt.tanh_constant))
  --model:add(nn.TotalVariation(opt.tv_strength))

  return model
end

local function disc1(opt)
  local model = nn.Sequential()
  local k, d, pad = 4, 2, 1
  local prev_dim = 3
  local img_size = opt.input_size
  local conv_dims = {32, 64, 64, 128,128}
  for _, dim in pairs(conv_dims) do
    model:add(nn.SpatialConvolutionMM(prev_dim, dim,k,k,d,d,pad,pad))
    model:add(nn.SpatialBatchNormalization(dim))
    model:add(nn.ReLU())
    prev_dim = dim
    img_size = img_size/d
  end
  local flatten_size = img_size * img_size * prev_dim
  model:add(nn.Reshape(flatten_size))
  model:add(nn.Linear(flatten_size, 512)):add(nn.ReLU())
  model:add(nn.Linear(512, 1))
  return model
end

function M.build_discriminator(opt)
  local model = nn.Sequential()
  model:add(nn.JoinTable(1))
  model:add(disc1(opt))
  model:add(nn.Sigmoid())
  return model
end

function M.build_discriminator2(opt)
  local model = nn.Sequential()
  model:add(nn.JoinTable(1))
  local k, d, pad = 4, 2, 1
  local prev_dim = 3
  local img_size = opt.input_size
  local conv_dims = {32, 64, 64, 128,128}
  for _, dim in pairs(conv_dims) do
    model:add(nn.SpatialConvolutionMM(prev_dim, dim,k,k,d,d,pad,pad))
    model:add(nn.SpatialBatchNormalization(dim))
    model:add(nn.ReLU())
    prev_dim = dim
    img_size = img_size/d
  end
  local flatten_size = img_size * img_size * prev_dim
  model:add(nn.Reshape(flatten_size))
  model:add(nn.Linear(flatten_size, 512)):add(nn.ReLU())
  model:add(nn.Linear(512, 1))
  model:add(nn.Sigmoid())
  return model
end

function M.build_discriminator_wgan(opt)
  local real_d = disc1(opt)
  real_d:add(nn.Mean(1))
  local gen_d = real_d:clone('weight', 'bias', 'gradWeight', 'gradBias')
  local model = nn.ParallelTable():add(real_d):add(gen_d)
  return model
end

return M
