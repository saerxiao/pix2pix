-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'fast_neural_style.PerceptualCriterion'
local percept_utils = require 'fast_neural_style.utils'
local customModels = require 'fast_neural_style.models'

opt = {
  DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
  name = 'facade_uncgan_wgan',              -- name of the experiment, should generally be passed on the command line
  batchSize = 16,          -- # images in batch
  loadSize = 256,         -- scale images to this size
  fineSize = 256,         --  then crop to this size
  ngf = 64,               -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  input_nc = 3,           -- #  of input image channels
  output_nc = 3,          -- #  of output image channels
  niter = 4000,            -- #  of iter at starting learning rate
  lr = 0.001, -- 0.0002            -- initial learning rate for adam
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  flip = 1,               -- if flip the images for data argumentation
  display = 1,            -- display samples while training. 0 = false
  display_id = 10,        -- display window id.
  display_plot = 'errContent',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  which_direction = 'AtoB',    -- AtoB or BtoA
  phase = 'train',             -- train, val, test, etc
  preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
  nThreads = 2,                -- # threads for loading data
  save_epoch_freq = 100,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
  save_latest_freq = 5000,    -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
  print_freq = 1,            -- print the debug information every print_freq iterations
  --display_freq = 100,          -- display the current results every display_freq iterations
  save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
  continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
  pretrain_netG = './checkpoints/mri_percept_myunet/50_net_G.t7',
  serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
  serial_batch_iter = 1,       -- iter into serial image list
  checkpoints_dir = './checkpoints', -- models are saved here
  cudnn = 1,                         -- set to 0 to not use cudnn
  condition_GAN = 0,                 -- set to 0 to use unconditional discriminator
  wGAN = 1,          -- set to 1 to use wGAN
  clamp = 0.01,      -- only used when wGAN = 1
  n_critic = 5,      -- only used when wGAN = 1 
  use_GAN = 1,                       -- set to 0 to turn off GAN term
  --use_L1 = 1,                        -- set to 0 to turn off L1 term
  which_model_netD = 'basic', -- selects model to use for netD
  which_model_netG = 'unet',  -- selects model to use for netG
  n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
  lambda = 50,   --100            -- weight on L1 term in objective
  -- opts added by me
  custom_data = false,      -- add custom logic in the dataset:getByClass
  validate_freq = 500,      -- run validation every validate_freq interations, set -1 to turn off validation
  content_loss = '',   -- L1|percept  
  pretrain_iters = 0,      -- train the network with only content loss for pertrain_iters iterations
  content_off_afterpertrain = 0,  -- set 1 to trun off content loss after pretrain_iters iterations
  ave_loss_freq = 50,         -- compute the average the training loss for every ave_loss_freq iterations
  -- percept options
  loss_network = 'percept-models/vgg16.t7',  -- loss network for computing the perceptual loss
  content_weights = '1.0',
  content_layers = '2',
  -- custom model options
  arch = 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3',   -- only used if which_model_netG = 'custom'
  padding_type = 'zero',  -- reflect-start|zero
  use_instance_normalization = 0,
  notanh = 0,  -- set 1 to remove the last Tanh() in the generator
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local content_loss = opt.content_loss
local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

local data_val
if opt.validate_freq > 0 then
  local opt_val = util.cloneTable(opt)
  opt_val.phase = 'val'  -- valid
  opt_val.serial_batches = 1
  data_val = data_loader.new(opt.nThreads, opt_val)
  print("Validate dataset size: ", data_val:size())
end

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf, opt.use_instance_normalization==1, opt.notanh==1)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "custom" then netG = customModels.build_model(opt)
    elseif opt.which_model_netG == "custom_unet" then netG = defineG_custom_unet()
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf, opt.fineSize, opt.use_instance_normalization == 1)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D, opt.fineSize, opt.use_instance_normalization == 1)
    elseif opt.which_model_netD == "single_output" then netD = defineD_single_output(input_nc_tmp, output_nc, ndf, opt.fineSize, opt.use_instance_normalization == 1)
    elseif opt.which_model_netD == "single_output_fc" then netD = defineD_single_output_fc(input_nc_tmp, output_nc, ndf, opt.fineSize, opt.use_instance_normalization == 1)
    else error("unsupported netD model")
    end
   
    if opt.wGAN then
      netD = netD:add(nn.Mean(1))
    end
 
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
  if opt.pretrain_netG then
    print('loading pretrianed netG')
    netG = util.load(opt.pretrain_netG, opt)
    print('define model netD...')
    netD = defineD(input_nc, output_nc, ndf)
  else
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
  end
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)

local criterion = nn.BCECriterion()
local criterionAE
if opt.content_loss == 'L1' then
  criterionAE = nn.AbsCriterion()
elseif opt.content_loss == 'percept' then
  opt.content_layers, opt.content_weights =
    percept_utils.parse_layers(opt.content_layers, opt.content_weights)
  local loss_net = torch.load(opt.loss_network)
    local crit_args = {
      cnn = loss_net,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
    }
    criterionAE = nn.PerceptualCriterion(crit_args)
end

---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errContent = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); 
  if criterionAE then criterionAE:cuda() end
   print('done')
else
	print('running model on CPU')
end

local dtype = netG:type()
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end


function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
   
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake
    fake_B = netG:forward(real_A)
    
    if opt.condition_GAN==1 then
      fake_AB = torch.cat(real_A,fake_B,2)
    else
      fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end

    local errD_real
    if opt.wGAN == 1 then
      errD_real = output:clone()
      netD:backward(real_AB, label)
    else
      errD_real = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      netD:backward(real_AB, df_do)
    end
    
    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake
    if opt.wGAN == 1 then
      errD_fake = output:clone()
      netD:backward(fake_AB, label)
    else
      errD_fake = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      netD:backward(fake_AB, df_do)
    end
    
    if opt.wGAN == 1 then
      errD = (errD_real - errD_fake):abs():mean()
      parametersD:clamp(-opt.clamp, opt.clamp)
      return errD, -gradParametersD
    else
      errD = (errD_real + errD_fake)/2
      return errD, gradParametersD
    end
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    if opt.use_GAN==1 then
      local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
      local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
      if opt.gpu>0 then 
       	label = label:cuda();
      end
      if opt.wGAN then
        errG = 0  -- errG is not meaningful in this case
        label:fill(fake_label)
        df_dg = netD:updateGradInput(fake_AB, label):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
      else
        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
      end
    else
        errG = 0
    end
   
    -- unary loss (content loss)
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end
    if opt.content_loss == 'L1' then
      errContent = criterionAE:forward(fake_B, real_B)
      df_do_AE = criterionAE:backward(fake_B, real_B)
    elseif opt.content_loss == 'percept' then
      local target = {content_target=real_B}
      errContent = criterionAE:forward(fake_B, target)
      df_do_AE = criterionAE:backward(fake_B, target)
    else
      errContent = 0
    end
   
    --print("df_dg:norm() / df_do_AE:norm()", df_dg:norm(), df_do_AE:norm(), df_dg_norm()/df_do_AE:norm())
    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    
    return errG, gradParametersG
end

local function validate()
  -- load real
  local real_data, data_path = data_val:getBatch()
  local real_A = real_data[{ {}, idx_A, {}, {} }]:type(dtype)
  local real_B = real_data[{ {}, idx_B, {}, {} }]:type(dtype)
  local real_AB
  if opt.condition_GAN==1 then
    real_AB = torch.cat(real_A,real_B,2)
  else
    real_AB = real_B -- unconditional GAN, only penalizes structure in B
  end

  -- create fake
  local fake_B = netG:forward(real_A)
  local fake_AB
  if opt.condition_GAN==1 then
    fake_AB = torch.cat(real_A,fake_B,2)
  else
    fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
  end

  local errD, errG = 0, 0
  if opt.use_GAN == 1 then
    if opt.wGAN == 1 then
      local errD_real = netD:forward(real_AB):clone()
      local errD_fake = netD:forward(fake_AB):clone()
      errD = (errD_real - errD_fake):abs():mean()
    else
      local output = netD:forward(real_AB)
      local label = torch.FloatTensor(output:size()):fill(real_label):type(dtype)
      local errD_real = criterion:forward(output, label)
      output = netD:forward(fake_AB)
      label:fill(fake_label)
      local errD_fake = criterion:forward(output, label)
      errD = (errD_real + errD_fake) / 2

      label = torch.FloatTensor(output:size()):fill(real_label):type(dtype) -- fake labels are real for generator cost
      errG = criterion:forward(output, label)
    end
  end

  local errContent = 0
  if content_loss == 'L1' then
    errContent = criterionAE:forward(fake_B, real_B)
  elseif content_loss == 'percept' then
    local target = {content_target=real_B}
    errContent = criterionAE:forward(fake_B, target)
  end

  return errD, errG, errContent
end



-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
--opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
--for k, v in ipairs(opt.display_plot) do
--    if not util.containsValue({"errG", "errD", "errContent"}, v) then 
--        error(string.format('bad display_plot value "%s"', v)) 
--    end
--end

-- display plot config
--local plot_config = {
--  title = "Loss over time",
--  labels = {"epoch", unpack(opt.display_plot)},
--  ylabel = "loss",
--}

-- display plot vars
local plot_data = {}
local plot_win
local loss_history = {errD={}, errG={}, errContent={}}
local errG_hist, errD_hist, errContent_hist = {}, {}, {}
local val_history = {ts={}, errD={}, errG={}, errContent={}}
local use_GAN = opt.use_GAN
local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()

	  if counter < opt.pretrain_iters then
      opt.use_GAN = 0
    else
	    opt.use_GAN = use_GAN
      if opt.content_off_afterpertrain == 1 then
        opt.content_loss = ''
      end
    end
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then 
          if opt.wGAN == 1 then
            for ii = 1, opt.n_critic do
              optim.adam(fDx, parametersD, optimStateD)
            end 
          else
            optim.adam(fDx, parametersD, optimStateD) 
          end
        end
        
        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        if counter > 0  and counter % opt.ave_loss_freq == 0 then
          if next(errG_hist) then
            table.insert(loss_history.errG, torch.Tensor(errG_hist):mean())
          else 
            table.insert(loss_history.errG, -1)
          end  
          errG_hist = {}
          if next(errD_hist) then
            table.insert(loss_history.errD, torch.Tensor(errD_hist):mean())
          else
            table.insert(loss_history.errD, -1)
          end
          errD_hist = {}
          table.insert(loss_history.errContent, torch.Tensor(errContent_hist):mean())
	        errContent_hist = {}
        else
          if counter > opt.pretrain_iters then
            table.insert(errG_hist, errG)
	          table.insert(errD_hist, errD)
          end
	        table.insert(errContent_hist, errContent)
	      end

	-- display
        counter = counter + 1
        --if counter % opt.display_freq == 0 and opt.display then
        --    createRealFake()
        --    if opt.preprocess == 'colorization' then 
        --        local real_A_s = util.scaleBatch(real_A:float(),100,100)
        --        local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
        --        local real_B_s = util.scaleBatch(real_B:float(),100,100)
        --        disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
        --        disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
        --        disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
        --    else
        --        disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
        --        disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
        --        disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target'})
        --    end
        --end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()), 3), util.deprocess(real_B[i2]:float()), 3)
                        else image_out = torch.cat(image_out, torch.cat(torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), util.deprocess(real_B[i2]:float()), 3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            --local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errContent=errContent and errContent or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrContent: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errContent))
           
            --local plot_vals = { epoch + curItInBatch / totalItInBatch }
            --for k, v in ipairs(opt.display_plot) do
            --  if loss[v] ~= nil then
            --   plot_vals[#plot_vals + 1] = loss[v] 
            --- end
            --end

            -- update display plot
            --if opt.display then
            --  table.insert(plot_data, plot_vals)
            --  plot_config.win = plot_win
            --  plot_win = disp.plot(plot_data, plot_config)
            --end
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
          print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
          torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
          if opt.use_GAN == 1 and counter > opt.pretrain_iters then
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
          end
          torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'loss.t7'), loss_history)
          local checkpoint_opt = {epoch=epoch, iter=counter, opt=opt, optimStateD=optimStateD, optimStateG=optimStateG}
          torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_opt.t7'), checkpoint_opt) 
        end
  
  -- save model when the pretrain is complete
  if counter == opt.pretrain_iters then
    print(string.format('save pretrained model: epoch %d, iters %d', epoch, counter))
    torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'pretrain_net_G.t7'), netG:clearState())
  end

	-- validate
	if opt.validate_freq > 0 and counter % opt.validate_freq == 0 then
	  netG:evaluate()
    local validate_size = 100 -- data_val:size()
	  local N_val_batches = math.ceil(validate_size / opt.batchSize)
    local errD_val, errG_val, errContent_val = 0, 0, 0
	  for i = 1, N_val_batches do
      local errD_val_b, errG_val_b, errContent_val_b = validate()
      errD_val = errD_val + errD_val_b
      errG_val = errG_val + errG_val_b
      errContent_val = errContent_val + errContent_val_b
    end
    errD_val, errG_val, errContent_val = errD_val / N_val_batches, errG_val / N_val_batches, errContent_val / N_val_batches
    if counter <= opt.pretrain_iters then errD_val, errG_val = -1, -1 end
    print(string.format("validate: iter = %d, errD = %.4f, errG = %.4f, errContent = %.4f", counter, errD_val, errG_val, errContent_val))
    table.insert(val_history.ts, counter)
    table.insert(val_history.errD, errD_val)
    table.insert(val_history.errG, errG_val)
    table.insert(val_history.errContent, errContent_val)
    torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'loss_val.t7'), val_history)
    netG:training()
	end        
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
      torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
      if opt.use_GAN == 1 and counter > opt.pretrain_iters then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
      end
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end
