require 'nngraph'
require 'fast_neural_style.InstanceNormalization'

function defineG_encoder_decoder(input_nc, output_nc, ngf)
    local netG = nil 
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})

    return netG
end

local function convModule(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, depth)
  if not depth then depth = 2 end
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
  local n1, n2 = nInputPlane, nOutputPlane
  for i = 1, depth do
    c = c
        - nn.SpatialConvolutionMM(n1, n2,kW,kH,dW,dH,padW,padH)
        - nn.SpatialBatchNormalization(nOutputPlane)
        - nn.ReLU()
    n1 = n2
  end
  return c
end

function defineG_custom_unet()
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,3,32,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,32,64,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,64,128,3,3,1,1,1,1) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule(pool3,128,256,3,3,1,1,1,1) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule(pool4,256,512,3,3,1,1,1,1) -- 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},512+256,256,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},256+128,128,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},128+64,64,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},64+32,32,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(32,3,1,1,1,1,0,0)
  local g = nn.gModule({input},{last})
  return g
end

function defineG_unet(input_nc, output_nc, ngf, use_instance_normalization, notanh)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      e2 = e2 - nn.InstanceNormalization(ngf * 2)
    else
      e2 = e2 - nn.SpatialBatchNormalization(ngf * 2)
    end
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      e3 = e3 - nn.InstanceNormalization(ngf * 4)
    else
      e3 = e3 - nn.SpatialBatchNormalization(ngf * 4)
    end
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      e4 = e4 - nn.InstanceNormalization(ngf * 8)
    else
      e4 = e4 - nn.SpatialBatchNormalization(ngf * 8)
    end
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      e5 = e5 - nn.InstanceNormalization(ngf * 8)
    else
      e5 = e5 - nn.SpatialBatchNormalization(ngf * 8)
    end
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      e6 = e6 - nn.InstanceNormalization(ngf * 8)
    else
      e6 = e6 - nn.SpatialBatchNormalization(ngf * 8)
    end
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
    if use_instance_normalization then
      e7 = e7 - nn.InstanceNormalization(ngf * 8)
    else
      e7 = e7 - nn.SpatialBatchNormalization(ngf * 8)
    end
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d1_ = d1_ - nn.InstanceNormalization(ngf * 8)
    else
      d1_ = d1_ - nn.SpatialBatchNormalization(ngf * 8)
    end
    d1_ = d1_ - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d2_ = d2_ - nn.InstanceNormalization(ngf * 8)
    else
      d2_ = d2_ - nn.SpatialBatchNormalization(ngf * 8)
    end
    d2_ = d2_ - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d3_ = d3_ - nn.InstanceNormalization(ngf * 8)
    else
      d3_ = d3_ - nn.SpatialBatchNormalization(ngf * 8)
    end
    d3_ = d3_ - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d4_ = d4_ - nn.InstanceNormalization(ngf * 8)
    else
      d4_ = d4_ - nn.SpatialBatchNormalization(ngf * 8)
    end
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d5_ = d5_ - nn.InstanceNormalization(ngf * 4)
    else
      d5_ = d5_ - nn.SpatialBatchNormalization(ngf * 4)
    end
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d6_ = d6_ - nn.InstanceNormalization(ngf * 2)
    else
      d6_ = d6_ - nn.SpatialBatchNormalization(ngf * 2)
    end
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1)
    if use_instance_normalization then
      d7_ = d7_ - nn.InstanceNormalization(ngf)
    else
      d7_ = d7_ - nn.SpatialBatchNormalization(ngf)
    end
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1
    if notanh then
      print("remove notanh")
      o1 = d8
    else
      print("add notanh")
      o1 = d8 - nn.Tanh()
    end
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineG_unet_128(input_nc, output_nc, ngf)
    -- Two layer less than the default unet to handle 128x128 input
    local netG = nil
    -- input is (nc) x 128 x 128
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 64 x 64
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 32 x 32
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 16 x 16
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e6} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e5} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e4} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e3} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e2} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e1} - nn.JoinTable(2)
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x128 x 128
    
    local o1 = d7 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineD_basic(input_nc, output_nc, ndf, N, use_instance_normalization)
    n_layers = 3
    return defineD_n_layers(input_nc, output_nc, ndf, n_layers, N, use_instance_normalization)
end

local function create_D_layers(input_nc, output_nc, ndf, n_layers, N, use_instance_normalization)
  print("N", N)
  if n_layers==0 then
    print("call defineD_pixelGAN")
    return defineD_pixelGAN(input_nc, output_nc, ndf, use_instance_normalization), N
  else

    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    N = math.floor(N/2)

    local nf_mult = 1
    local nf_mult_prev = 1
    for n = 1, n_layers-1 do
      nf_mult_prev = nf_mult
      nf_mult = math.min(2^n,8)
      netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
      if use_instance_normalization then
        netD:add(nn.InstanceNormalization(ndf * nf_mult))
      else
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult))
      end
      netD:add(nn.LeakyReLU(0.2, true))
      N = math.floor(N/2)
    end

    -- state size: (ndf*M) x N x N
    nf_mult_prev = nf_mult
    nf_mult = math.min(2^n_layers,8)
    netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
    N = N - 1
    if use_instance_normalization then
      netD:add(nn.InstanceNormalization(ndf * nf_mult))
    else
      netD:add(nn.SpatialBatchNormalization(ndf * nf_mult))
    end
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*M*2) x (N-1) x (N-1)
    netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
    N = N - 1
    -- state size: 1 x (N-2) x (N-2)

    return netD, N
  end
end

function defineD_single_output(input_nc, output_nc, ndf, input_size)
  local n_layers = 3
  local netD, img_size = create_D_layers(input_nc, output_nc, ndf, n_layers, input_size)
  
  -- flattern
  print("img_size = ", img_size)
  local flatten_size = img_size * img_size
  netD:add(nn.Reshape(flatten_size))
  netD:add(nn.Linear(flatten_size, 1))
  netD:add(nn.Sigmoid())

  return netD
end

function defineD_single_output_fc(input_nc, output_nc, ndf, input_size)
  local n_layers = 3
  local netD, img_size = create_D_layers(input_nc, output_nc, ndf, n_layers, input_size)

  -- flattern
  print("img_size = ", img_size)
  local fc_size = 256
  local flatten_size = img_size * img_size
  netD:add(nn.Reshape(flatten_size))
  netD:add(nn.Linear(flatten_size, fc_size)):add(nn.ReLU())
  netD:add(nn.Linear(fc_size, 1))
  netD:add(nn.Sigmoid())

  return netD
end

-- rf=1
function defineD_pixelGAN(input_nc, output_nc, ndf, use_instance_normalization)
    local netD = nn.Sequential()
    
    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    if use_instance_normalization then
      netD:add(nn.InstanceNormalization(ndf * 2))
    else
      netD:add(nn.SpatialBatchNormalization(ndf * 2))
    end
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    netD:add(nn.Sigmoid())
    -- state size: 1 x 256 x 256
        
    return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers, input_size, use_instance_normalization)
  local netD = create_D_layers(input_nc, output_nc, ndf, n_layers, input_size, use_instance_normalization)
  netD:add(nn.Sigmoid())
  -- state size: 1 x (N-2) x (N-2)
        
  return netD
end
